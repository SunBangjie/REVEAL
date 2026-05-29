from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

LABEL_EXCLUDE_COLS = {
    "image_id",
    "entry_key",
    "source_json",
    "source_dir",
    "image_relpath",
    "image_path",
}

EPS = 1e-12


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def merged_confirmation_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    legacy = dict((cfg.get("measure") or {}))
    current = dict((cfg.get("confirmation") or {}))
    return {**legacy, **current}


def target_split_from_config(cfg: Mapping[str, Any], cli_target_split: str | None) -> str:
    confirm_cfg = merged_confirmation_config(cfg)
    return (
        cli_target_split
        or confirm_cfg.get("target_split")
        or (cfg.get("audit") or {}).get("target_split")
        or "audited"
    )


def infer_binary_attribute_columns(labels: Any) -> list[str]:
    secret_cols: list[str] = []
    for col in labels.columns:
        if col in LABEL_EXCLUDE_COLS:
            continue
        non_null = labels[col].dropna()
        if len(non_null) == 0:
            continue
        try:
            unique_vals = set(non_null.astype(int).unique().tolist())
        except Exception:
            continue
        if unique_vals.issubset({0, 1}):
            secret_cols.append(col)
    return secret_cols


def clamp_probability(p: float) -> float:
    return min(max(float(p), EPS), 1.0 - EPS)


def entropy_binary(p: float) -> float:
    p = clamp_probability(p)
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def local_kl_binary(post_p: float, prior_p: float) -> float:
    p = clamp_probability(post_p)
    q = clamp_probability(prior_p)
    return float(p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q)))


def pointwise_leakage_binary(y_true: int, post_p: float, prior_p: float) -> float:
    p = clamp_probability(post_p)
    q = clamp_probability(prior_p)
    if int(y_true) == 1:
        return float(np.log(p / q))
    return float(np.log((1.0 - p) / (1.0 - q)))


def cross_entropy_binary(y_true: int, pred_p: float) -> float:
    q = clamp_probability(pred_p)
    if int(y_true) == 1:
        return float(-np.log(q))
    return float(-np.log(1.0 - q))


def task_kl(baseline_p: float, prior_p: float) -> float:
    return local_kl_binary(baseline_p, prior_p)


def excess_lift(conditional_p: float, baseline_p: float) -> float:
    return float(clamp_probability(conditional_p) / clamp_probability(baseline_p))


def normalized_excess(excess_kl: float, total_kl: float) -> float:
    if total_kl <= EPS:
        return 0.0
    return float(min(max(excess_kl / total_kl, 0.0), 1.0))


def posterior_shift(conditional_p: float, baseline_p: float) -> float:
    return float(conditional_p - baseline_p)


def decomposition_residual(total_kl: float, task_kl_value: float, excess_kl_value: float) -> float:
    return float(total_kl - task_kl_value - excess_kl_value)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.square(w).sum())
    if denom <= 0:
        return 0.0
    num = float(w.sum()) ** 2
    return num / denom


def weighted_knn_posterior(
    sims: np.ndarray,
    labels: np.ndarray,
    *,
    tau: float,
    top_k: int,
    alpha: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    sims = np.asarray(sims, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(sims) == 0:
        p = 0.5
        return p, 0.0, np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)

    if top_k > 0 and len(sims) > top_k:
        order = np.argsort(-sims)[:top_k]
        sims = sims[order]
        labels = labels[order]
    else:
        order = np.arange(len(sims), dtype=np.int64)

    tau = max(float(tau), 1e-6)
    shifted = (sims - sims.max()) / tau
    raw = np.exp(np.clip(shifted, -80, 80))
    if raw.sum() <= 0:
        weights = np.ones_like(raw) / max(len(raw), 1)
    else:
        weights = raw / raw.sum()

    n_eff = effective_sample_size(weights)
    weighted_pos = float(np.dot(weights, labels))
    posterior = (float(alpha) + n_eff * weighted_pos) / (2.0 * float(alpha) + n_eff)
    return float(posterior), float(n_eff), weights, order


def prior_with_smoothing(labels: np.ndarray, alpha: float) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    return float((float(alpha) + labels.sum()) / (2.0 * float(alpha) + len(labels))) if len(labels) else 0.5


def baseline_from_pred_label(
    target_label: int | None,
    support_pred_labels: np.ndarray | None,
    support_labels: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, int, np.ndarray]:
    if target_label is None or support_pred_labels is None or len(support_pred_labels) != len(support_labels):
        return prior_with_smoothing(support_labels, alpha), 0, np.zeros(len(support_labels), dtype=bool)
    mask = np.asarray(support_pred_labels).astype(int) == int(target_label)
    matched = support_labels[mask]
    if len(matched) == 0:
        return prior_with_smoothing(support_labels, alpha), 0, mask
    return prior_with_smoothing(matched, alpha), int(len(matched)), mask


def build_joint_features(
    z_rows: np.ndarray,
    o_rows: np.ndarray,
    embed_weight: float,
    output_weight: float,
) -> np.ndarray:
    joined = np.concatenate([float(embed_weight) * z_rows, float(output_weight) * o_rows], axis=1)
    return _normalize_rows(joined)


def label_signature(image_ids: Sequence[str], labels: np.ndarray) -> str:
    pairs = sorted((str(i), int(v)) for i, v in zip(image_ids, np.asarray(labels).astype(int).tolist()))
    payload = "|".join(f"{image_id}:{value}" for image_id, value in pairs)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def top_support_ids(
    order: np.ndarray,
    image_ids: Sequence[str],
    weights: np.ndarray,
    max_items: int = 5,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, (idx, weight) in enumerate(zip(order[:max_items], weights[:max_items]), start=1):
        out.append({"rank": rank, "image_id": str(image_ids[int(idx)]), "weight": float(weight)})
    return out


def mean_top_similarity(sims: np.ndarray, order: np.ndarray, top_k: int) -> float:
    if len(order) == 0:
        return float("nan")
    use = order[: min(len(order), max(top_k, 1))]
    return float(np.mean(sims[use]))


def aggregate_excess_rate(excess_kl_series: Any, tau: float = 0.01) -> float:
    return float((excess_kl_series > tau).mean()) if len(excess_kl_series) else 0.0


def aggregate_excess_to_total_ratio(excess_kl_series: Any, total_kl_series: Any) -> float:
    total_sum = float(total_kl_series.sum())
    if total_sum <= EPS:
        return 0.0
    return float(min(max(float(excess_kl_series.sum()) / total_sum, 0.0), 1.0))


def aggregate_v_info_excess(y_true_series: Any, baseline_p_series: Any, conditional_p_series: Any) -> float:
    valid = y_true_series.notna()
    if valid.sum() == 0:
        return float("nan")
    y_true = y_true_series[valid].astype(int).to_numpy()
    baseline_p = baseline_p_series[valid].to_numpy(dtype=np.float64)
    conditional_p = conditional_p_series[valid].to_numpy(dtype=np.float64)
    ce_baseline = np.array([cross_entropy_binary(y_i, p_i) for y_i, p_i in zip(y_true, baseline_p)])
    ce_conditional = np.array([cross_entropy_binary(y_i, p_i) for y_i, p_i in zip(y_true, conditional_p)])
    return float(np.mean(ce_baseline) - np.mean(ce_conditional))


def normalize_attribute_name(name: str) -> str:
    value = str(name or "").strip().lower().replace("-", "_")
    value = re.sub(r"[^a-z0-9_\s]+", " ", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _safe_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(v) for v in values if str(v).strip())


def _float_or_none(value: Any) -> float | None:
    if value in {None, "", "null"}:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return None if np.isnan(parsed) else parsed


def _target_id_from_stage1_record(record: Mapping[str, Any]) -> str:
    for key in ("target_image_id", "image_id", "id"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def iter_stage1_attribute_items(record: Mapping[str, Any], field: str) -> list[tuple[str, dict[str, Any]]]:
    def _coerce_items(key: str) -> list[tuple[str, dict[str, Any]]]:
        return [(key, item) for item in (record.get(key, []) or []) if isinstance(item, dict)]

    if field == "both":
        out: list[tuple[str, dict[str, Any]]] = []
        for key in ("shared_attributes", "excess_secrets"):
            out.extend(_coerce_items(key))
        return out

    if field == "excess_secrets":
        explicit = _coerce_items("excess_secrets")
        if explicit:
            return explicit
        return _coerce_items("shared_attributes")

    return _coerce_items(field)


@dataclass(frozen=True)
class AttributeSpec:
    attribute_name: str
    description: str = ""
    source_field: str = ""
    specificity: float | None = None
    privacy_relevance: float | None = None
    task_relevance: float | None = None
    predicted_label_covers: str = ""
    positive_patterns: tuple[str, ...] = ()
    negative_patterns: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        attribute_name: str,
        payload: Mapping[str, Any] | None = None,
        *,
        source_field: str = "",
        predicted_label_covers: str = "",
    ) -> "AttributeSpec":
        payload = payload or {}
        return cls(
            attribute_name=str(attribute_name),
            description=str(payload.get("description", "") or ""),
            source_field=str(payload.get("source_field", source_field) or ""),
            specificity=_float_or_none(payload.get("specificity")),
            privacy_relevance=_float_or_none(payload.get("privacy_relevance", payload.get("relevance"))),
            task_relevance=_float_or_none(payload.get("task_relevance")),
            predicted_label_covers=str(payload.get("predicted_label_covers", predicted_label_covers) or ""),
            positive_patterns=_safe_tuple(payload.get("positive_patterns")),
            negative_patterns=_safe_tuple(payload.get("negative_patterns")),
        )


def build_attribute_spec_index(
    stage1_records: Sequence[Mapping[str, Any]],
    *,
    field: str = "shared_attributes",
) -> dict[tuple[str, str], AttributeSpec]:
    index: dict[tuple[str, str], AttributeSpec] = {}
    for record in stage1_records:
        target_id = _target_id_from_stage1_record(record)
        if not target_id:
            continue
        predicted_label_covers = str(record.get("predicted_label_covers", "") or "")
        for source_field, item in iter_stage1_attribute_items(record, field):
            name = normalize_attribute_name(item.get("name") or "")
            if not name:
                continue
            key = (str(target_id), name)
            spec = AttributeSpec.from_mapping(
                name,
                item,
                source_field=source_field,
                predicted_label_covers=predicted_label_covers,
            )
            previous = index.get(key)
            if previous is None:
                index[key] = spec
                continue
            index[key] = AttributeSpec(
                attribute_name=spec.attribute_name,
                description=spec.description or previous.description,
                source_field=spec.source_field or previous.source_field,
                specificity=max(
                    value for value in (previous.specificity, spec.specificity) if value is not None
                ) if (previous.specificity is not None or spec.specificity is not None) else None,
                privacy_relevance=max(
                    value for value in (previous.privacy_relevance, spec.privacy_relevance) if value is not None
                ) if (previous.privacy_relevance is not None or spec.privacy_relevance is not None) else None,
                task_relevance=max(
                    value for value in (previous.task_relevance, spec.task_relevance) if value is not None
                ) if (previous.task_relevance is not None or spec.task_relevance is not None) else None,
                predicted_label_covers=spec.predicted_label_covers or previous.predicted_label_covers,
                positive_patterns=tuple(dict.fromkeys([*previous.positive_patterns, *spec.positive_patterns])),
                negative_patterns=tuple(dict.fromkeys([*previous.negative_patterns, *spec.negative_patterns])),
            )
    return index


@dataclass(frozen=True)
class ConfirmationThresholds:
    min_posterior_shift: float = 0.02
    min_excess_lift: float = 1.10
    min_excess_kl: float = 0.01
    min_neighbor_support: int = 3
    min_effective_sample_size: float = 3.0
    min_baseline_effective_sample_size: float = 1.0
    min_attribute_specificity: float = 0.20
    max_attribute_task_relevance: float | None = None
    require_task_baseline: bool = True

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ConfirmationThresholds":
        confirm_cfg = merged_confirmation_config(cfg)
        op_cfg = cfg.get("operationalization") or {}

        max_task_rel = confirm_cfg.get(
            "max_attribute_task_relevance",
            op_cfg.get("max_task_relevance"),
        )
        if max_task_rel in {None, "", "null"}:
            max_task_rel = None

        min_neighbor_support = int(confirm_cfg.get("min_neighbor_support", confirm_cfg.get("min_external_support", 3)))
        min_effective_sample_size = float(
            confirm_cfg.get("min_effective_sample_size", max(3, min_neighbor_support))
        )
        min_baseline_effective_sample_size = float(
            confirm_cfg.get("min_baseline_effective_sample_size", 1.0)
        )

        return cls(
            min_posterior_shift=float(confirm_cfg.get("min_posterior_shift", 0.02)),
            min_excess_lift=float(confirm_cfg.get("min_excess_lift", 1.10)),
            min_excess_kl=float(confirm_cfg.get("min_excess_kl", confirm_cfg.get("excess_rate_threshold_nats", 0.01))),
            min_neighbor_support=min_neighbor_support,
            min_effective_sample_size=min_effective_sample_size,
            min_baseline_effective_sample_size=min_baseline_effective_sample_size,
            min_attribute_specificity=float(
                confirm_cfg.get("min_attribute_specificity", op_cfg.get("min_specificity", 0.20))
            ),
            max_attribute_task_relevance=None if max_task_rel is None else float(max_task_rel),
            require_task_baseline=bool(confirm_cfg.get("require_task_baseline", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_posterior_shift": float(self.min_posterior_shift),
            "min_excess_lift": float(self.min_excess_lift),
            "min_excess_kl": float(self.min_excess_kl),
            "min_neighbor_support": int(self.min_neighbor_support),
            "min_effective_sample_size": float(self.min_effective_sample_size),
            "min_baseline_effective_sample_size": float(self.min_baseline_effective_sample_size),
            "min_attribute_specificity": float(self.min_attribute_specificity),
            "max_attribute_task_relevance": self.max_attribute_task_relevance,
            "require_task_baseline": bool(self.require_task_baseline),
        }


@dataclass(frozen=True)
class ConfirmationResult:
    attribute_name: str
    status: str
    baseline_posterior: float
    conditional_posterior: float
    posterior_shift: float
    excess_lift: float
    excess_kl: float
    confidence: str
    support_score: float
    support_diagnostics: dict[str, Any] = field(default_factory=dict)
    support_failures: list[str] = field(default_factory=list)
    support_warnings: list[str] = field(default_factory=list)
    rationale: str = ""
    thresholds: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute_name": self.attribute_name,
            "confirmation_status": self.status,
            "status": self.status,
            "baseline_posterior": float(self.baseline_posterior),
            "conditional_posterior": float(self.conditional_posterior),
            "posterior_shift": float(self.posterior_shift),
            "excess_lift": float(self.excess_lift),
            "excess_kl_nats": float(self.excess_kl),
            "confidence": self.confidence,
            "support_score": float(self.support_score),
            "support_failures_json": json.dumps(self.support_failures, ensure_ascii=False),
            "support_warnings_json": json.dumps(self.support_warnings, ensure_ascii=False),
            "support_diagnostics_json": json.dumps(self.support_diagnostics, ensure_ascii=False, sort_keys=True),
            "rationale": self.rationale,
            "thresholds_json": json.dumps(self.thresholds, ensure_ascii=False, sort_keys=True),
        }


class ExcessLeakageConfirmer:
    def __init__(self, thresholds: ConfirmationThresholds | Mapping[str, Any] | None = None) -> None:
        if thresholds is None:
            self.thresholds = ConfirmationThresholds()
        elif isinstance(thresholds, ConfirmationThresholds):
            self.thresholds = thresholds
        else:
            self.thresholds = ConfirmationThresholds(**dict(thresholds))

    @staticmethod
    def _support_confidence_label(status: str, support_score: float) -> str:
        if status == "confirmed":
            return "strong"
        if support_score >= 0.75:
            return "moderate"
        if support_score >= 0.50:
            return "limited"
        return "low"

    @staticmethod
    def _baseline_available(baseline_mode: str, baseline_status: str, require_task_baseline: bool) -> bool:
        mode = str(baseline_mode or "NONE").strip().upper()
        status = str(baseline_status or "").strip().lower()
        if mode == "NONE":
            return not require_task_baseline
        if status.startswith("missing") or status.startswith("no_support"):
            return False
        return True

    def confirm_excess_leakage(
        self,
        z_i: np.ndarray | None,
        y_i: int | None,
        attribute_spec: AttributeSpec | str,
        config: Mapping[str, Any] | ConfirmationThresholds | None = None,
        *,
        baseline_posterior: float,
        conditional_posterior: float,
        excess_kl: float,
        n_external_support: int,
        effective_sample_size: float,
        baseline_effective_sample_size: float,
        baseline_mode: str,
        baseline_status: str,
        support_status: str = "ok",
        total_kl: float | None = None,
        task_kl_value: float | None = None,
        mean_support_similarity: float | None = None,
        baseline_mean_support_similarity: float | None = None,
        joint_mean_support_similarity: float | None = None,
        extra_diagnostics: Mapping[str, Any] | None = None,
    ) -> ConfirmationResult:
        thresholds = config if isinstance(config, ConfirmationThresholds) else self.thresholds
        if isinstance(config, Mapping):
            thresholds = ConfirmationThresholds(**{**thresholds.to_dict(), **dict(config)})
        if isinstance(attribute_spec, str):
            attribute_spec = AttributeSpec(attribute_name=str(attribute_spec))

        base_p = float(baseline_posterior)
        cond_p = float(conditional_posterior)
        shift = posterior_shift(cond_p, base_p)
        lift = excess_lift(cond_p, base_p)

        support_checks = {
            "baseline_available": self._baseline_available(
                baseline_mode,
                baseline_status,
                thresholds.require_task_baseline,
            ),
            "support_status_ok": str(support_status or "ok") == "ok",
            "neighbor_support_ok": int(n_external_support) >= int(thresholds.min_neighbor_support),
            "effective_sample_size_ok": float(effective_sample_size) >= float(thresholds.min_effective_sample_size),
            "baseline_support_ok": float(baseline_effective_sample_size) >= float(thresholds.min_baseline_effective_sample_size),
            "attribute_specificity_ok": (
                attribute_spec.specificity is None
                or float(attribute_spec.specificity) >= float(thresholds.min_attribute_specificity)
            ),
            "attribute_task_relevance_ok": (
                thresholds.max_attribute_task_relevance is None
                or attribute_spec.task_relevance is None
                or float(attribute_spec.task_relevance) <= float(thresholds.max_attribute_task_relevance)
            ),
        }
        evidence_checks = {
            "positive_shift": shift > 0.0,
            "posterior_shift_ok": shift >= float(thresholds.min_posterior_shift),
            "excess_lift_ok": lift >= float(thresholds.min_excess_lift),
            "excess_kl_ok": float(excess_kl) >= float(thresholds.min_excess_kl),
        }
        incremental_evidence = all(evidence_checks.values())
        hard_support_keys = {
            "baseline_available",
            "support_status_ok",
            "neighbor_support_ok",
        }
        support_failures = [
            name for name, passed in support_checks.items()
            if not passed and name in hard_support_keys
        ]
        support_warnings = [
            name for name, passed in support_checks.items()
            if not passed and name not in hard_support_keys
        ]
        support_score = (
            float(sum(1 for passed in support_checks.values() if passed)) / float(len(support_checks))
            if support_checks
            else 1.0
        )

        if incremental_evidence and not support_failures:
            status = "confirmed"
        elif incremental_evidence:
            status = "inconclusive"
        else:
            status = "rejected"

        confidence = self._support_confidence_label(status, support_score)
        rationale = build_confirmation_rationale(
            attribute_spec=attribute_spec,
            status=status,
            baseline_posterior=base_p,
            conditional_posterior=cond_p,
            posterior_shift_value=shift,
            excess_lift_value=lift,
            excess_kl_value=float(excess_kl),
            support_failures=support_failures,
            support_status=support_status,
            thresholds=thresholds,
        )

        diagnostics = {
            "baseline_mode": str(baseline_mode),
            "baseline_status": str(baseline_status),
            "support_status": str(support_status),
            "n_external_support": int(n_external_support),
            "effective_sample_size": float(effective_sample_size),
            "baseline_effective_sample_size": float(baseline_effective_sample_size),
            "mean_support_similarity": mean_support_similarity,
            "baseline_mean_support_similarity": baseline_mean_support_similarity,
            "joint_mean_support_similarity": joint_mean_support_similarity,
            "attribute_specificity": attribute_spec.specificity,
            "attribute_privacy_relevance": attribute_spec.privacy_relevance,
            "attribute_task_relevance": attribute_spec.task_relevance,
            "attribute_source_field": attribute_spec.source_field,
            "predicted_label_covers": attribute_spec.predicted_label_covers,
            "evidence_checks": evidence_checks,
            "support_checks": support_checks,
            "hard_support_failures": support_failures,
            "support_warnings": support_warnings,
            "target_label": None if y_i is None else int(y_i),
            "embedding_present": bool(z_i is not None),
        }
        if total_kl is not None:
            diagnostics["total_kl_nats"] = float(total_kl)
        if task_kl_value is not None:
            diagnostics["task_kl_nats"] = float(task_kl_value)
        if extra_diagnostics:
            diagnostics.update(dict(extra_diagnostics))

        return ConfirmationResult(
            attribute_name=attribute_spec.attribute_name,
            status=status,
            baseline_posterior=base_p,
            conditional_posterior=cond_p,
            posterior_shift=shift,
            excess_lift=lift,
            excess_kl=float(excess_kl),
            confidence=confidence,
            support_score=support_score,
            support_diagnostics=diagnostics,
            support_failures=support_failures,
            support_warnings=support_warnings,
            rationale=rationale,
            thresholds=thresholds.to_dict(),
        )


def build_confirmation_rationale(
    *,
    attribute_spec: AttributeSpec,
    status: str,
    baseline_posterior: float,
    conditional_posterior: float,
    posterior_shift_value: float,
    excess_lift_value: float,
    excess_kl_value: float,
    support_failures: Sequence[str],
    support_status: str,
    thresholds: ConfirmationThresholds,
) -> str:
    attribute = attribute_spec.attribute_name or "this attribute"
    baseline_pct = 100.0 * float(baseline_posterior)
    conditional_pct = 100.0 * float(conditional_posterior)

    if status == "confirmed":
        return (
            f"The embedding provides additional evidence for `{attribute}` beyond the task label. "
            f"The posterior rises from {baseline_pct:.1f}% under q(S|Y) to {conditional_pct:.1f}% under q(S|Z,Y), "
            f"with shift {posterior_shift_value:+.3f}, lift {excess_lift_value:.2f}x, and excess KL {excess_kl_value:.4f} nats. "
            f"Reliability checks passed under the configured confirmation thresholds."
        )

    if status == "inconclusive":
        failure_text = ", ".join(support_failures) if support_failures else f"support_status={support_status}"
        return (
            f"There is some incremental evidence for `{attribute}`, but evidence for excess leakage is insufficient to confirm it. "
            f"The posterior rises from {baseline_pct:.1f}% to {conditional_pct:.1f}% beyond the task baseline, "
            f"yet confirmation is withheld because {failure_text}."
        )

    if posterior_shift_value <= 0.0:
        return (
            f"The observed signal for `{attribute}` is largely explained by the task label. "
            f"Conditioning on Z does not raise q(S|Z,Y) above q(S|Y) in a meaningful way "
            f"({baseline_pct:.1f}% to {conditional_pct:.1f}%)."
        )

    return (
        f"The observed signal for `{attribute}` is not confirmed as excess leakage beyond the task label. "
        f"Although q(S|Z,Y) moves from {baseline_pct:.1f}% to {conditional_pct:.1f}%, "
        f"the increase falls below at least one decision threshold "
        f"(min shift {thresholds.min_posterior_shift:.3f}, min lift {thresholds.min_excess_lift:.2f}x, "
        f"min excess KL {thresholds.min_excess_kl:.4f} nats)."
    )
