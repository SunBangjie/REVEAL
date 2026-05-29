"""Stage 3 — Confirm or reject excess semantic leakage per audited embedding.

This stage treats excess leakage as a conservative decision problem rather than
as a quantity whose precise magnitude is the primary product.

For each (target, attribute) pair, it:
  - Estimates the baseline posterior q(S | Y = y_i) when available.
  - Estimates the conditional posterior q(S | Z = z_i, Y = y_i).
  - Computes supporting metrics such as posterior shift, excess lift, and excess KL.
  - Applies conservative reliability gates before assigning one of:
      confirmed | rejected | inconclusive

Usage:
    python scripts/stage2_confirm_excess_leakage.py --config configs/mvp.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.config import load_config, resolve
from audit.confirmation import (
    AttributeSpec,
    ConfirmationThresholds,
    ExcessLeakageConfirmer,
    aggregate_excess_rate,
    aggregate_excess_to_total_ratio,
    aggregate_v_info_excess,
    baseline_from_pred_label,
    build_attribute_spec_index,
    build_joint_features,
    clamp_probability,
    cross_entropy_binary,
    decomposition_residual,
    entropy_binary,
    excess_lift,
    infer_binary_attribute_columns,
    iter_stage1_attribute_items,
    label_signature,
    local_kl_binary,
    mean_top_similarity,
    merged_confirmation_config,
    normalized_excess,
    normalize_attribute_name,
    pointwise_leakage_binary,
    posterior_shift,
    prior_with_smoothing,
    target_split_from_config,
    task_kl,
    top_support_ids,
    weighted_knn_posterior,
)
from audit.task_data import (
    load_task_outputs as _load_task_outputs,
    safe_pred_label as _safe_pred_label,
    task_output_matrix as _get_output_matrix,
)
from audit.utils import ensure_parent, normalize_rows as _normalize_rows, set_seed


def _resolve_output_path(cfg: dict[str, Any], project_root: Path, keys: tuple[str, ...], default_rel: str) -> Path:
    paths_cfg = cfg.get("paths") or {}
    for key in keys:
        candidate = paths_cfg.get(key)
        if candidate:
            return resolve(candidate, project_root)
    return resolve(default_rel, project_root)


def _load_stage1_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _build_summary_row(
    sub: pd.DataFrame,
    scope: str,
    target_split: str,
    baseline_mode: str,
    *,
    attribute_name: str | None = None,
    confirmation_rate_tau: float = 0.01,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scope": scope,
        "method": "weighted_knn",
        "target_split": target_split,
        "baseline_mode": baseline_mode,
        "num_rows": int(len(sub)),
        "num_targets": int(sub["target_image_id"].nunique()),
    }
    if attribute_name is not None:
        row["attribute_name"] = str(attribute_name)
        row["secret"] = str(attribute_name)
    else:
        num_pairs = sub[["target_image_id", "attribute_name"]].drop_duplicates().shape[0]
        row["num_attributes"] = int(num_pairs)
        row["num_secrets"] = int(num_pairs)

    status_col = "confirmation_status" if "confirmation_status" in sub.columns else "status"
    if status_col in sub.columns:
        status_series = sub[status_col].fillna("").astype(str)
        for status in ("confirmed", "rejected", "inconclusive"):
            row[f"num_{status}"] = int((status_series == status).sum())
        denom = max(len(status_series), 1)
        row["confirmation_rate"] = float(row.get("num_confirmed", 0) / denom)
        row["rejection_rate"] = float(row.get("num_rejected", 0) / denom)
        row["inconclusive_rate"] = float(row.get("num_inconclusive", 0) / denom)

    for col in (
        "baseline_posterior",
        "conditional_posterior",
        "posterior_shift",
        "excess_lift",
        "excess_kl_nats",
        "local_kl_nats",
        "task_kl_nats",
        "support_score",
    ):
        vals = pd.to_numeric(sub[col], errors="coerce").dropna() if col in sub.columns else pd.Series(dtype=float)
        if len(vals):
            row[f"mean_{col}"] = float(vals.mean())
            row[f"median_{col}"] = float(vals.median())

    if "excess_kl_nats" in sub.columns:
        ekl = pd.to_numeric(sub["excess_kl_nats"], errors="coerce").dropna()
        row["legacy_excess_rate"] = aggregate_excess_rate(ekl, tau=confirmation_rate_tau)

    if {"excess_kl_nats", "local_kl_nats"}.issubset(sub.columns):
        ekl = pd.to_numeric(sub["excess_kl_nats"], errors="coerce").fillna(0.0)
        tkl = pd.to_numeric(sub["local_kl_nats"], errors="coerce").fillna(0.0)
        row["legacy_excess_to_total_ratio"] = aggregate_excess_to_total_ratio(ekl, tkl)

    if {"y_true", "baseline_posterior", "conditional_posterior"}.issubset(sub.columns):
        row["usable_confirmation_info_nats"] = aggregate_v_info_excess(
            sub["y_true"],
            sub["baseline_posterior"],
            sub["conditional_posterior"],
        )

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target-split", default=None)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 123))
    set_seed(seed)

    splits = pd.read_csv(resolve(cfg["paths"]["splits_csv"], project_root))
    labels = pd.read_csv(resolve(cfg["paths"]["labels_csv"], project_root))
    emb = np.load(resolve(cfg["paths"]["embeddings_npz"], project_root), allow_pickle=True)

    out_confirm = _resolve_output_path(
        cfg,
        project_root,
        ("target_confirmations_csv", "target_posteriors_csv"),
        "data/reports/target_confirmations.csv",
    )
    out_summary = _resolve_output_path(
        cfg,
        project_root,
        ("confirmation_summary_csv", "metrics_csv"),
        "data/reports/confirmation_summary.csv",
    )
    ensure_parent(out_confirm)
    ensure_parent(out_summary)

    stage1_path_cfg = (
        ((cfg.get("operationalization") or {}).get("stage1_path"))
        or ((cfg.get("paths") or {}).get("stage1_jsonl"))
    )
    stage1_records = _load_stage1_records(resolve(stage1_path_cfg, project_root) if stage1_path_cfg else None)
    stage1_field = str(((cfg.get("operationalization") or {}).get("from_stage1_field", "shared_attributes"))).strip().lower()
    attribute_spec_index = build_attribute_spec_index(stage1_records, field=stage1_field)

    ids = emb["image_ids"].astype(str)
    embs = _normalize_rows(emb["embeddings"].astype(np.float32))
    id_to_pos = {image_id: idx for idx, image_id in enumerate(ids)}

    task_outputs = _load_task_outputs(cfg, project_root, emb, ids, error_on_missing_image_id=True)
    task_outputs["image_id"] = task_outputs["image_id"].astype(str)
    task_outputs = task_outputs.drop_duplicates(subset=["image_id"], keep="first")
    task_by_id = (
        task_outputs.set_index("image_id")
        if not task_outputs.empty
        else pd.DataFrame().set_index(pd.Index([], name="image_id"))
    )
    output_matrix, output_cols = _get_output_matrix(task_outputs)
    pred_label_by_id = task_by_id["pred_label"].to_dict() if "pred_label" in task_by_id.columns else {}

    target_split = target_split_from_config(cfg, args.target_split)
    available_splits = set(splits["split"].astype(str).unique().tolist())
    if target_split not in available_splits:
        fallback = "audited" if "audited" in available_splits else sorted(available_splits)[0]
        print(f"[WARN] Requested target split '{target_split}' not found. Falling back to '{fallback}'.")
        target_split = fallback
    target_ids = set(splits.loc[splits["split"].astype(str) == target_split, "image_id"].astype(str).tolist())
    print(f"[INFO] Using confirmation target split: {target_split} ({len(target_ids)} targets)")

    confirm_cfg = merged_confirmation_config(cfg)
    fallback_probe = cfg.get("probe") or {}
    thresholds = ConfirmationThresholds.from_config(cfg)
    confirmer = ExcessLeakageConfirmer(thresholds)

    top_k = int(confirm_cfg.get("top_k_support", fallback_probe.get("top_k_support", 25)))
    tau = float(confirm_cfg.get("temperature", fallback_probe.get("temperature", 0.10)))
    alpha = float(confirm_cfg.get("beta_alpha", fallback_probe.get("beta_alpha", 1.0)))
    exclude_self = bool(confirm_cfg.get("exclude_self", True))
    min_labeled = int(confirm_cfg.get("min_labeled_examples", fallback_probe.get("min_labeled_examples", 5)))
    min_external = int(confirm_cfg.get("min_external_support", fallback_probe.get("min_external_support", 1)))
    max_top_support_to_store = int(confirm_cfg.get("max_top_support_to_store", 5))
    confirmation_rate_tau = float(confirm_cfg.get("min_excess_kl", confirm_cfg.get("excess_rate_threshold_nats", 0.01)))

    baseline_mode = str(confirm_cfg.get("baseline_mode", confirm_cfg.get("excess_baseline", "Y"))).strip().upper()
    if baseline_mode not in {"NONE", "Y", "O"}:
        raise ValueError("confirmation.baseline_mode must be one of NONE, Y, O")
    include_debug_metrics = bool(confirm_cfg.get("include_debug_metrics", False))
    joint_embed_weight = float(confirm_cfg.get("joint_embed_weight", 1.0))
    joint_output_weight = float(confirm_cfg.get("joint_output_weight", 1.0))

    if baseline_mode == "Y" and not pred_label_by_id:
        msg = (
            "baseline_mode=Y requested but no predicted labels were found. "
            "Regenerate embeddings/task outputs with predicted labels so q(S|Y) can be estimated."
        )
        if thresholds.require_task_baseline:
            raise ValueError(msg + " `confirmation.require_task_baseline=true`, so Stage 3 will not fall back to NONE.")
        print(f"[WARN] {msg} Falling back to NONE.")
        baseline_mode = "NONE"
    if baseline_mode == "O" and (output_matrix is None or not output_cols):
        msg = (
            "baseline_mode=O requested but no logits were found. "
            "Regenerate embeddings/task outputs with logits so q(S|Y) can be estimated from outputs."
        )
        if thresholds.require_task_baseline:
            raise ValueError(msg + " `confirmation.require_task_baseline=true`, so Stage 3 will not fall back to NONE.")
        print(f"[WARN] {msg} Falling back to NONE.")
        baseline_mode = "NONE"

    confirmation_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    # PATH A: per-target long-format labels
    if {"target_image_id", "image_id", "secret", "label"}.issubset(labels.columns):
        labels = labels.copy()
        labels["target_image_id"] = labels["target_image_id"].astype(str)
        labels["image_id"] = labels["image_id"].astype(str)
        labels["secret"] = labels["secret"].astype(str)

        target_labels = labels[labels["target_image_id"].isin(target_ids)].copy()
        groups = list(target_labels.groupby(["target_image_id", "secret"], sort=True))
        alias_map_by_group: dict[tuple[str, str], list[str]] = {}
        signature_by_group: dict[tuple[str, str], str] = {}
        if groups:
            tmp_meta: dict[str, dict[str, list[str]]] = {}
            for (target_image_id, secret), sub_meta in groups:
                vec = sub_meta[["image_id", "label"]].dropna().copy()
                vec = vec[vec["image_id"].astype(str).isin(id_to_pos)]
                vec["image_id"] = vec["image_id"].astype(str)
                vec["label"] = pd.to_numeric(vec["label"], errors="coerce").fillna(0).astype(int)
                vec = vec.drop_duplicates(subset=["image_id"], keep="last")
                signature = label_signature(vec["image_id"].tolist(), vec["label"].to_numpy())
                signature_by_group[(str(target_image_id), str(secret))] = signature
                tmp_meta.setdefault(str(target_image_id), {}).setdefault(signature, []).append(str(secret))
            for target_image_id, signature_map in tmp_meta.items():
                for signature, aliases in signature_map.items():
                    for secret in aliases:
                        alias_map_by_group[(str(target_image_id), str(secret))] = list(aliases)

        pbar = tqdm(groups, desc="Confirm excess leakage", unit="target-attribute")
        for (target_image_id, secret), sub in pbar:
            pbar.set_postfix_str(f"target={target_image_id} attribute={secret}")
            sub = sub[["image_id", "label"]].dropna().copy()
            sub = sub[sub["image_id"].isin(id_to_pos)]
            if target_image_id not in id_to_pos:
                continue

            y_all = sub["label"].astype(int).to_numpy()
            n_labeled = int(len(sub))
            n_pos = int((y_all == 1).sum())
            n_neg = int((y_all == 0).sum())
            prior_p = prior_with_smoothing(y_all, alpha)

            target_match = sub[sub["image_id"] == target_image_id]
            y_target = int(target_match["label"].iloc[0]) if len(target_match) else None

            support = sub.copy()
            if exclude_self:
                support = support[support["image_id"] != target_image_id]
            n_external = int(len(support))
            support_status = "ok"
            if n_labeled < min_labeled:
                support_status = f"too_few_labeled:{n_labeled}"
            elif n_external < min_external:
                support_status = f"too_few_external:{n_external}"

            support_ids = support["image_id"].astype(str).tolist() if n_external > 0 else []
            support_y = support["label"].astype(int).to_numpy() if n_external > 0 else np.empty((0,), dtype=int)
            x_target_z = embs[id_to_pos[target_image_id]]
            X_support_z = (
                np.stack([embs[id_to_pos[image_id]] for image_id in support_ids])
                if n_external > 0
                else np.empty((0, embs.shape[1]), dtype=np.float32)
            )

            # Embedding-only posterior q(S|Z)
            if n_external > 0:
                sims_z = X_support_z @ x_target_z
                post_p, n_eff, weights, order = weighted_knn_posterior(
                    sims_z,
                    support_y,
                    tau=tau,
                    top_k=top_k,
                    alpha=alpha,
                )
                top_support = top_support_ids(order, support_ids, weights, max_items=max_top_support_to_store)
                mean_support_sim = mean_top_similarity(sims_z, order, top_k)
            else:
                post_p, n_eff = prior_p, 0.0
                top_support = []
                mean_support_sim = float("nan")
                weights = np.empty((0,), dtype=np.float64)
                order = np.empty((0,), dtype=np.int64)
                sims_z = np.empty((0,), dtype=np.float32)

            # Baseline q(S|Y) and conditional q(S|Z,Y)
            baseline_p = prior_p
            baseline_n_eff = 0.0
            baseline_status = "prior"
            baseline_support: list[dict[str, Any]] = []
            baseline_mean_support_sim = float("nan")
            joint_p = post_p
            joint_n_eff = n_eff
            joint_support = top_support
            joint_mean_support_sim = mean_support_sim

            if baseline_mode == "Y" and n_external > 0:
                target_pred_label = _safe_pred_label(pred_label_by_id.get(target_image_id))
                support_pred_labels = np.array([_safe_pred_label(pred_label_by_id.get(image_id)) for image_id in support_ids], dtype=object)
                valid_mask = np.array([value is not None for value in support_pred_labels], dtype=bool)
                if target_pred_label is None:
                    baseline_status = "missing_target_pred_label"
                    baseline_p = prior_p
                    joint_p = post_p
                    joint_n_eff = n_eff
                elif not valid_mask.any():
                    baseline_status = "missing_support_pred_labels"
                    baseline_p = prior_p
                    joint_p = post_p
                    joint_n_eff = n_eff
                else:
                    valid_support_ids = [image_id for image_id, keep in zip(support_ids, valid_mask) if keep]
                    valid_support_y = support_y[valid_mask]
                    valid_support_pred = np.asarray([int(value) for value in support_pred_labels[valid_mask]], dtype=int)
                    baseline_p, baseline_count, same_label_mask = baseline_from_pred_label(
                        target_pred_label,
                        valid_support_pred,
                        valid_support_y,
                        alpha=alpha,
                    )
                    baseline_n_eff = float(baseline_count)
                    matched_ids = [image_id for image_id, keep in zip(valid_support_ids, same_label_mask) if keep]
                    if baseline_count <= 0:
                        baseline_status = f"no_support_with_same_pred_label:{target_pred_label}"
                        joint_p = baseline_p
                        joint_n_eff = 0.0
                        joint_support = []
                        joint_mean_support_sim = float("nan")
                    else:
                        baseline_status = f"pred_label:{target_pred_label}"
                        baseline_support = [{"image_id": image_id} for image_id in matched_ids[:max_top_support_to_store]]
                        same_y = valid_support_y[same_label_mask]
                        X_same_z = np.stack([embs[id_to_pos[image_id]] for image_id in matched_ids])
                        sims_same = X_same_z @ x_target_z
                        joint_p, joint_n_eff, joint_weights, joint_order = weighted_knn_posterior(
                            sims_same,
                            same_y,
                            tau=tau,
                            top_k=top_k,
                            alpha=alpha,
                        )
                        joint_support = top_support_ids(joint_order, matched_ids, joint_weights, max_items=max_top_support_to_store)
                        joint_mean_support_sim = mean_top_similarity(sims_same, joint_order, top_k)

            elif baseline_mode == "O" and n_external > 0:
                if output_matrix is None or target_image_id not in task_by_id.index:
                    baseline_status = "missing_logits"
                    baseline_p = prior_p
                    joint_p = post_p
                    joint_n_eff = n_eff
                else:
                    try:
                        target_out = task_by_id.loc[target_image_id, output_cols].to_numpy(dtype=np.float32)
                    except Exception:
                        target_out = None
                    if target_out is None or np.isnan(target_out).all():
                        baseline_status = "missing_target_logits"
                        baseline_p = prior_p
                        joint_p = post_p
                        joint_n_eff = n_eff
                    else:
                        support_out_rows = task_by_id.reindex(support_ids)[output_cols].to_numpy(dtype=np.float32)
                        valid_mask = ~np.isnan(support_out_rows).all(axis=1)
                        if not valid_mask.any():
                            baseline_status = "missing_support_logits"
                            baseline_p = prior_p
                            joint_p = post_p
                            joint_n_eff = n_eff
                        else:
                            target_out = _normalize_rows(np.nan_to_num(target_out, nan=0.0))[0]
                            support_out_rows = _normalize_rows(np.nan_to_num(support_out_rows[valid_mask], nan=0.0))
                            valid_support_ids = [image_id for image_id, keep in zip(support_ids, valid_mask) if keep]
                            valid_support_y = support_y[valid_mask]

                            sims_o = support_out_rows @ target_out
                            baseline_p, baseline_n_eff, baseline_weights, baseline_order = weighted_knn_posterior(
                                sims_o,
                                valid_support_y,
                                tau=tau,
                                top_k=top_k,
                                alpha=alpha,
                            )
                            baseline_status = "logits_knn"
                            baseline_support = top_support_ids(
                                baseline_order,
                                valid_support_ids,
                                baseline_weights,
                                max_items=max_top_support_to_store,
                            )
                            baseline_mean_support_sim = mean_top_similarity(sims_o, baseline_order, top_k)

                            joint_target = build_joint_features(
                                x_target_z[None, :],
                                target_out[None, :],
                                joint_embed_weight,
                                joint_output_weight,
                            )[0]
                            joint_support_rows = build_joint_features(
                                np.stack([embs[id_to_pos[image_id]] for image_id in valid_support_ids]),
                                support_out_rows,
                                joint_embed_weight,
                                joint_output_weight,
                            )
                            sims_joint = joint_support_rows @ joint_target
                            joint_p, joint_n_eff, joint_weights, joint_order = weighted_knn_posterior(
                                sims_joint,
                                valid_support_y,
                                tau=tau,
                                top_k=top_k,
                                alpha=alpha,
                            )
                            joint_support = top_support_ids(
                                joint_order,
                                valid_support_ids,
                                joint_weights,
                                max_items=max_top_support_to_store,
                            )
                            joint_mean_support_sim = mean_top_similarity(sims_joint, joint_order, top_k)

            total_kl = local_kl_binary(post_p, prior_p)
            task_kl_value = task_kl(baseline_p, prior_p)
            excess_kl_value = local_kl_binary(joint_p, baseline_p)
            excess_lift_value = excess_lift(joint_p, baseline_p)
            norm_excess_value = normalized_excess(excess_kl_value, total_kl)
            shift_value = posterior_shift(joint_p, baseline_p)
            residual_value = decomposition_residual(total_kl, task_kl_value, excess_kl_value)

            attr_key = (str(target_image_id), normalize_attribute_name(secret))
            attribute_spec = attribute_spec_index.get(attr_key, AttributeSpec(attribute_name=str(secret)))
            target_task_label = _safe_pred_label(pred_label_by_id.get(target_image_id))
            confirmation = confirmer.confirm_excess_leakage(
                x_target_z,
                target_task_label,
                attribute_spec,
                thresholds,
                baseline_posterior=baseline_p,
                conditional_posterior=joint_p,
                excess_kl=excess_kl_value,
                n_external_support=n_external,
                effective_sample_size=joint_n_eff,
                baseline_effective_sample_size=baseline_n_eff,
                baseline_mode=baseline_mode,
                baseline_status=baseline_status,
                support_status=support_status,
                total_kl=total_kl,
                task_kl_value=task_kl_value,
                mean_support_similarity=mean_support_sim,
                baseline_mean_support_similarity=baseline_mean_support_sim,
                joint_mean_support_similarity=joint_mean_support_sim,
                extra_diagnostics={
                    "prior_p": float(prior_p),
                    "embedding_only_posterior": float(post_p),
                    "y_true": y_target,
                },
            )

            row: dict[str, Any] = {
                "target_image_id": target_image_id,
                "image_id": target_image_id,
                "attribute_name": attribute_spec.attribute_name,
                "secret": secret,
                "method": "weighted_knn",
                "target_split": target_split,
                "baseline_mode": baseline_mode,
                "baseline_status": baseline_status,
                "support_status": support_status,
                "n_labeled": n_labeled,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_external_support": n_external,
                "effective_sample_size": float(n_eff),
                "joint_effective_sample_size": float(joint_n_eff),
                "baseline_effective_sample_size": float(baseline_n_eff),
                "prior_p": float(prior_p),
                "embedding_posterior": float(post_p),
                "baseline_posterior": float(baseline_p),
                "conditional_posterior": float(joint_p),
                "p_secret_1": float(post_p),
                "p_secret_0": float(1.0 - post_p),
                "p_baseline_1": float(baseline_p),
                "p_baseline_0": float(1.0 - baseline_p),
                "p_secret_given_baseline_1": float(joint_p),
                "p_secret_given_baseline_0": float(1.0 - joint_p),
                "pred_label": int(post_p >= 0.5),
                "baseline_pred_label": int(baseline_p >= 0.5),
                "joint_pred_label": int(joint_p >= 0.5),
                "local_kl_nats": total_kl,
                "task_kl_nats": task_kl_value,
                "excess_kl_nats": excess_kl_value,
                "excess_local_kl_nats": excess_kl_value,
                "excess_lift": excess_lift_value,
                "norm_excess": norm_excess_value,
                "posterior_shift": shift_value,
                "delta_p": shift_value,
                "decomposition_residual_nats": residual_value,
                "exclude_self": bool(exclude_self),
                "temperature": float(tau),
                "top_k_support": int(top_k),
                "beta_alpha": float(alpha),
                "mean_support_similarity": mean_support_sim,
                "baseline_mean_support_similarity": baseline_mean_support_sim,
                "joint_mean_support_similarity": joint_mean_support_sim,
                "joint_embed_weight": float(joint_embed_weight),
                "joint_output_weight": float(joint_output_weight),
                "label_signature": signature_by_group.get((str(target_image_id), str(secret)), ""),
                "secret_alias_count": int(len(alias_map_by_group.get((str(target_image_id), str(secret)), [str(secret)]))),
                "secret_aliases_json": json.dumps(alias_map_by_group.get((str(target_image_id), str(secret)), [str(secret)])),
                "top_support": json.dumps(top_support),
                "top_baseline_support": json.dumps(baseline_support),
                "top_joint_support": json.dumps(joint_support),
                "attribute_source_field": attribute_spec.source_field,
                "attribute_description": attribute_spec.description,
                "attribute_specificity": attribute_spec.specificity,
                "attribute_privacy_relevance": attribute_spec.privacy_relevance,
                "attribute_task_relevance": attribute_spec.task_relevance,
                "predicted_label_covers": attribute_spec.predicted_label_covers,
                "attribute_positive_patterns_json": json.dumps(list(attribute_spec.positive_patterns)),
                "attribute_negative_patterns_json": json.dumps(list(attribute_spec.negative_patterns)),
            }
            row.update(confirmation.to_dict())
            for key, value in thresholds.to_dict().items():
                row[f"threshold_{key}"] = value

            if target_task_label is not None:
                row["task_pred_label"] = int(target_task_label)
            if y_target is not None:
                row["y_true"] = int(y_target)
            if include_debug_metrics:
                row["posterior_entropy_nats"] = entropy_binary(post_p)
                row["prior_entropy_nats"] = entropy_binary(prior_p)
                row["baseline_entropy_nats"] = entropy_binary(baseline_p)
                row["joint_entropy_nats"] = entropy_binary(joint_p)
                if y_target is not None:
                    row["pointwise_leakage_nats"] = pointwise_leakage_binary(y_target, post_p, prior_p)
                    row["excess_pointwise_leakage_nats"] = pointwise_leakage_binary(y_target, joint_p, baseline_p)
                    row["ce_baseline_nats"] = cross_entropy_binary(y_target, baseline_p)
                    row["ce_joint_nats"] = cross_entropy_binary(y_target, joint_p)
            confirmation_rows.append(row)

        confirmation_df = pd.DataFrame(confirmation_rows)
        if confirmation_df.empty:
            raise ValueError("No per-target confirmation rows were produced. Check labels_csv and target split.")

        summary_rows.append(
            _build_summary_row(
                confirmation_df,
                "overall",
                target_split,
                baseline_mode,
                confirmation_rate_tau=confirmation_rate_tau,
            )
        )

        for attribute_name, sub in confirmation_df.groupby("attribute_name", dropna=False):
            summary_rows.append(
                _build_summary_row(
                    sub,
                    "attribute",
                    target_split,
                    baseline_mode,
                    attribute_name=str(attribute_name),
                    confirmation_rate_tau=confirmation_rate_tau,
                )
            )

        status_counts = confirmation_df["confirmation_status"].fillna("").astype(str).value_counts().to_dict()
        summary_rows.append(
            {
                "scope": "global",
                "method": "weighted_knn",
                "target_split": target_split,
                "baseline_mode": baseline_mode,
                "num_confirmed": int(status_counts.get("confirmed", 0)),
                "num_rejected": int(status_counts.get("rejected", 0)),
                "num_inconclusive": int(status_counts.get("inconclusive", 0)),
                "num_attributes": int(confirmation_df["attribute_name"].nunique()),
                "max_posterior_shift": float(pd.to_numeric(confirmation_df["posterior_shift"], errors="coerce").max()),
                "max_shift_attribute": str(
                    confirmation_df.loc[pd.to_numeric(confirmation_df["posterior_shift"], errors="coerce").idxmax(), "attribute_name"]
                ) if not confirmation_df.empty else "",
            }
        )

    # PATH B: backward-compatible wide-format labels
    else:
        attribute_cols = infer_binary_attribute_columns(labels)
        if not attribute_cols:
            raise ValueError("No usable label columns found in labels CSV.")
        print(f"[WARN] Using backward-compatible global wide-label mode with attributes: {attribute_cols}")

        target_df = splits[splits["split"].astype(str) == target_split][["image_id"]].copy()
        target_df["image_id"] = target_df["image_id"].astype(str)
        merged = target_df.merge(labels, on="image_id", how="inner")
        merged["image_id"] = merged["image_id"].astype(str)

        signature_by_attribute: dict[str, str] = {}
        alias_map_by_attribute: dict[str, list[str]] = {}
        tmp_meta: dict[str, list[str]] = {}
        merged_sorted = merged.sort_values("image_id").copy()
        for attribute_name in attribute_cols:
            values = pd.to_numeric(merged_sorted[attribute_name], errors="coerce").fillna(0).astype(int).to_numpy()
            signature = label_signature(merged_sorted["image_id"].astype(str).tolist(), values)
            signature_by_attribute[str(attribute_name)] = signature
            tmp_meta.setdefault(signature, []).append(str(attribute_name))
        for signature, aliases in tmp_meta.items():
            for attribute_name in aliases:
                alias_map_by_attribute[str(attribute_name)] = list(aliases)

        work_items: list[tuple[str, str, pd.DataFrame]] = []
        for attribute_name in attribute_cols:
            sub = merged[["image_id", attribute_name]].dropna().copy()
            sub = sub[sub["image_id"].isin(id_to_pos)]
            work_items.append((attribute_name, target_split, sub))

        pbar = tqdm(work_items, desc="Confirm excess leakage", unit="attribute")
        for attribute_name, _, sub in pbar:
            pbar.set_postfix_str(f"attribute={attribute_name}")
            y_all = sub[attribute_name].astype(int).to_numpy()
            prior_p = prior_with_smoothing(y_all, alpha)
            for target_image_id in sub["image_id"].astype(str).tolist():
                support = sub.copy()
                if exclude_self:
                    support = support[support["image_id"] != target_image_id]
                if len(support) == 0:
                    post_p, n_eff = prior_p, 0.0
                    top_support = []
                    mean_support_sim = float("nan")
                else:
                    support_ids = support["image_id"].astype(str).tolist()
                    support_y = support[attribute_name].astype(int).to_numpy()
                    x_target = embs[id_to_pos[target_image_id]]
                    X_support = np.stack([embs[id_to_pos[image_id]] for image_id in support_ids])
                    sims = X_support @ x_target
                    post_p, n_eff, weights, order = weighted_knn_posterior(
                        sims,
                        support_y,
                        tau=tau,
                        top_k=top_k,
                        alpha=alpha,
                    )
                    top_support = top_support_ids(order, support_ids, weights, max_items=max_top_support_to_store)
                    mean_support_sim = mean_top_similarity(sims, order, top_k)

                y_target = int(sub.loc[sub["image_id"] == target_image_id, attribute_name].iloc[0])
                total_kl = local_kl_binary(post_p, prior_p)
                excess_kl_value = total_kl
                shift_value = posterior_shift(post_p, prior_p)
                attribute_spec = AttributeSpec(attribute_name=str(attribute_name))
                target_task_label = _safe_pred_label(pred_label_by_id.get(target_image_id))
                confirmation = confirmer.confirm_excess_leakage(
                    embs[id_to_pos[target_image_id]],
                    target_task_label,
                    attribute_spec,
                    thresholds,
                    baseline_posterior=prior_p,
                    conditional_posterior=post_p,
                    excess_kl=excess_kl_value,
                    n_external_support=int(len(support)),
                    effective_sample_size=n_eff,
                    baseline_effective_sample_size=0.0,
                    baseline_mode="NONE",
                    baseline_status="prior",
                    support_status="ok",
                    total_kl=total_kl,
                    task_kl_value=0.0,
                    mean_support_similarity=mean_support_sim,
                    joint_mean_support_similarity=mean_support_sim,
                )

                row = {
                    "target_image_id": target_image_id,
                    "image_id": target_image_id,
                    "attribute_name": attribute_name,
                    "secret": attribute_name,
                    "method": "weighted_knn",
                    "target_split": target_split,
                    "baseline_mode": "NONE",
                    "baseline_status": "prior",
                    "support_status": "ok",
                    "n_labeled": int(len(sub)),
                    "n_positive": int((y_all == 1).sum()),
                    "n_negative": int((y_all == 0).sum()),
                    "n_external_support": int(len(support)),
                    "effective_sample_size": float(n_eff),
                    "joint_effective_sample_size": float(n_eff),
                    "baseline_effective_sample_size": 0.0,
                    "prior_p": float(prior_p),
                    "embedding_posterior": float(post_p),
                    "baseline_posterior": float(prior_p),
                    "conditional_posterior": float(post_p),
                    "p_secret_1": float(post_p),
                    "p_secret_0": float(1.0 - post_p),
                    "p_baseline_1": float(prior_p),
                    "p_baseline_0": float(1.0 - prior_p),
                    "p_secret_given_baseline_1": float(post_p),
                    "p_secret_given_baseline_0": float(1.0 - post_p),
                    "pred_label": int(post_p >= 0.5),
                    "baseline_pred_label": int(prior_p >= 0.5),
                    "joint_pred_label": int(post_p >= 0.5),
                    "local_kl_nats": total_kl,
                    "task_kl_nats": 0.0,
                    "excess_kl_nats": excess_kl_value,
                    "excess_local_kl_nats": excess_kl_value,
                    "excess_lift": excess_lift(post_p, prior_p),
                    "norm_excess": 1.0,
                    "posterior_shift": shift_value,
                    "delta_p": shift_value,
                    "decomposition_residual_nats": 0.0,
                    "y_true": int(y_target),
                    "exclude_self": bool(exclude_self),
                    "temperature": float(tau),
                    "top_k_support": int(top_k),
                    "beta_alpha": float(alpha),
                    "mean_support_similarity": mean_support_sim,
                    "baseline_mean_support_similarity": float("nan"),
                    "joint_mean_support_similarity": mean_support_sim,
                    "joint_embed_weight": float(joint_embed_weight),
                    "joint_output_weight": float(joint_output_weight),
                    "label_signature": signature_by_attribute.get(str(attribute_name), ""),
                    "secret_alias_count": int(len(alias_map_by_attribute.get(str(attribute_name), [str(attribute_name)]))),
                    "secret_aliases_json": json.dumps(alias_map_by_attribute.get(str(attribute_name), [str(attribute_name)])),
                    "top_support": json.dumps(top_support),
                    "top_baseline_support": json.dumps([]),
                    "top_joint_support": json.dumps(top_support),
                    "attribute_source_field": "",
                    "attribute_description": "",
                    "attribute_specificity": None,
                    "attribute_privacy_relevance": None,
                    "attribute_task_relevance": None,
                    "predicted_label_covers": "",
                    "attribute_positive_patterns_json": json.dumps([]),
                    "attribute_negative_patterns_json": json.dumps([]),
                }
                row.update(confirmation.to_dict())
                for key, value in thresholds.to_dict().items():
                    row[f"threshold_{key}"] = value

                if include_debug_metrics:
                    row["posterior_entropy_nats"] = entropy_binary(post_p)
                    row["prior_entropy_nats"] = entropy_binary(prior_p)
                    row["baseline_entropy_nats"] = entropy_binary(prior_p)
                    row["joint_entropy_nats"] = entropy_binary(post_p)
                    row["pointwise_leakage_nats"] = pointwise_leakage_binary(y_target, post_p, prior_p)
                    row["excess_pointwise_leakage_nats"] = pointwise_leakage_binary(y_target, post_p, prior_p)
                    row["ce_baseline_nats"] = cross_entropy_binary(y_target, prior_p)
                    row["ce_joint_nats"] = cross_entropy_binary(y_target, post_p)
                confirmation_rows.append(row)

        confirmation_df = pd.DataFrame(confirmation_rows)
        summary_rows.append(
            _build_summary_row(
                confirmation_df,
                "overall",
                target_split,
                "NONE",
                confirmation_rate_tau=confirmation_rate_tau,
            )
        )
        status_counts = confirmation_df["confirmation_status"].fillna("").astype(str).value_counts().to_dict()
        summary_rows.append(
            {
                "scope": "global",
                "method": "weighted_knn",
                "target_split": target_split,
                "baseline_mode": "NONE",
                "num_confirmed": int(status_counts.get("confirmed", 0)),
                "num_rejected": int(status_counts.get("rejected", 0)),
                "num_inconclusive": int(status_counts.get("inconclusive", 0)),
                "num_attributes": int(confirmation_df["attribute_name"].nunique()),
            }
        )

    confirmation_df = pd.DataFrame(confirmation_rows)
    summary_df = pd.DataFrame(summary_rows)
    confirmation_df.to_csv(out_confirm, index=False)
    summary_df.to_csv(out_summary, index=False)

    print(f"Saved target-centric confirmation results to {out_confirm}")
    print(f"Saved confirmation summary to {out_summary}")
    print(f"[INFO] Rows: {len(confirmation_df)} | Targets: {confirmation_df['target_image_id'].nunique() if not confirmation_df.empty else 0}")
    print(f"[INFO] Baseline mode: {baseline_mode}")
    print(f"[INFO] Debug metrics: {'enabled' if include_debug_metrics else 'disabled'}")
    if not confirmation_df.empty and "confirmation_status" in confirmation_df.columns:
        status_counts = confirmation_df["confirmation_status"].fillna("").astype(str).value_counts().to_dict()
        print(
            "[INFO] Confirmation statuses: "
            + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
        )


if __name__ == "__main__":
    main()
