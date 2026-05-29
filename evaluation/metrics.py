from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from statistics import median
from typing import Any, Iterable


EPS = 1e-12
RECOVERY_FIELDS = ("primary", "secondary", "ternary", "background")
TASK_LABEL_TO_FIELD = {
    "primary_object": "primary",
    "secondary_object": "secondary",
    "ternary_object": "ternary",
    "background_scene": "background",
    "background_label": "background",
}


def _json_number(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return None if math.isnan(number) else number


def _task_field(task_label: str) -> str:
    return TASK_LABEL_TO_FIELD.get(str(task_label).strip(), "")


@dataclass(frozen=True)
class AttributeAudit:
    attribute_name: str
    excess_kl: float
    task_kl: float
    excess_to_task_kl_ratio: float | None
    ratio_is_infinite: bool
    confirmation_status: str
    supported: bool
    matched_field: str
    support_score: float
    matched_terms: tuple[str, ...] = ()
    matched_candidate: str = ""
    attribute_description: str = ""
    positive_patterns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matched_terms"] = list(self.matched_terms)
        payload["positive_patterns"] = list(self.positive_patterns)
        payload["excess_kl"] = _json_number(self.excess_kl)
        payload["task_kl"] = _json_number(self.task_kl)
        payload["support_score"] = _json_number(self.support_score)
        if self.ratio_is_infinite:
            payload["excess_to_task_kl_ratio"] = "inf"
        else:
            payload["excess_to_task_kl_ratio"] = _json_number(self.excess_to_task_kl_ratio)
        return payload


@dataclass(frozen=True)
class EmbeddingAuditRecord:
    image_id: str
    model_name: str
    k: int
    llm_name: str
    task_label: str
    available_fields: tuple[str, ...] = field(default_factory=tuple)
    flagged_attributes: tuple[AttributeAudit, ...] = field(default_factory=tuple)

    @property
    def num_flagged(self) -> int:
        return len(self.flagged_attributes)

    @property
    def num_valid_flagged(self) -> int:
        return sum(int(item.supported) for item in self.flagged_attributes)

    @property
    def num_invalid_flagged(self) -> int:
        return sum(int(not item.supported) for item in self.flagged_attributes)

    @property
    def task_field(self) -> str:
        return _task_field(self.task_label)

    @property
    def recovered_fields(self) -> tuple[str, ...]:
        allowed = tuple(field for field in self.available_fields if field in RECOVERY_FIELDS)
        allowed_set = set(allowed)
        recovered = {
            str(item.matched_field)
            for item in self.flagged_attributes
            if item.supported and str(item.matched_field) in allowed_set
        }
        return tuple(field for field in RECOVERY_FIELDS if field in recovered)

    @property
    def num_recovered_fields(self) -> int:
        return len(self.recovered_fields)

    @property
    def annotated_slot_recall(self) -> float:
        if not self.available_fields:
            return 0.0
        return float(self.num_recovered_fields / len(self.available_fields))

    @property
    def any_annotated_slot_recovered(self) -> bool:
        return self.num_recovered_fields > 0

    @property
    def full_annotated_slot_recovered(self) -> bool:
        return bool(self.available_fields) and self.num_recovered_fields >= len(self.available_fields)

    @property
    def leakage_available_fields(self) -> tuple[str, ...]:
        task_field = self.task_field
        return tuple(field for field in self.available_fields if field != task_field)

    @property
    def leakage_recovered_fields(self) -> tuple[str, ...]:
        task_field = self.task_field
        return tuple(field for field in self.recovered_fields if field != task_field)

    @property
    def num_leakage_recovered_fields(self) -> int:
        return len(self.leakage_recovered_fields)

    @property
    def slot_recall(self) -> float:
        if not self.leakage_available_fields:
            return 0.0
        return float(self.num_leakage_recovered_fields / len(self.leakage_available_fields))

    @property
    def any_slot_recovered(self) -> bool:
        return self.num_leakage_recovered_fields > 0

    @property
    def full_slot_recovered(self) -> bool:
        return bool(self.leakage_available_fields) and self.num_leakage_recovered_fields >= len(self.leakage_available_fields)

    @property
    def num_task_aligned_flagged(self) -> int:
        task_field = self.task_field
        if not task_field:
            return 0
        return sum(int(item.supported and str(item.matched_field) == task_field) for item in self.flagged_attributes)

    @property
    def num_excess_grounded_flagged(self) -> int:
        task_field = self.task_field
        return sum(
            int(item.supported and str(item.matched_field) in RECOVERY_FIELDS and str(item.matched_field) != task_field)
            for item in self.flagged_attributes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": str(self.image_id),
            "model_name": str(self.model_name),
            "K": int(self.k),
            "llm_name": str(self.llm_name),
            "task_label": str(self.task_label),
            "task_field": str(self.task_field),
            "available_fields": list(self.available_fields),
            "recovered_fields": list(self.recovered_fields),
            "leakage_available_fields": list(self.leakage_available_fields),
            "leakage_recovered_fields": list(self.leakage_recovered_fields),
            "annotated_available_slot_count": int(len(self.available_fields)),
            "annotated_recovered_slot_count": int(self.num_recovered_fields),
            "available_slot_count": int(len(self.leakage_available_fields)),
            "recovered_slot_count": int(self.num_leakage_recovered_fields),
            "annotated_slot_recall": float(self.annotated_slot_recall),
            "slot_recall": float(self.slot_recall),
            "any_slot_recovered": bool(self.any_slot_recovered),
            "full_slot_recovered": bool(self.full_slot_recovered),
            "any_annotated_slot_recovered": bool(self.any_annotated_slot_recovered),
            "full_annotated_slot_recovered": bool(self.full_annotated_slot_recovered),
            "num_task_aligned_flagged": int(self.num_task_aligned_flagged),
            "num_excess_grounded_flagged": int(self.num_excess_grounded_flagged),
            "num_flagged": int(self.num_flagged),
            "num_valid_flagged": int(self.num_valid_flagged),
            "num_invalid_flagged": int(self.num_invalid_flagged),
            "flagged_attributes": [item.to_dict() for item in self.flagged_attributes],
        }


def compute_excess_to_task_ratio(excess_kl: float, task_kl: float, eps: float = EPS) -> tuple[float | None, bool]:
    excess = float(excess_kl)
    task = float(task_kl)
    if math.isnan(excess) or math.isnan(task):
        return None, False
    if abs(task) <= eps:
        if abs(excess) <= eps:
            return 0.0, False
        return None, True
    return float(excess / task), False


def exceeds_ratio_threshold(value: float | None, *, is_infinite: bool, threshold: float) -> bool:
    if is_infinite:
        return True
    if value is None:
        return False
    return float(value) >= float(threshold)


def _mean(values: Iterable[float]) -> float | None:
    seq = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not seq:
        return None
    return float(sum(seq) / len(seq))


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    seq = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not seq:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(sum(seq) / len(seq)),
        "median": float(median(seq)),
        "min": float(min(seq)),
        "max": float(max(seq)),
    }


def summarize_audits(
    audits: list[EmbeddingAuditRecord],
    *,
    label_key: str,
    label_value: str,
    model_name: str,
    k: int,
    llm_name: str,
) -> dict[str, Any]:
    n_embeddings = len(audits)
    total_flagged = sum(item.num_flagged for item in audits)
    total_valid = sum(item.num_valid_flagged for item in audits)
    total_invalid = sum(item.num_invalid_flagged for item in audits)
    total_task_aligned = sum(item.num_task_aligned_flagged for item in audits)
    total_excess_grounded = sum(item.num_excess_grounded_flagged for item in audits)
    total_available_slots = sum(len(item.leakage_available_fields) for item in audits)
    total_recovered_slots = sum(item.num_leakage_recovered_fields for item in audits)
    total_annotated_available_slots = sum(len(item.available_fields) for item in audits)
    total_annotated_recovered_slots = sum(item.num_recovered_fields for item in audits)
    any_recovery_count = sum(int(item.any_slot_recovered) for item in audits)
    full_recovery_count = sum(int(item.full_slot_recovered) for item in audits)
    any_annotated_recovery_count = sum(int(item.any_annotated_slot_recovered) for item in audits)
    full_annotated_recovery_count = sum(int(item.full_annotated_slot_recovered) for item in audits)

    valid_fields = Counter()
    invalid_fields = Counter()
    available_slot_fields = Counter()
    recovered_slot_fields = Counter()
    annotated_available_slot_fields = Counter()
    annotated_recovered_slot_fields = Counter()
    excess_values: list[float] = []
    ratio_values: list[float] = []
    infinite_ratio_count = 0
    flagged_per_embedding = [item.num_flagged for item in audits]
    recovered_slots_per_embedding = [item.num_leakage_recovered_fields for item in audits]
    available_slots_per_embedding = [len(item.leakage_available_fields) for item in audits]
    annotated_recovered_slots_per_embedding = [item.num_recovered_fields for item in audits]

    for audit in audits:
        for field in audit.available_fields:
            if field in RECOVERY_FIELDS:
                annotated_available_slot_fields[str(field)] += 1
        for field in audit.leakage_available_fields:
            if field in RECOVERY_FIELDS:
                available_slot_fields[str(field)] += 1
        for field in audit.recovered_fields:
            if field in RECOVERY_FIELDS:
                annotated_recovered_slot_fields[str(field)] += 1
        for field in audit.leakage_recovered_fields:
            if field in RECOVERY_FIELDS:
                recovered_slot_fields[str(field)] += 1
        for attr in audit.flagged_attributes:
            excess_values.append(float(attr.excess_kl))
            if attr.ratio_is_infinite:
                infinite_ratio_count += 1
            elif attr.excess_to_task_kl_ratio is not None:
                ratio_values.append(float(attr.excess_to_task_kl_ratio))
            if attr.supported:
                valid_fields[str(attr.matched_field)] += 1
            else:
                invalid_fields[str(attr.matched_field)] += 1

    svr = float(total_excess_grounded / total_flagged) if total_flagged else 0.0
    hce = float(total_invalid / n_embeddings) if n_embeddings else 0.0
    esy = float(total_excess_grounded / n_embeddings) if n_embeddings else 0.0
    slot_recall = float(total_recovered_slots / total_available_slots) if total_available_slots else 0.0
    any_recovery_rate = float(any_recovery_count / n_embeddings) if n_embeddings else 0.0
    full_recovery_rate = float(full_recovery_count / n_embeddings) if n_embeddings else 0.0
    avg_recovered_slots = float(total_recovered_slots / n_embeddings) if n_embeddings else 0.0
    annotated_slot_recall = float(total_annotated_recovered_slots / total_annotated_available_slots) if total_annotated_available_slots else 0.0
    annotated_any_recovery_rate = float(any_annotated_recovery_count / n_embeddings) if n_embeddings else 0.0
    annotated_full_recovery_rate = float(full_annotated_recovery_count / n_embeddings) if n_embeddings else 0.0
    annotated_avg_recovered_slots = float(total_annotated_recovered_slots / n_embeddings) if n_embeddings else 0.0

    out: dict[str, Any] = {
        label_key: label_value,
        "SVR": svr,
        "HCE": hce,
        "ESY": esy,
        "SlotRecall": slot_recall,
        "AnyRecoveryRate": any_recovery_rate,
        "FullRecoveryRate": full_recovery_rate,
        "AvgRecoveredSlotsPerImage": avg_recovered_slots,
        "GroundedPredictionRate": svr,
        "GroundedPredictionYield": esy,
        "OpenWorldCandidateYield": hce,
        "AnnotatedSlotRecall": annotated_slot_recall,
        "AnnotatedAnyRecoveryRate": annotated_any_recovery_rate,
        "AnnotatedFullRecoveryRate": annotated_full_recovery_rate,
        "AnnotatedAvgRecoveredSlotsPerImage": annotated_avg_recovered_slots,
        "TaskAlignedPredictionYield": float(total_task_aligned / n_embeddings) if n_embeddings else 0.0,
        "model_name": str(model_name),
        "K": int(k),
        "llm_name": str(llm_name),
        "embeddings_evaluated": int(n_embeddings),
        "total_flagged_attributes": int(total_flagged),
        "total_valid_flagged": int(total_valid),
        "total_invalid_flagged": int(total_invalid),
        "total_grounded_predictions": int(total_excess_grounded),
        "total_unmapped_predictions": int(total_invalid),
        "total_task_aligned_predictions": int(total_task_aligned),
        "total_available_slots": int(total_available_slots),
        "total_recovered_slots": int(total_recovered_slots),
        "total_annotated_available_slots": int(total_annotated_available_slots),
        "total_annotated_recovered_slots": int(total_annotated_recovered_slots),
        "avg_flagged_per_embedding": _mean(flagged_per_embedding) or 0.0,
        "avg_valid_flagged_per_embedding": esy,
        "avg_invalid_flagged_per_embedding": hce,
        "avg_available_slots_per_embedding": _mean(available_slots_per_embedding) or 0.0,
        "avg_recovered_slots_per_embedding": avg_recovered_slots,
        "valid_primary": int(valid_fields.get("primary", 0)),
        "valid_secondary": int(valid_fields.get("secondary", 0)),
        "valid_ternary": int(valid_fields.get("ternary", 0)),
        "valid_background": int(valid_fields.get("background", 0)),
        "valid_none": int(valid_fields.get("none", 0)),
        "invalid_primary": int(invalid_fields.get("primary", 0)),
        "invalid_secondary": int(invalid_fields.get("secondary", 0)),
        "invalid_ternary": int(invalid_fields.get("ternary", 0)),
        "invalid_background": int(invalid_fields.get("background", 0)),
        "invalid_none": int(invalid_fields.get("none", 0)),
        "available_primary_slots": int(available_slot_fields.get("primary", 0)),
        "available_secondary_slots": int(available_slot_fields.get("secondary", 0)),
        "available_ternary_slots": int(available_slot_fields.get("ternary", 0)),
        "available_background_slots": int(available_slot_fields.get("background", 0)),
        "recovered_primary_slots": int(recovered_slot_fields.get("primary", 0)),
        "recovered_secondary_slots": int(recovered_slot_fields.get("secondary", 0)),
        "recovered_ternary_slots": int(recovered_slot_fields.get("ternary", 0)),
        "recovered_background_slots": int(recovered_slot_fields.get("background", 0)),
        "primary_recall": float(recovered_slot_fields.get("primary", 0) / available_slot_fields.get("primary", 1))
        if available_slot_fields.get("primary", 0)
        else 0.0,
        "secondary_recall": float(recovered_slot_fields.get("secondary", 0) / available_slot_fields.get("secondary", 1))
        if available_slot_fields.get("secondary", 0)
        else 0.0,
        "ternary_recall": float(recovered_slot_fields.get("ternary", 0) / available_slot_fields.get("ternary", 1))
        if available_slot_fields.get("ternary", 0)
        else 0.0,
        "background_recall": float(recovered_slot_fields.get("background", 0) / available_slot_fields.get("background", 1))
        if available_slot_fields.get("background", 0)
        else 0.0,
        "annotated_available_primary_slots": int(annotated_available_slot_fields.get("primary", 0)),
        "annotated_available_secondary_slots": int(annotated_available_slot_fields.get("secondary", 0)),
        "annotated_available_ternary_slots": int(annotated_available_slot_fields.get("ternary", 0)),
        "annotated_available_background_slots": int(annotated_available_slot_fields.get("background", 0)),
        "annotated_recovered_primary_slots": int(annotated_recovered_slot_fields.get("primary", 0)),
        "annotated_recovered_secondary_slots": int(annotated_recovered_slot_fields.get("secondary", 0)),
        "annotated_recovered_ternary_slots": int(annotated_recovered_slot_fields.get("ternary", 0)),
        "annotated_recovered_background_slots": int(annotated_recovered_slot_fields.get("background", 0)),
        "infinite_ratio_count": int(infinite_ratio_count),
    }

    for prefix, values in (
        ("flagged_per_embedding", flagged_per_embedding),
        ("recovered_slots_per_embedding", recovered_slots_per_embedding),
        ("annotated_recovered_slots_per_embedding", annotated_recovered_slots_per_embedding),
        ("excess_kl", excess_values),
        ("ratio", ratio_values),
    ):
        summary = _summary(values)
        out[f"{prefix}_mean"] = summary["mean"]
        out[f"{prefix}_median"] = summary["median"]
        out[f"{prefix}_min"] = summary["min"]
        out[f"{prefix}_max"] = summary["max"]
    return out
