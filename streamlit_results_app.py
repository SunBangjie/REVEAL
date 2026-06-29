from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import altair as alt
except Exception:  # pragma: no cover - Streamlit normally provides Altair.
    alt = None

import pandas as pd
import streamlit as st

from audit.config import (
    infer_project_root_from_config,
    load_config as load_yaml,
    resolve_image_path_from_record,
    semantic_parts_from_record,
)
from evaluation.metrics import summarize_audits
from evaluation.runner import EvaluationRunner, EvaluationSetting
from evaluation.semantic_support import normalize_text
from scripts.run_evaluation import _flagged_image_rows, _review_rows


MODEL_ORDER = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_base_patch16_224",
]

SUMMARY_NUMERIC_COLUMNS = [
    "NetSemanticDiscoveryYield",
    "NetSemanticDiscoveryYieldConservative",
    "AutoVerifiedExtraYield",
    "HallucinationYield",
    "SlotRecall",
    "AnyRecoveryRate",
    "FullRecoveryRate",
    "AvgRecoveredSlotsPerImage",
    "GroundedPredictionRate",
    "GroundedPredictionYield",
    "OpenWorldCandidateYield",
    "AnnotatedSlotRecall",
    "AnnotatedAnyRecoveryRate",
    "AnnotatedFullRecoveryRate",
    "AnnotatedAvgRecoveredSlotsPerImage",
    "TaskAlignedPredictionYield",
    "SVR",
    "HCE",
    "ESY",
    "K",
    "embeddings_evaluated",
    "total_available_slots",
    "total_recovered_slots",
    "total_task_aligned_predictions",
    "total_annotated_available_slots",
    "total_annotated_recovered_slots",
    "total_grounded_predictions",
    "total_unmapped_predictions",
    "total_flagged_attributes",
    "total_valid_flagged",
    "total_invalid_flagged",
    "avg_flagged_per_embedding",
    "avg_valid_flagged_per_embedding",
    "avg_invalid_flagged_per_embedding",
    "avg_available_slots_per_embedding",
    "avg_recovered_slots_per_embedding",
    "available_primary_slots",
    "available_secondary_slots",
    "available_ternary_slots",
    "available_background_slots",
    "recovered_primary_slots",
    "recovered_secondary_slots",
    "recovered_ternary_slots",
    "recovered_background_slots",
    "primary_recall",
    "secondary_recall",
    "ternary_recall",
    "background_recall",
    "AutoVerificationCoverage",
    "AutoVerifiedRateAll",
    "AutoHallucinationRateAll",
    "AutoUncertainRateAll",
    "AutoVerifiedRateDecided",
    "auto_reviewed_unmapped_candidates",
    "auto_decided_unmapped_candidates",
    "auto_verified_candidates",
    "auto_likely_hallucination_candidates",
    "auto_uncertain_candidates",
    "auto_missing_image_candidates",
    "valid_primary",
    "valid_secondary",
    "valid_ternary",
    "valid_background",
    "valid_none",
    "invalid_primary",
    "invalid_secondary",
    "invalid_ternary",
    "invalid_background",
    "invalid_none",
    "infinite_ratio_count",
    "flagged_per_embedding_mean",
    "flagged_per_embedding_median",
    "flagged_per_embedding_min",
    "flagged_per_embedding_max",
    "excess_kl_mean",
    "excess_kl_median",
    "excess_kl_min",
    "excess_kl_max",
    "ratio_mean",
    "ratio_median",
    "ratio_min",
    "ratio_max",
]

FLAGGED_NUMERIC_COLUMNS = [
    "K",
    "available_slot_count",
    "recovered_slot_count",
    "slot_recall_image",
    "num_task_aligned_for_image",
    "num_flagged_for_image",
    "flagged_attribute_index",
    "support_score",
    "excess_kl",
    "task_kl",
    "excess_to_task_kl_ratio",
    "auto_similarity",
    "auto_global_percentile",
    "auto_scene_percentile",
    "auto_global_margin",
    "auto_scene_margin",
    "auto_prompt_margin",
    "auto_valid_probability",
    "num_valid_flagged_for_image",
    "num_invalid_flagged_for_image",
]

IMAGE_NUMERIC_COLUMNS = [
    "K",
    "available_slot_count",
    "recovered_slot_count",
    "slot_recall_image",
    "num_task_aligned",
    "num_flagged",
    "num_valid_flagged",
    "num_invalid_flagged",
]

TEXT_FILL_COLUMNS = [
    "image_id",
    "image_path",
    "image_relpath",
    "output_file",
    "task_label",
    "model_name",
    "llm_name",
    "attribute_name",
    "attribute_description",
    "positive_patterns",
    "available_fields",
    "recovered_fields",
    "grounding_label",
    "auto_best_prompt",
    "auto_verdict",
    "auto_verdict_display",
    "matched_slot",
    "matched_candidate",
    "matched_field",
    "matched_terms",
    "confirmation_status",
    "semantic_text",
    "scene_family",
    "scene_family_label",
    "primary_label",
    "secondary_label",
    "ternary_label",
    "background_label",
]
EMPTY_FLAGGED_COLUMNS = list(
    dict.fromkeys(
        [
            "display_setting",
            *TEXT_FILL_COLUMNS,
            *FLAGGED_NUMERIC_COLUMNS,
            "annotation_grounded",
            "semantic_supported",
            "ratio_is_infinite",
            "task_aligned_prediction",
            "is_unmapped_candidate",
            "human_review_label",
            "human_reviewer",
            "human_review_notes",
        ]
    )
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = APP_DIR / "output"
TASK_LABEL_TO_FIELD = {
    "primary_object": "primary",
    "secondary_object": "secondary",
    "ternary_object": "ternary",
    "background_scene": "background",
    "background_label": "background",
}
RECOVERY_SLOT_SPECS = {
    "primary": ("primary_label",),
    "secondary": ("secondary_label",),
    "ternary": ("ternary_label",),
    "background": ("background_label",),
}
AUTO_VERDICT_DISPLAY = {
    "auto_verified_extra": "Auto-verified extra",
    "auto_likely_hallucination": "Likely hallucination",
    "auto_uncertain": "Auto-uncertain",
    "auto_missing_image": "Missing image",
}
SUPPORT_LABEL_COLOR_DOMAIN = ["Task-aligned recovery", "Excess grounded recovery", "Unmapped candidate"]
SUPPORT_LABEL_COLOR_RANGE = ["#4C78A8", "#2F855A", "#C53030"]

SCENE_ANALYSIS_METRICS: dict[str, dict[str, str]] = {
    "NetSemanticDiscoveryYield": {
        "label": "Net discovery / image",
        "title": "Recovered non-task slots + auto-verified extra - hallucinations, per image",
        "kind": "float",
        "tooltip_format": ".3f",
    },
    "NetSemanticDiscoveryYieldConservative": {
        "label": "Conservative net discovery / image",
        "title": "Recovered non-task slots + auto-verified extra - 2x hallucinations, per image",
        "kind": "float",
        "tooltip_format": ".3f",
    },
    "SlotRecall": {
        "label": "Excess slot recall",
        "title": "Recovered non-task slots / non-task slots",
        "kind": "percent",
        "tooltip_format": ".1%",
    },
    "AnyRecoveryRate": {
        "label": "Any excess recovery",
        "title": "Images with any recovered non-task slot / target images",
        "kind": "percent",
        "tooltip_format": ".1%",
    },
    "FullRecoveryRate": {
        "label": "Full excess recovery",
        "title": "Images with all non-task slots recovered / target images",
        "kind": "percent",
        "tooltip_format": ".1%",
    },
    "AvgRecoveredSlotsPerImage": {
        "label": "Excess recovered / image",
        "title": "Recovered non-task slots per target image",
        "kind": "float",
        "tooltip_format": ".3f",
    },
    "GroundedPredictionRate": {
        "label": "Excess grounded rate",
        "title": "Predictions grounded to non-task slots / flagged attributes",
        "kind": "percent",
        "tooltip_format": ".1%",
    },
    "OpenWorldCandidateYield": {
        "label": "Unmapped candidates / image",
        "title": "Unmapped high-evidence candidates per target image",
        "kind": "float",
        "tooltip_format": ".3f",
    },
    "recovered_slots": {
        "label": "Recovered slots",
        "title": "Recovered annotated slots",
        "kind": "int",
        "tooltip_format": ",d",
    },
    "num_flagged": {
        "label": "Flagged attributes",
        "title": "Flagged attributes",
        "kind": "int",
        "tooltip_format": ",d",
    },
    "grounded_predictions": {
        "label": "Grounded predictions",
        "title": "Predictions grounded to annotated slots",
        "kind": "int",
        "tooltip_format": ",d",
    },
    "unmapped_predictions": {
        "label": "Unmapped candidates",
        "title": "Unmapped high-evidence candidates",
        "kind": "int",
        "tooltip_format": ",d",
    },
}


@dataclass(frozen=True)
class ResultsBundle:
    mode: str
    project_root: Path
    output_dir: Path
    config_path: Path | None
    cfg: dict[str, Any]
    eval_meta: dict[str, Any]
    summary_df: pd.DataFrame
    flagged_df: pd.DataFrame
    image_df: pd.DataFrame
    target_df: pd.DataFrame
    data_source: str = "aggregate_files"
    discovered_run_count: int = 0


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _bool_series(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    return series.map(lambda value: mapping.get(_text(value).lower(), False)).astype(bool)


def _text_series(series: pd.Series) -> pd.Series:
    return series.map(_text).astype(object)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _read_csv_or_empty(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])


@st.cache_data(show_spinner=False)
def load_unmapped_verification(
    verification_csv_path_str: str,
    verification_summary_path_str: str,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    verification_df = pd.DataFrame()
    verification_meta: dict[str, Any] = {}

    verification_csv_path = Path(verification_csv_path_str).expanduser().resolve() if _text(verification_csv_path_str) else None
    verification_summary_path = Path(verification_summary_path_str).expanduser().resolve() if _text(verification_summary_path_str) else None

    if verification_csv_path is not None and verification_csv_path.exists():
        verification_df = _read_csv_or_empty(verification_csv_path)
    if verification_summary_path is not None and verification_summary_path.exists():
        verification_meta = _load_json(verification_summary_path)

    return verification_df, verification_meta


def _sort_models(values: list[str]) -> list[str]:
    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    return sorted(values, key=lambda item: (order.get(item, len(order)), item))


def _setting_sort_key(value: Any) -> tuple[Any, ...]:
    text = _text(value)
    model_order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    if text in model_order:
        return (0, model_order[text], text)
    if text.startswith("K="):
        try:
            return (1, float(text.split("=", 1)[1]), text)
        except Exception:
            return (1, float("inf"), text)
    if text.startswith("LLM="):
        return (2, text.split("=", 1)[1].lower(), text)
    return (3, text.lower())


def _sort_settings(values: list[str]) -> list[str]:
    return sorted(values, key=_setting_sort_key)


def _normalize_numeric_token(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def _setting_signature(model_name: Any, k: Any, llm_name: Any) -> str:
    return "||".join([_text(model_name), _normalize_numeric_token(k), _text(llm_name)])


def _pair_signature(image_id: Any, model_name: Any, k: Any, llm_name: Any) -> str:
    return "||".join([_text(image_id), _setting_signature(model_name, k, llm_name)])


def _verification_pair_signature(image_id: Any, attribute_name: Any) -> str:
    return f"{_text(image_id)}::{normalize_text(_text(attribute_name))}"


def _fallback_setting_label(row: Mapping[str, Any]) -> str:
    model_name = _text(row.get("model_name"))
    k = _normalize_numeric_token(row.get("K"))
    llm_name = _text(row.get("llm_name"))
    if llm_name and k:
        return f"K={k} | LLM={llm_name}"
    return model_name or "Unknown"


def _setting_type(value: Any) -> str:
    text = _text(value)
    if text.startswith("K="):
        return "K Ablation"
    if text.startswith("LLM="):
        return "LLM Ablation"
    return "Model"


def _display_entity_name(mode: str) -> str:
    return "Model" if mode == "rq1" else "Setting"


def _display_pair_name(mode: str) -> str:
    return "image-model pairs" if mode == "rq1" else "image-setting pairs"


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _format_scene_family(record: Mapping[str, Any]) -> str:
    label = _text(record.get("scene_family_label"))
    return label or _text(record.get("scene_family")) or "Unknown"


def _format_auto_verdict(value: Any) -> str:
    return AUTO_VERDICT_DISPLAY.get(_text(value), _text(value))


def _available_slot_fields_from_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    available: list[str] = []
    for field_name, columns in RECOVERY_SLOT_SPECS.items():
        if any(_text(record.get(column)) for column in columns):
            available.append(field_name)
    return tuple(available)


def _task_field_from_record(record: Mapping[str, Any]) -> str:
    return TASK_LABEL_TO_FIELD.get(_text(record.get("task_label")), "")


def _leakage_available_slot_fields_from_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    task_field = _task_field_from_record(record)
    return tuple(field for field in _available_slot_fields_from_record(record) if field != task_field)


def _prepare_summary_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    summary = df.copy()
    for col in SUMMARY_NUMERIC_COLUMNS:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    if "model_name" not in summary.columns:
        summary["model_name"] = ""
    if "llm_name" not in summary.columns:
        summary["llm_name"] = ""
    if mode == "rq1":
        if "Embedding Model" not in summary.columns and "model_name" in summary.columns:
            summary["Embedding Model"] = summary["model_name"]
        if "model_name" not in summary.columns and "Embedding Model" in summary.columns:
            summary["model_name"] = summary["Embedding Model"]
    elif "Setting" not in summary.columns:
        summary["Setting"] = summary.apply(_fallback_setting_label, axis=1)
    summary["model_name"] = summary["model_name"].map(_text)
    summary["llm_name"] = summary["llm_name"].map(_text)
    summary["display_setting"] = summary["model_name"].map(_text) if mode == "rq1" else summary["Setting"].map(_text)
    summary["setting_type"] = summary["display_setting"].map(_setting_type)
    summary["setting_signature"] = summary.apply(
        lambda row: _setting_signature(row.get("model_name"), row.get("K"), row.get("llm_name")),
        axis=1,
    )
    if "total_grounded_predictions" not in summary.columns and "total_valid_flagged" in summary.columns:
        summary["total_grounded_predictions"] = pd.to_numeric(summary["total_valid_flagged"], errors="coerce")
    if "total_unmapped_predictions" not in summary.columns and "total_invalid_flagged" in summary.columns:
        summary["total_unmapped_predictions"] = pd.to_numeric(summary["total_invalid_flagged"], errors="coerce")
    if "GroundedPredictionRate" not in summary.columns and "SVR" in summary.columns:
        summary["GroundedPredictionRate"] = pd.to_numeric(summary["SVR"], errors="coerce")
    if "GroundedPredictionYield" not in summary.columns and "ESY" in summary.columns:
        summary["GroundedPredictionYield"] = pd.to_numeric(summary["ESY"], errors="coerce")
    if "OpenWorldCandidateYield" not in summary.columns and "HCE" in summary.columns:
        summary["OpenWorldCandidateYield"] = pd.to_numeric(summary["HCE"], errors="coerce")
    return summary


def _fill_display_setting(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    if df.empty:
        frame = df.copy()
        if "display_setting" not in frame.columns:
            frame["display_setting"] = ""
        return frame
    frame = df.copy()
    existing = frame["display_setting"].map(_text) if "display_setting" in frame.columns else pd.Series("", index=frame.index)
    inferred_mode = mode
    if inferred_mode is None:
        if "Embedding Model" in frame.columns:
            inferred_mode = "rq1"
        else:
            model_count = frame["model_name"].dropna().astype(str).nunique() if "model_name" in frame.columns else 0
            k_count = frame["K"].dropna().astype(str).nunique() if "K" in frame.columns else 0
            llm_count = frame["llm_name"].dropna().astype(str).nunique() if "llm_name" in frame.columns else 0
            inferred_mode = "rq1" if model_count > 1 and k_count <= 1 and llm_count <= 1 else "rq2"
    if inferred_mode == "rq1" and "model_name" in frame.columns:
        fallback = frame["model_name"].map(_text)
    else:
        fallback = frame.apply(_fallback_setting_label, axis=1)
    frame["display_setting"] = existing
    missing = frame["display_setting"] == ""
    frame.loc[missing, "display_setting"] = fallback[missing]
    return frame


def _backfill_summary_attack_metrics(
    summary_df: pd.DataFrame,
    flagged_df: pd.DataFrame,
    image_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    summary = summary_df.copy()
    flagged = _prepare_flagged_df(flagged_df) if not flagged_df.empty else flagged_df.copy()
    image = _prepare_image_df(image_df, flagged) if not image_df.empty else _derive_image_df_from_flagged(flagged)
    target = _ensure_target_slot_columns(target_df) if not target_df.empty else target_df.copy()

    if "display_setting" not in summary.columns:
        return summary

    settings = summary[["display_setting"]].drop_duplicates().copy()
    settings["target_images"] = int(target["image_id"].nunique()) if (not target.empty and "image_id" in target.columns) else 0
    settings["total_available_slots"] = int(target["available_slot_count"].sum()) if (not target.empty and "available_slot_count" in target.columns) else 0
    for field_name in RECOVERY_SLOT_SPECS:
        col = f"available_{field_name}_slots"
        source_col = f"available_{field_name}_slot"
        settings[col] = int(target[source_col].sum()) if (not target.empty and source_col in target.columns) else 0

    if not flagged.empty:
        grounded = flagged["annotation_grounded"].astype(bool)
        flagged_agg = (
            flagged.groupby("display_setting", as_index=False)
            .agg(
                total_flagged_attributes=("attribute_name", "size"),
                total_task_aligned_predictions=("task_aligned_prediction", "sum"),
                total_grounded_predictions=("annotation_grounded", "sum"),
                total_unmapped_predictions=("is_unmapped_candidate", "sum"),
            )
        )
        grounded_slots = flagged[grounded & flagged["matched_slot"].isin(list(RECOVERY_SLOT_SPECS.keys()))].copy()
        if not grounded_slots.empty:
            recovered_slot_agg = (
                grounded_slots.groupby("display_setting", as_index=False)
                .agg(
                    total_recovered_slots=("matched_slot", "size"),
                    recovered_primary_slots=("matched_slot", lambda s: int((pd.Series(s).astype(str) == "primary").sum())),
                    recovered_secondary_slots=("matched_slot", lambda s: int((pd.Series(s).astype(str) == "secondary").sum())),
                    recovered_ternary_slots=("matched_slot", lambda s: int((pd.Series(s).astype(str) == "ternary").sum())),
                    recovered_background_slots=("matched_slot", lambda s: int((pd.Series(s).astype(str) == "background").sum())),
                )
            )
            flagged_agg = flagged_agg.merge(recovered_slot_agg, on="display_setting", how="left")
        settings = settings.merge(flagged_agg, on="display_setting", how="left")

    if not image.empty:
        image_agg = (
            image.groupby("display_setting", as_index=False)
            .agg(
                total_recovered_slots_image=("recovered_slot_count", "sum"),
                any_slot_recovered_images=("any_slot_recovered", "sum"),
                full_slot_recovered_images=("full_slot_recovered", "sum"),
            )
        )
        settings = settings.merge(image_agg, on="display_setting", how="left")

    summary = summary.merge(settings, on="display_setting", how="left", suffixes=("", "__derived"))

    preferred_numeric = [
        "total_flagged_attributes",
        "total_task_aligned_predictions",
        "total_grounded_predictions",
        "total_unmapped_predictions",
        "total_available_slots",
        "total_recovered_slots",
        "available_primary_slots",
        "available_secondary_slots",
        "available_ternary_slots",
        "available_background_slots",
        "recovered_primary_slots",
        "recovered_secondary_slots",
        "recovered_ternary_slots",
        "recovered_background_slots",
        "target_images",
        "any_slot_recovered_images",
        "full_slot_recovered_images",
    ]
    for col in preferred_numeric:
        derived_col = f"{col}__derived"
        if col == "total_recovered_slots" and derived_col not in summary.columns and "total_recovered_slots_image__derived" in summary.columns:
            derived_col = "total_recovered_slots_image__derived"
        if derived_col in summary.columns:
            summary[col] = pd.to_numeric(summary[derived_col], errors="coerce")
            summary = summary.drop(columns=[derived_col])

    for col in preferred_numeric:
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)

    if "embeddings_evaluated" in summary.columns:
        embedding_denom = pd.to_numeric(summary["embeddings_evaluated"], errors="coerce").fillna(0)
    else:
        embedding_denom = summary["target_images"]
    image_denom = embedding_denom.where(embedding_denom > 0)
    slot_denom = summary["total_available_slots"].where(summary["total_available_slots"] > 0)
    summary["GroundedPredictionRate"] = (
        summary["total_grounded_predictions"] / summary["total_flagged_attributes"].where(summary["total_flagged_attributes"] > 0)
    ).fillna(0.0)
    summary["GroundedPredictionYield"] = (summary["total_grounded_predictions"] / image_denom).fillna(0.0)
    summary["OpenWorldCandidateYield"] = (
        summary["total_unmapped_predictions"] / image_denom
    ).fillna(0.0)
    summary["SlotRecall"] = (summary["total_recovered_slots"] / slot_denom).fillna(0.0)
    summary["AnyRecoveryRate"] = (summary["any_slot_recovered_images"] / image_denom).fillna(0.0)
    summary["FullRecoveryRate"] = (summary["full_slot_recovered_images"] / image_denom).fillna(0.0)
    summary["AvgRecoveredSlotsPerImage"] = (summary["total_recovered_slots"] / image_denom).fillna(0.0)
    summary["primary_recall"] = (
        summary["recovered_primary_slots"] / summary["available_primary_slots"].where(summary["available_primary_slots"] > 0)
    ).fillna(0.0)
    summary["secondary_recall"] = (
        summary["recovered_secondary_slots"] / summary["available_secondary_slots"].where(summary["available_secondary_slots"] > 0)
    ).fillna(0.0)
    summary["ternary_recall"] = (
        summary["recovered_ternary_slots"] / summary["available_ternary_slots"].where(summary["available_ternary_slots"] > 0)
    ).fillna(0.0)
    summary["background_recall"] = (
        summary["recovered_background_slots"] / summary["available_background_slots"].where(summary["available_background_slots"] > 0)
    ).fillna(0.0)

    summary["SVR"] = summary["GroundedPredictionRate"]
    summary["ESY"] = summary["GroundedPredictionYield"]
    summary["HCE"] = summary["OpenWorldCandidateYield"]
    return summary


def _apply_net_discovery_metrics(
    df: pd.DataFrame,
    *,
    recovered_col: str,
    verified_col: str,
    hallucination_col: str,
    image_denom: pd.Series,
) -> pd.DataFrame:
    frame = df.copy()
    recovered = (
        pd.to_numeric(frame[recovered_col], errors="coerce").fillna(0)
        if recovered_col in frame.columns
        else pd.Series(0, index=frame.index, dtype=float)
    )
    verified = (
        pd.to_numeric(frame[verified_col], errors="coerce").fillna(0)
        if verified_col in frame.columns
        else pd.Series(0, index=frame.index, dtype=float)
    )
    hallucinations = (
        pd.to_numeric(frame[hallucination_col], errors="coerce").fillna(0)
        if hallucination_col in frame.columns
        else pd.Series(0, index=frame.index, dtype=float)
    )
    frame["AutoVerifiedExtraYield"] = (verified / image_denom).fillna(0.0)
    frame["HallucinationYield"] = (hallucinations / image_denom).fillna(0.0)
    frame["NetSemanticDiscoveryYield"] = ((recovered + verified - hallucinations) / image_denom).fillna(0.0)
    frame["NetSemanticDiscoveryYieldConservative"] = (
        (recovered + verified - (2 * hallucinations)) / image_denom
    ).fillna(0.0)
    return frame


def _prepare_flagged_df(df: pd.DataFrame) -> pd.DataFrame:
    flagged = _fill_display_setting(df)
    if "display_setting" in flagged.columns:
        flagged["display_setting"] = flagged["display_setting"].map(_text)
    for col in TEXT_FILL_COLUMNS:
        if col in flagged.columns:
            flagged[col] = flagged[col].map(_text)
    for col in FLAGGED_NUMERIC_COLUMNS:
        if col in flagged.columns:
            flagged[col] = pd.to_numeric(flagged[col], errors="coerce")
    if "semantic_supported" in flagged.columns:
        flagged["semantic_supported"] = _bool_series(flagged["semantic_supported"])
    else:
        flagged["semantic_supported"] = False
    if "task_field" not in flagged.columns:
        flagged["task_field"] = flagged.apply(_task_field_from_record, axis=1)
    if "available_fields" not in flagged.columns:
        flagged["available_fields"] = flagged.apply(lambda row: " | ".join(_available_slot_fields_from_record(row)), axis=1)
    if "leakage_available_fields" not in flagged.columns:
        flagged["leakage_available_fields"] = flagged.apply(
            lambda row: " | ".join(_leakage_available_slot_fields_from_record(row)),
            axis=1,
        )
    if "available_slot_count" not in flagged.columns:
        flagged["available_slot_count"] = flagged.apply(lambda row: len(_leakage_available_slot_fields_from_record(row)), axis=1)
    if "ratio_is_infinite" in flagged.columns:
        flagged["ratio_is_infinite"] = _bool_series(flagged["ratio_is_infinite"])
    else:
        flagged["ratio_is_infinite"] = False
    if "matched_slot" not in flagged.columns:
        flagged["matched_slot"] = flagged["matched_field"].map(_text) if "matched_field" in flagged.columns else ""
    matched_slot = flagged["matched_slot"].map(_text)
    task_field = flagged["task_field"].map(_text)
    leakage_field_sets = flagged["leakage_available_fields"].map(
        lambda value: {item.strip() for item in _text(value).split("|") if item.strip()}
    )
    flagged["task_aligned_prediction"] = flagged["semantic_supported"] & matched_slot.eq(task_field) & task_field.ne("")
    leakage_match_mask = pd.Series(
        [slot in allowed if slot else False for slot, allowed in zip(matched_slot.tolist(), leakage_field_sets.tolist())],
        index=flagged.index,
        dtype=bool,
    )
    flagged["annotation_grounded"] = (
        flagged["semantic_supported"] & ~flagged["task_aligned_prediction"] & leakage_match_mask
    ).astype(bool)
    flagged["grounding_label"] = "Unmapped candidate"
    flagged.loc[flagged["task_aligned_prediction"], "grounding_label"] = "Task-aligned recovery"
    flagged.loc[flagged["annotation_grounded"], "grounding_label"] = "Excess grounded recovery"
    flagged["support_label"] = flagged["grounding_label"]
    flagged["is_unmapped_candidate"] = flagged["grounding_label"].eq("Unmapped candidate")
    flagged["scene_family_display"] = flagged.apply(_format_scene_family, axis=1)
    flagged["setting_signature"] = flagged.apply(
        lambda row: _setting_signature(row.get("model_name"), row.get("K"), row.get("llm_name")),
        axis=1,
    )
    flagged["pair_key"] = flagged.apply(
        lambda row: _pair_signature(row.get("image_id"), row.get("model_name"), row.get("K"), row.get("llm_name")),
        axis=1,
    )
    flagged["verification_pair_key"] = flagged.apply(
        lambda row: _verification_pair_signature(row.get("image_id"), row.get("attribute_name")),
        axis=1,
    )
    if "display_setting" in flagged.columns:
        flagged["display_pair_key"] = _text_series(flagged["pair_key"]) + "||" + _text_series(flagged["display_setting"])
    return flagged


def _prepare_verification_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "verification_pair_key",
                "auto_best_prompt",
                "auto_similarity",
                "auto_global_percentile",
                "auto_scene_percentile",
                "auto_global_margin",
                "auto_scene_margin",
                "auto_prompt_margin",
                "auto_valid_probability",
                "auto_verdict",
                "run_name",
            ]
        )

    verification = df.copy()
    for col in [
        "pair_key",
        "image_id",
        "attribute_name",
        "attribute_description",
        "positive_patterns",
        "scene_family_label",
        "auto_best_prompt",
        "auto_verdict",
        "run_name",
    ]:
        if col in verification.columns:
            verification[col] = verification[col].map(_text)
    for col in [
        "auto_similarity",
        "auto_global_percentile",
        "auto_scene_percentile",
        "auto_global_margin",
        "auto_scene_margin",
        "auto_prompt_margin",
        "auto_valid_probability",
    ]:
        if col in verification.columns:
            verification[col] = pd.to_numeric(verification[col], errors="coerce")
    if "run_name" in verification.columns:
        run_names = {value for value in verification["run_name"].dropna().map(_text).tolist() if value}
        if mode in run_names:
            verification = verification[verification["run_name"].map(_text) == mode].copy()
    if "pair_key" in verification.columns and verification["pair_key"].astype(str).str.len().gt(0).any():
        verification["verification_pair_key"] = verification["pair_key"].map(_text)
    else:
        verification["verification_pair_key"] = verification.apply(
            lambda row: _verification_pair_signature(row.get("image_id"), row.get("attribute_name")),
            axis=1,
        )
    sort_cols = [col for col in ["auto_valid_probability", "auto_similarity"] if col in verification.columns]
    if sort_cols:
        verification = verification.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="stable")
    verification = verification.drop_duplicates(subset=["verification_pair_key"], keep="first").reset_index(drop=True)
    return verification


def _attach_verification_to_flagged(flagged_df: pd.DataFrame, verification_df: pd.DataFrame) -> pd.DataFrame:
    flagged = flagged_df.copy()
    if "verification_pair_key" not in flagged.columns:
        flagged["verification_pair_key"] = flagged.apply(
            lambda row: _verification_pair_signature(row.get("image_id"), row.get("attribute_name")),
            axis=1,
        )

    if verification_df.empty:
        flagged["verification_available"] = False
        flagged["auto_verdict"] = ""
        flagged["auto_best_prompt"] = ""
        flagged["auto_valid_probability"] = pd.NA
        flagged["auto_verdict_display"] = "Unavailable"
    else:
        verification_cols = [
            "verification_pair_key",
            "auto_best_prompt",
            "auto_similarity",
            "auto_global_percentile",
            "auto_scene_percentile",
            "auto_global_margin",
            "auto_scene_margin",
            "auto_prompt_margin",
            "auto_valid_probability",
            "auto_verdict",
        ]
        available_cols = [col for col in verification_cols if col in verification_df.columns]
        flagged = flagged.merge(
            verification_df[available_cols].copy(),
            on="verification_pair_key",
            how="left",
        )
        flagged["verification_available"] = flagged["auto_verdict"].map(_text).ne("")
        flagged["auto_verdict"] = flagged["auto_verdict"].map(_text)
        if "auto_best_prompt" in flagged.columns:
            flagged["auto_best_prompt"] = flagged["auto_best_prompt"].map(_text)

    if "auto_verdict" not in flagged.columns:
        flagged["auto_verdict"] = ""
    flagged["auto_verdict_display"] = "Unavailable"
    not_applicable = flagged["grounding_label"].map(_text) != "Unmapped candidate"
    flagged.loc[not_applicable, "auto_verdict_display"] = "Not applicable"
    verdict_mask = flagged["auto_verdict"].map(_text).ne("")
    flagged.loc[verdict_mask, "auto_verdict_display"] = flagged.loc[verdict_mask, "auto_verdict"].map(_format_auto_verdict)
    return flagged


def _backfill_summary_unmapped_verification_metrics(
    summary_df: pd.DataFrame,
    flagged_df: pd.DataFrame,
) -> pd.DataFrame:
    if summary_df.empty or "display_setting" not in summary_df.columns:
        return summary_df.copy()

    summary = summary_df.copy()
    if "embeddings_evaluated" in summary.columns:
        image_denom = pd.to_numeric(summary["embeddings_evaluated"], errors="coerce").fillna(0)
    else:
        image_denom = pd.to_numeric(summary.get("target_images"), errors="coerce").fillna(0)
    image_denom = image_denom.where(image_denom > 0)

    if flagged_df.empty:
        for col in [
            "auto_reviewed_unmapped_candidates",
            "auto_decided_unmapped_candidates",
            "auto_verified_candidates",
            "auto_likely_hallucination_candidates",
            "auto_uncertain_candidates",
            "auto_missing_image_candidates",
        ]:
            if col not in summary.columns:
                summary[col] = 0
        return _apply_net_discovery_metrics(
            summary,
            recovered_col="total_recovered_slots",
            verified_col="auto_verified_candidates",
            hallucination_col="auto_likely_hallucination_candidates",
            image_denom=image_denom,
        )

    flagged = flagged_df.copy()
    if "auto_verdict" not in flagged.columns:
        for col in [
            "auto_reviewed_unmapped_candidates",
            "auto_decided_unmapped_candidates",
            "auto_verified_candidates",
            "auto_likely_hallucination_candidates",
            "auto_uncertain_candidates",
            "auto_missing_image_candidates",
            "AutoVerificationCoverage",
            "AutoVerifiedRateAll",
            "AutoHallucinationRateAll",
            "AutoUncertainRateAll",
            "AutoVerifiedRateDecided",
        ]:
            if col not in summary.columns:
                summary[col] = pd.NA
        for col in [
            "auto_verified_candidates",
            "auto_likely_hallucination_candidates",
        ]:
            if col not in summary.columns:
                summary[col] = 0
        return _apply_net_discovery_metrics(
            summary,
            recovered_col="total_recovered_slots",
            verified_col="auto_verified_candidates",
            hallucination_col="auto_likely_hallucination_candidates",
            image_denom=image_denom,
        )

    unmapped = flagged[flagged["grounding_label"].map(_text) == "Unmapped candidate"].copy()
    if unmapped.empty:
        for col in [
            "auto_reviewed_unmapped_candidates",
            "auto_decided_unmapped_candidates",
            "auto_verified_candidates",
            "auto_likely_hallucination_candidates",
            "auto_uncertain_candidates",
            "auto_missing_image_candidates",
            "AutoVerificationCoverage",
            "AutoVerifiedRateAll",
            "AutoHallucinationRateAll",
            "AutoUncertainRateAll",
            "AutoVerifiedRateDecided",
        ]:
            if col not in summary.columns:
                summary[col] = 0 if col.startswith("auto_") and col.endswith("candidates") else pd.NA
        return _apply_net_discovery_metrics(
            summary,
            recovered_col="total_recovered_slots",
            verified_col="auto_verified_candidates",
            hallucination_col="auto_likely_hallucination_candidates",
            image_denom=image_denom,
        )

    verdict_series = unmapped["auto_verdict"].map(_text)
    reviewed = verdict_series.ne("")
    agg = (
        unmapped.assign(
            auto_reviewed_unmapped_candidates=reviewed.astype(int),
            auto_verified_candidates=verdict_series.eq("auto_verified_extra").astype(int),
            auto_likely_hallucination_candidates=verdict_series.eq("auto_likely_hallucination").astype(int),
            auto_uncertain_candidates=verdict_series.eq("auto_uncertain").astype(int),
            auto_missing_image_candidates=verdict_series.eq("auto_missing_image").astype(int),
        )
        .groupby("display_setting", as_index=False)
        .agg(
            auto_reviewed_unmapped_candidates=("auto_reviewed_unmapped_candidates", "sum"),
            auto_verified_candidates=("auto_verified_candidates", "sum"),
            auto_likely_hallucination_candidates=("auto_likely_hallucination_candidates", "sum"),
            auto_uncertain_candidates=("auto_uncertain_candidates", "sum"),
            auto_missing_image_candidates=("auto_missing_image_candidates", "sum"),
        )
    )
    agg["auto_decided_unmapped_candidates"] = (
        agg["auto_verified_candidates"] + agg["auto_likely_hallucination_candidates"]
    )

    summary = summary.merge(agg, on="display_setting", how="left", suffixes=("", "__auto"))
    for col in [
        "auto_reviewed_unmapped_candidates",
        "auto_decided_unmapped_candidates",
        "auto_verified_candidates",
        "auto_likely_hallucination_candidates",
        "auto_uncertain_candidates",
        "auto_missing_image_candidates",
    ]:
        auto_col = f"{col}__auto"
        if auto_col in summary.columns:
            summary[col] = pd.to_numeric(summary[auto_col], errors="coerce")
            summary = summary.drop(columns=[auto_col])
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)

    total_unmapped = pd.to_numeric(summary.get("total_unmapped_predictions"), errors="coerce")
    reviewed_denom = summary["auto_reviewed_unmapped_candidates"].where(summary["auto_reviewed_unmapped_candidates"] > 0)
    decided_denom = summary["auto_decided_unmapped_candidates"].where(summary["auto_decided_unmapped_candidates"] > 0)
    total_unmapped_denom = total_unmapped.where(total_unmapped > 0)

    summary["AutoVerificationCoverage"] = (
        summary["auto_reviewed_unmapped_candidates"] / total_unmapped_denom
    )
    summary["AutoVerifiedRateAll"] = (
        summary["auto_verified_candidates"] / total_unmapped_denom
    ).where(reviewed_denom.notna())
    summary["AutoHallucinationRateAll"] = (
        summary["auto_likely_hallucination_candidates"] / total_unmapped_denom
    ).where(reviewed_denom.notna())
    summary["AutoUncertainRateAll"] = (
        summary["auto_uncertain_candidates"] / total_unmapped_denom
    ).where(reviewed_denom.notna())
    summary["AutoVerifiedRateDecided"] = (
        summary["auto_verified_candidates"] / decided_denom
    )
    return _apply_net_discovery_metrics(
        summary,
        recovered_col="total_recovered_slots",
        verified_col="auto_verified_candidates",
        hallucination_col="auto_likely_hallucination_candidates",
        image_denom=image_denom,
    )


def _derive_image_df_from_flagged(flagged: pd.DataFrame) -> pd.DataFrame:
    flagged = _fill_display_setting(flagged)
    grouping = [
        "display_setting",
        "image_id",
        "image_path",
        "image_relpath",
        "output_file",
        "task_label",
        "semantic_text",
        "scene_family",
        "scene_family_label",
        "primary_label",
        "secondary_label",
        "ternary_label",
        "background_label",
        "model_name",
        "K",
        "llm_name",
    ]
    keep = [col for col in grouping if col in flagged.columns]
    image_df = (
        flagged.groupby(keep, dropna=False, as_index=False)
        .agg(
            available_slot_count=("available_slot_count", "max"),
            num_flagged=("attribute_name", "size"),
            num_task_aligned=("task_aligned_prediction", "sum"),
            num_valid_flagged=("annotation_grounded", "sum"),
            num_invalid_flagged=("is_unmapped_candidate", "sum"),
        )
        .copy()
    )
    grounded_pairs = flagged[flagged["annotation_grounded"] & flagged["matched_slot"].isin(RECOVERY_SLOT_SPECS.keys())].copy()
    if not grounded_pairs.empty:
        grounded_pairs["slot_pair_key"] = _text_series(grounded_pairs["image_id"]) + "||" + _text_series(grounded_pairs["matched_slot"])
        recovered = (
            grounded_pairs.groupby(keep, dropna=False, as_index=False)
            .agg(recovered_slot_count=("matched_slot", lambda s: pd.Series(s).dropna().astype(str).nunique()))
            .copy()
        )
        image_df = image_df.merge(recovered, on=keep, how="left")
    else:
        image_df["recovered_slot_count"] = 0
    image_df["available_slot_count"] = pd.to_numeric(image_df.get("available_slot_count"), errors="coerce").fillna(0).astype(int)
    image_df["recovered_slot_count"] = pd.to_numeric(image_df.get("recovered_slot_count"), errors="coerce").fillna(0).astype(int)
    denom = image_df["available_slot_count"].where(image_df["available_slot_count"] > 0)
    image_df["slot_recall_image"] = (image_df["recovered_slot_count"] / denom).fillna(0.0)
    image_df["any_slot_recovered"] = image_df["recovered_slot_count"] > 0
    image_df["full_slot_recovered"] = (image_df["available_slot_count"] > 0) & (
        image_df["recovered_slot_count"] >= image_df["available_slot_count"]
    )
    image_df["setting_signature"] = image_df.apply(
        lambda row: _setting_signature(row.get("model_name"), row.get("K"), row.get("llm_name")),
        axis=1,
    )
    image_df["pair_key"] = image_df.apply(
        lambda row: _pair_signature(row.get("image_id"), row.get("model_name"), row.get("K"), row.get("llm_name")),
        axis=1,
    )
    image_df["display_pair_key"] = _text_series(image_df["pair_key"]) + "||" + _text_series(image_df["display_setting"])
    image_df["scene_family_display"] = image_df.apply(_format_scene_family, axis=1)
    return image_df


def _prepare_image_df(df: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _derive_image_df_from_flagged(flagged)
    image_df = _fill_display_setting(df)
    if "display_setting" in image_df.columns:
        image_df["display_setting"] = image_df["display_setting"].map(_text)
    for col in TEXT_FILL_COLUMNS:
        if col in image_df.columns:
            image_df[col] = image_df[col].map(_text)
    for col in IMAGE_NUMERIC_COLUMNS:
        if col in image_df.columns:
            image_df[col] = pd.to_numeric(image_df[col], errors="coerce")
    for col in ("any_slot_recovered", "full_slot_recovered"):
        if col in image_df.columns:
            image_df[col] = _bool_series(image_df[col])
    image_df["scene_family_display"] = image_df.apply(_format_scene_family, axis=1)
    image_df["setting_signature"] = image_df.apply(
        lambda row: _setting_signature(row.get("model_name"), row.get("K"), row.get("llm_name")),
        axis=1,
    )
    image_df["pair_key"] = image_df.apply(
        lambda row: _pair_signature(row.get("image_id"), row.get("model_name"), row.get("K"), row.get("llm_name")),
        axis=1,
    )
    if "display_setting" in image_df.columns:
        image_df["display_pair_key"] = _text_series(image_df["pair_key"]) + "||" + _text_series(image_df["display_setting"])
    derived_cols = [
        "display_pair_key",
        "pair_key",
        "num_task_aligned",
        "num_valid_flagged",
        "num_invalid_flagged",
        "available_slot_count",
        "recovered_slot_count",
        "slot_recall_image",
        "any_slot_recovered",
        "full_slot_recovered",
    ]
    if not flagged.empty:
        derived = _derive_image_df_from_flagged(flagged)[derived_cols]
        merge_key = "display_pair_key" if "display_pair_key" in image_df.columns and "display_pair_key" in derived.columns else "pair_key"
        derived = derived.drop_duplicates(subset=[merge_key], keep="first")
        image_df = image_df.merge(derived, on=merge_key, how="left", suffixes=("", "_derived"))
        for col in derived_cols[1:]:
            derived_col = f"{col}_derived"
            if derived_col in image_df.columns:
                image_df[col] = image_df[derived_col].combine_first(image_df[col]) if col in image_df.columns else image_df[derived_col]
                image_df = image_df.drop(columns=[derived_col])
    return image_df


def _prepare_target_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["image_id", "scene_family_display", "available_slot_count"])
    target_df = df.copy()
    for col in TEXT_FILL_COLUMNS:
        if col in target_df.columns:
            target_df[col] = target_df[col].map(_text)
    if "image_id" in target_df.columns:
        target_df["image_id"] = target_df["image_id"].map(_text)
    target_df["scene_family_display"] = target_df.apply(_format_scene_family, axis=1)
    available_fields = target_df.apply(_available_slot_fields_from_record, axis=1)
    leakage_available_fields = target_df.apply(_leakage_available_slot_fields_from_record, axis=1)
    target_df["task_field"] = target_df.apply(_task_field_from_record, axis=1)
    target_df["available_fields"] = available_fields.map(lambda values: " | ".join(values))
    target_df["leakage_available_fields"] = leakage_available_fields.map(lambda values: " | ".join(values))
    for field_name in RECOVERY_SLOT_SPECS:
        target_df[f"annotated_available_{field_name}_slot"] = available_fields.map(lambda values, field=field_name: int(field in values))
        target_df[f"available_{field_name}_slot"] = leakage_available_fields.map(lambda values, field=field_name: int(field in values))
    target_df["annotated_available_slot_count"] = sum(target_df[f"annotated_available_{field}_slot"] for field in RECOVERY_SLOT_SPECS)
    target_df["available_slot_count"] = sum(target_df[f"available_{field}_slot"] for field in RECOVERY_SLOT_SPECS)
    return target_df


def _ensure_target_slot_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _prepare_target_df(df)
    target_df = df.copy()
    if "scene_family_display" not in target_df.columns:
        target_df["scene_family_display"] = target_df.apply(_format_scene_family, axis=1)
    if "image_id" in target_df.columns:
        target_df["image_id"] = target_df["image_id"].map(_text)
    if "task_field" not in target_df.columns:
        target_df["task_field"] = target_df.apply(_task_field_from_record, axis=1)
    for field_name in RECOVERY_SLOT_SPECS:
        annotated_col = f"annotated_available_{field_name}_slot"
        col = f"available_{field_name}_slot"
        if annotated_col not in target_df.columns:
            target_df[annotated_col] = target_df.apply(
                lambda row, field=field_name: int(field in _available_slot_fields_from_record(row)),
                axis=1,
            )
        target_df[annotated_col] = pd.to_numeric(target_df[annotated_col], errors="coerce").fillna(0).astype(int)
        if col not in target_df.columns:
            target_df[col] = target_df.apply(
                lambda row, field=field_name: int(field in _leakage_available_slot_fields_from_record(row)),
                axis=1,
            )
        target_df[col] = pd.to_numeric(target_df[col], errors="coerce").fillna(0).astype(int)
    if "annotated_available_slot_count" not in target_df.columns:
        target_df["annotated_available_slot_count"] = sum(target_df[f"annotated_available_{field}_slot"] for field in RECOVERY_SLOT_SPECS)
    target_df["annotated_available_slot_count"] = pd.to_numeric(target_df["annotated_available_slot_count"], errors="coerce").fillna(0).astype(int)
    if "available_slot_count" not in target_df.columns:
        target_df["available_slot_count"] = sum(target_df[f"available_{field}_slot"] for field in RECOVERY_SLOT_SPECS)
    target_df["available_slot_count"] = pd.to_numeric(target_df["available_slot_count"], errors="coerce").fillna(0).astype(int)
    return target_df


def _attach_setting_labels(df: pd.DataFrame, summary_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    labeled = df.copy()
    if labeled.empty:
        labeled["display_setting"] = ""
        labeled["setting_type"] = ""
        if "pair_key" in labeled.columns:
            labeled["display_pair_key"] = ""
        return labeled

    setting_map: dict[str, str] = {}
    if not summary_df.empty and {"setting_signature", "display_setting"}.issubset(summary_df.columns):
        signature_map_df = summary_df[["setting_signature", "display_setting"]].drop_duplicates().copy()
        signature_counts = signature_map_df.groupby("setting_signature")["display_setting"].nunique()
        unique_signatures = signature_counts[signature_counts == 1].index.tolist()
        if unique_signatures:
            setting_map = (
                signature_map_df[signature_map_df["setting_signature"].isin(unique_signatures)]
                .set_index("setting_signature")["display_setting"]
                .to_dict()
            )

    existing_display = (
        labeled["display_setting"].map(_text) if "display_setting" in labeled.columns else pd.Series("", index=labeled.index)
    )
    mapped_display = (
        labeled["setting_signature"].map(setting_map).fillna("").map(_text)
        if "setting_signature" in labeled.columns
        else pd.Series("", index=labeled.index)
    )
    if mode == "rq1":
        fallback = labeled["model_name"].map(_text)
    else:
        fallback = labeled.apply(_fallback_setting_label, axis=1)
    labeled["display_setting"] = existing_display
    missing = labeled["display_setting"] == ""
    labeled.loc[missing, "display_setting"] = mapped_display[missing]
    missing = labeled["display_setting"] == ""
    labeled.loc[missing, "display_setting"] = fallback[missing]
    labeled["setting_type"] = labeled["display_setting"].map(_setting_type)
    if "pair_key" in labeled.columns:
        labeled["display_pair_key"] = _text_series(labeled["pair_key"]) + "||" + _text_series(labeled["display_setting"])
    return labeled


def _load_optional_config(eval_meta: Mapping[str, Any], fallback_root: Path) -> tuple[Path | None, dict[str, Any]]:
    candidates = [
        _text(eval_meta.get("base_config_path")),
        _text(eval_meta.get("config_path")),
        str(fallback_root / "configs" / "mvp.yaml"),
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.exists():
            continue
        try:
            resolved = path.resolve()
            return resolved, load_yaml(resolved)
        except Exception:
            continue
    return None, {}


def _resolve_eval_config_path(eval_meta: Mapping[str, Any], output_dir: Path) -> Path | None:
    candidates = [
        _text(eval_meta.get("config_path")),
        str(output_dir.parent / "configs" / "eval.yaml"),
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _load_target_manifest_df(eval_meta: Mapping[str, Any], output_dir: Path) -> pd.DataFrame:
    eval_config_path = _resolve_eval_config_path(eval_meta, output_dir)
    if eval_config_path is None:
        return pd.DataFrame()
    try:
        runner = EvaluationRunner(eval_config_path, output_dir=str(output_dir))
    except Exception:
        return pd.DataFrame()
    return _prepare_target_df(runner.target_manifest.copy())


def _rows_with_display_setting(rows: list[dict[str, Any]], display_setting: str) -> list[dict[str, Any]]:
    labeled_rows: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["display_setting"] = display_setting
        labeled_rows.append(payload)
    return labeled_rows


def _rq2_setting_label(k: int, llm_name: str, reference_k: int) -> str:
    return f"LLM={llm_name}" if int(k) == int(reference_k) else f"K={int(k)}"


def _infer_preserved_rq2_setting(
    run_dir: Path,
    runner: EvaluationRunner,
    *,
    reference_k: int,
) -> EvaluationSetting | None:
    if not run_dir.is_dir():
        return None
    if not (run_dir / "target_confirmations.csv").exists():
        return None

    run_config_path = run_dir / "eval_run_config.yaml"
    if not run_config_path.exists():
        return None

    try:
        run_cfg = load_yaml(run_config_path)
    except Exception:
        return None

    meta = run_cfg.get("evaluation_setting") or {}
    model_name = _text(meta.get("model_name")) or _text((run_cfg.get("embeddings") or {}).get("model_name")) or "resnet50"
    try:
        k = int(meta.get("k") if meta.get("k") is not None else (run_cfg.get("retrieval") or {}).get("top_k"))
    except Exception:
        return None

    explicit_llm = _text(meta.get("llm_name"))
    candidate_llms = [explicit_llm] if explicit_llm else list(runner.llm_specs.keys())

    matches: list[EvaluationSetting] = []
    for llm_name in candidate_llms:
        llm_spec = runner.llm_specs.get(llm_name)
        if llm_spec is None:
            continue
        setting = EvaluationSetting(
            mode="rq2",
            label=_rq2_setting_label(k, llm_name, reference_k),
            model_name=model_name,
            k=int(k),
            llm_name=llm_name,
            llm_spec=llm_spec,
        )
        if runner._preserved_run_matches_setting(run_dir, setting):
            matches.append(setting)

    if not matches:
        return None
    return max(matches, key=lambda item: len(item.slug))


def _build_rq1_reference_frames_for_rq2(
    output_dir: Path,
    eval_meta: Mapping[str, Any],
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eval_config_path = _resolve_eval_config_path(eval_meta, output_dir)
    if eval_config_path is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    runner = EvaluationRunner(eval_config_path, output_dir=str(output_dir))
    existing_labels = set(summary_df["display_setting"].dropna().map(_text).tolist()) if "display_setting" in summary_df.columns else set()
    rq2_model = _text((runner.eval_cfg.get("rq2") or {}).get("model"))
    if not rq2_model and "model_name" in summary_df.columns:
        rq2_models = [value for value in summary_df["model_name"].dropna().map(_text).unique().tolist() if value]
        rq2_model = rq2_models[0] if rq2_models else ""
    if not rq2_model:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    rq1_setting = next((item for item in runner.settings_for_mode("rq1") if item.model_name == rq2_model), None)
    if rq1_setting is None or runner.find_existing_run_dir(rq1_setting) is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    _, audits = runner.run_existing_setting(rq1_setting)
    if not audits:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    summary_rows: list[dict[str, Any]] = []
    flagged_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    added_labels: list[str] = []
    candidate_labels = [f"K={int(rq1_setting.k)}", f"LLM={rq1_setting.llm_name}"]
    for label in candidate_labels:
        if label in existing_labels:
            continue
        summary_rows.append(
            summarize_audits(
                audits,
                label_key="Setting",
                label_value=label,
                model_name=rq1_setting.model_name,
                k=rq1_setting.k,
                llm_name=rq1_setting.llm_name,
            )
        )
        flagged_rows.extend(_rows_with_display_setting(_review_rows(audits, runner.target_manifest), label))
        image_rows.extend(_rows_with_display_setting(_flagged_image_rows(audits, runner.target_manifest), label))
        existing_labels.add(label)
        added_labels.append(label)

    meta_updates = {"includes_rq1_reference": bool(added_labels), "rq1_reference_labels": added_labels} if added_labels else {}
    summary_ref = _prepare_summary_df(pd.DataFrame(summary_rows), "rq2") if summary_rows else pd.DataFrame()
    flagged_ref = _prepare_flagged_df(pd.DataFrame(flagged_rows)) if flagged_rows else pd.DataFrame()
    image_ref = _prepare_image_df(pd.DataFrame(image_rows), flagged_ref) if image_rows else pd.DataFrame()
    return summary_ref, flagged_ref, image_ref, meta_updates


def _discover_preserved_rq2_settings(output_dir: Path, eval_config_path: Path) -> tuple[EvaluationRunner, list[EvaluationSetting]]:
    runner = EvaluationRunner(eval_config_path, output_dir=str(output_dir))
    rq1_cfg = runner.eval_cfg.get("rq1") or {}
    rq2_cfg = runner.eval_cfg.get("rq2") or {}
    reference_k = int(rq1_cfg.get("k", rq2_cfg.get("reference_k", 10)))
    latest_settings: dict[tuple[str, int, str], tuple[float, EvaluationSetting]] = {}

    for run_dir in sorted(output_dir.glob("rq2_*")):
        setting = _infer_preserved_rq2_setting(run_dir, runner, reference_k=reference_k)
        if setting is None:
            continue
        key = (setting.model_name, setting.k, setting.llm_name)
        confirmations_path = run_dir / "target_confirmations.csv"
        try:
            mtime = confirmations_path.stat().st_mtime
        except Exception:
            mtime = run_dir.stat().st_mtime
        existing = latest_settings.get(key)
        if existing is None or mtime >= existing[0]:
            latest_settings[key] = (mtime, setting)

    settings = [item[1] for item in latest_settings.values()]
    settings = sorted(settings, key=lambda item: (_setting_sort_key(item.label), item.model_name, item.llm_name))
    return runner, settings


def _finalize_results_bundle(
    mode: str,
    *,
    project_root: Path,
    output_dir: Path,
    config_path: Path | None,
    cfg: dict[str, Any],
    eval_meta: dict[str, Any],
    summary_df: pd.DataFrame,
    flagged_df: pd.DataFrame,
    image_df: pd.DataFrame,
    target_df: pd.DataFrame,
    data_source: str = "aggregate_files",
    discovered_run_count: int = 0,
) -> ResultsBundle:
    flagged_df = _attach_setting_labels(flagged_df, summary_df, mode)
    image_df = _attach_setting_labels(image_df, summary_df, mode)

    flagged_image_counts = (
        image_df.groupby("display_setting", as_index=False)["pair_key"]
        .nunique()
        .rename(columns={"pair_key": "flagged_image_pairs"})
    )
    summary_df = summary_df.merge(flagged_image_counts, on="display_setting", how="left")
    summary_df["flagged_image_pairs"] = summary_df["flagged_image_pairs"].fillna(0).astype(int)
    summary_df["flagged_image_coverage"] = (
        summary_df["flagged_image_pairs"] / summary_df["embeddings_evaluated"].where(summary_df["embeddings_evaluated"] > 0)
    )

    return ResultsBundle(
        mode=mode,
        project_root=project_root,
        output_dir=output_dir,
        config_path=config_path,
        cfg=cfg,
        eval_meta=eval_meta,
        summary_df=summary_df,
        flagged_df=flagged_df,
        image_df=image_df,
        target_df=target_df,
        data_source=data_source,
        discovered_run_count=int(discovered_run_count),
    )


def _load_preserved_rq2_bundle(output_dir: Path, eval_meta: Mapping[str, Any]) -> ResultsBundle | None:
    eval_config_path = _resolve_eval_config_path(eval_meta, output_dir)
    if eval_config_path is None:
        return None

    runner, settings = _discover_preserved_rq2_settings(output_dir, eval_config_path)
    if not settings:
        return None

    summaries: list[dict[str, Any]] = []
    flagged_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for setting in settings:
        summary, audit_rows = runner.run_existing_setting(setting)
        summaries.append(summary)
        flagged_rows.extend(_rows_with_display_setting(_review_rows(audit_rows, runner.target_manifest), setting.label))
        image_rows.extend(_rows_with_display_setting(_flagged_image_rows(audit_rows, runner.target_manifest), setting.label))

    eval_meta_out = dict(eval_meta)
    eval_meta_out.update(
        {
            "mode": "rq2",
            "config_path": str(eval_config_path),
            "base_config_path": str(runner.base_config_path),
            "output_dir": str(output_dir),
            "data_source": "preserved_runs",
            "preserved_run_count": len(settings),
        }
    )

    config_path, cfg = _load_optional_config(eval_meta_out, output_dir.parent)
    summary_df = _prepare_summary_df(pd.DataFrame(summaries), "rq2")
    flagged_df = _prepare_flagged_df(pd.DataFrame(flagged_rows))
    image_df = _prepare_image_df(pd.DataFrame(image_rows), flagged_df)
    target_df = _prepare_target_df(runner.target_manifest.copy())
    ref_summary_df, ref_flagged_df, ref_image_df, ref_meta = _build_rq1_reference_frames_for_rq2(
        output_dir,
        eval_meta_out,
        summary_df,
    )
    if not ref_summary_df.empty:
        summary_df = pd.concat([summary_df, ref_summary_df], ignore_index=True)
    if not ref_flagged_df.empty:
        flagged_df = pd.concat([flagged_df, ref_flagged_df], ignore_index=True)
    if not ref_image_df.empty:
        image_df = pd.concat([image_df, ref_image_df], ignore_index=True)
    if ref_meta:
        eval_meta_out.update(ref_meta)
    return _finalize_results_bundle(
        "rq2",
        project_root=runner.project_root,
        output_dir=output_dir,
        config_path=config_path,
        cfg=cfg,
        eval_meta=eval_meta_out,
        summary_df=summary_df,
        flagged_df=flagged_df,
        image_df=image_df,
        target_df=target_df,
        data_source="preserved_runs",
        discovered_run_count=len(settings),
    )


@st.cache_data(show_spinner=False)
def load_results_bundle(
    mode: str,
    summary_path_str: str,
    eval_json_path_str: str,
    flagged_csv_path_str: str,
    flagged_images_path_str: str,
    auto_merge_preserved_runs: bool = False,
) -> ResultsBundle:
    summary_path = Path(summary_path_str).expanduser().resolve()
    eval_json_path = Path(eval_json_path_str).expanduser().resolve()
    flagged_csv_path = Path(flagged_csv_path_str).expanduser().resolve()
    flagged_images_path = Path(flagged_images_path_str).expanduser().resolve()

    eval_meta = _load_json(eval_json_path) if eval_json_path.exists() else {}
    if auto_merge_preserved_runs and mode == "rq2":
        preserved_bundle = _load_preserved_rq2_bundle(summary_path.parent, eval_meta)
        if preserved_bundle is not None:
            return preserved_bundle

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    if not flagged_csv_path.exists():
        raise FileNotFoundError(f"Flagged-attributes CSV not found: {flagged_csv_path}")

    config_path, cfg = _load_optional_config(eval_meta, summary_path.parent.parent)

    project_root = summary_path.parent.parent
    if config_path is not None:
        project_root = infer_project_root_from_config(config_path)

    summary_df = _prepare_summary_df(_read_csv_or_empty(summary_path), mode)
    flagged_df = _prepare_flagged_df(_fill_display_setting(_read_csv_or_empty(flagged_csv_path, EMPTY_FLAGGED_COLUMNS), mode))
    target_df = _load_target_manifest_df(eval_meta, summary_path.parent)

    if flagged_images_path.exists():
        image_rows = _load_jsonl(flagged_images_path)
        image_df = _prepare_image_df(_fill_display_setting(pd.DataFrame(image_rows), mode), flagged_df)
    else:
        image_df = _derive_image_df_from_flagged(flagged_df)
    if mode == "rq2":
        ref_summary_df, ref_flagged_df, ref_image_df, ref_meta = _build_rq1_reference_frames_for_rq2(
            summary_path.parent,
            eval_meta,
            summary_df,
        )
        if not ref_summary_df.empty:
            summary_df = pd.concat([summary_df, ref_summary_df], ignore_index=True)
        if not ref_flagged_df.empty:
            flagged_df = pd.concat([flagged_df, ref_flagged_df], ignore_index=True)
        if not ref_image_df.empty:
            image_df = pd.concat([image_df, ref_image_df], ignore_index=True)
        if ref_meta:
            eval_meta.update(ref_meta)

    return _finalize_results_bundle(
        mode,
        project_root=project_root,
        output_dir=summary_path.parent,
        config_path=config_path,
        cfg=cfg,
        eval_meta=eval_meta,
        summary_df=summary_df,
        flagged_df=flagged_df,
        image_df=image_df,
        target_df=target_df,
    )


def _apply_row_filters(
    flagged_df: pd.DataFrame,
    image_df: pd.DataFrame,
    *,
    settings: list[str],
    llms: list[str],
    k_values: list[int],
    scenes: list[str],
    confirmation_statuses: list[str],
    support_labels: list[str],
    auto_verdicts: list[str],
    matched_fields: list[str],
    search_text: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = flagged_df.copy()
    filtered = filtered[filtered["display_setting"].isin(settings)]
    if llms:
        filtered = filtered[filtered["llm_name"].isin(llms)]
    if k_values:
        filtered = filtered[filtered["K"].isin(k_values)]
    if (
        not settings
        or not llms
        or not k_values
        or not scenes
        or not confirmation_statuses
        or not support_labels
        or not auto_verdicts
        or not matched_fields
    ):
        return filtered.iloc[0:0].copy(), image_df.iloc[0:0].copy()
    if scenes:
        filtered = filtered[filtered["scene_family_display"].isin(scenes)]
    if confirmation_statuses:
        filtered = filtered[filtered["confirmation_status"].isin(confirmation_statuses)]
    if support_labels:
        filtered = filtered[filtered["support_label"].isin(support_labels)]
    if auto_verdicts and "auto_verdict_display" in filtered.columns:
        filtered = filtered[filtered["auto_verdict_display"].isin(auto_verdicts)]
    if matched_fields:
        filtered = filtered[filtered["matched_field"].isin(matched_fields)]
    query = _text(search_text).lower()
    if query:
        search_cols = [
            "image_id",
            "attribute_name",
            "attribute_description",
            "scene_family_display",
            "primary_label",
            "secondary_label",
            "ternary_label",
            "background_label",
            "matched_terms",
        ]
        mask = pd.Series(False, index=filtered.index)
        for col in search_cols:
            if col in filtered.columns:
                mask = mask | filtered[col].astype(str).str.lower().str.contains(query, na=False)
        filtered = filtered[mask]

    allowed_pairs = set(filtered["display_pair_key"].tolist()) if "display_pair_key" in filtered.columns else set()
    if allowed_pairs and "display_pair_key" in image_df.columns:
        filtered_images = image_df[image_df["display_pair_key"].isin(allowed_pairs)].copy()
    else:
        filtered_images = image_df.iloc[0:0].copy()
    return filtered, filtered_images


def _summary_table(summary_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    columns = [
        "display_setting",
        "NetSemanticDiscoveryYield",
        "NetSemanticDiscoveryYieldConservative",
        "SlotRecall",
        "AnyRecoveryRate",
        "FullRecoveryRate",
        "AvgRecoveredSlotsPerImage",
        "GroundedPredictionRate",
        "OpenWorldCandidateYield",
        "AutoVerifiedExtraYield",
        "HallucinationYield",
        "AutoVerificationCoverage",
        "AutoVerifiedRateAll",
        "AutoHallucinationRateAll",
        "AutoVerifiedRateDecided",
        "total_available_slots",
        "total_recovered_slots",
        "total_task_aligned_predictions",
        "total_grounded_predictions",
        "total_unmapped_predictions",
        "auto_verified_candidates",
        "auto_likely_hallucination_candidates",
        "auto_uncertain_candidates",
        "flagged_image_pairs",
        "flagged_image_coverage",
        "total_flagged_attributes",
        "avg_flagged_per_embedding",
        "excess_kl_mean",
        "ratio_median",
    ]
    view = summary_df[[col for col in columns if col in summary_df.columns]].copy()
    if "display_setting" in view.columns:
        view = view.rename(columns={"display_setting": _display_entity_name(mode)})
    view = view.rename(
        columns={
            "NetSemanticDiscoveryYield": "Net Discovery / Image",
            "NetSemanticDiscoveryYieldConservative": "Conservative Net / Image",
            "SlotRecall": "Excess Slot Recall",
            "AnyRecoveryRate": "Any Excess Recovery",
            "FullRecoveryRate": "Full Excess Recovery",
            "AvgRecoveredSlotsPerImage": "Excess Recovered / Image",
            "GroundedPredictionRate": "Excess Grounded Rate",
            "OpenWorldCandidateYield": "Unmapped Candidates / Image",
            "AutoVerifiedExtraYield": "Auto-Verified Extra / Image",
            "HallucinationYield": "Likely Hallucinations / Image",
            "AutoVerificationCoverage": "Verifier Coverage",
            "AutoVerifiedRateAll": "Auto-Verified / Unmapped",
            "AutoHallucinationRateAll": "Likely Hallucination / Unmapped",
            "AutoVerifiedRateDecided": "Auto-Verified / Decided",
            "total_available_slots": "Non-Task Slots",
            "total_recovered_slots": "Recovered Non-Task Slots",
            "total_task_aligned_predictions": "Task-Aligned Predictions",
            "total_grounded_predictions": "Excess Grounded Predictions",
            "total_unmapped_predictions": "Unmapped Candidates",
            "auto_verified_candidates": "Auto-Verified Unmapped",
            "auto_likely_hallucination_candidates": "Likely Hallucinations",
            "auto_uncertain_candidates": "Auto-Uncertain",
            "flagged_image_coverage": "Flagged Pair Coverage",
        }
    )
    for col in [
        "Excess Slot Recall",
        "Any Excess Recovery",
        "Full Excess Recovery",
        "Excess Grounded Rate",
        "Verifier Coverage",
        "Auto-Verified / Unmapped",
        "Likely Hallucination / Unmapped",
        "Auto-Verified / Decided",
        "Flagged Pair Coverage",
    ]:
        if col in view.columns:
            view[col] = view[col].map(_format_percent)
    for col in [
        "Net Discovery / Image",
        "Conservative Net / Image",
        "Excess Recovered / Image",
        "Unmapped Candidates / Image",
        "Auto-Verified Extra / Image",
        "Likely Hallucinations / Image",
        "avg_flagged_per_embedding",
        "excess_kl_mean",
        "ratio_median",
    ]:
        if col in view.columns:
            view[col] = view[col].map(_format_float)
    return view


def _download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _scene_metric_format(metric: str, value: Any) -> str:
    spec = SCENE_ANALYSIS_METRICS.get(metric, {})
    kind = spec.get("kind", "float")
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    if kind == "percent":
        return _format_percent(float(value))
    if kind == "int":
        return f"{int(value)}"
    return _format_float(float(value))


def _build_scene_setting_df(summary_df: pd.DataFrame, flagged_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or target_df.empty:
        return pd.DataFrame()

    target_df = _ensure_target_slot_columns(target_df)
    flagged_df = _prepare_flagged_df(flagged_df) if not flagged_df.empty else flagged_df.copy()
    target_images_df = target_df[
        [
            "scene_family_display",
            "image_id",
            "available_slot_count",
            "available_primary_slot",
            "available_secondary_slot",
            "available_ternary_slot",
            "available_background_slot",
        ]
    ].drop_duplicates()
    scene_totals = (
        target_images_df.groupby("scene_family_display", as_index=False)
        .agg(
            target_images=("image_id", "nunique"),
            total_available_slots=("available_slot_count", "sum"),
            available_primary_slots=("available_primary_slot", "sum"),
            available_secondary_slots=("available_secondary_slot", "sum"),
            available_ternary_slots=("available_ternary_slot", "sum"),
            available_background_slots=("available_background_slot", "sum"),
        )
    )
    settings_df = (
        summary_df[[col for col in ["display_setting", "setting_type", "model_name", "llm_name", "K"] if col in summary_df.columns]]
        .drop_duplicates()
        .copy()
    )
    if scene_totals.empty or settings_df.empty:
        return pd.DataFrame()

    scene_totals["__join_key"] = 1
    settings_df["__join_key"] = 1
    scene_setting_df = scene_totals.merge(settings_df, on="__join_key", how="inner").drop(columns=["__join_key"])

    image_status_agg = pd.DataFrame()
    if not flagged_df.empty:
        flagged_agg = (
            flagged_df.groupby(["scene_family_display", "display_setting"], as_index=False)
            .agg(
                flagged_pairs=("display_pair_key", "nunique"),
                unique_flagged_images=("image_id", "nunique"),
                num_flagged=("attribute_name", "size"),
                grounded_predictions=("annotation_grounded", "sum"),
                unmapped_predictions=("is_unmapped_candidate", "sum"),
            )
        )
        flagged_agg["num_valid_flagged"] = flagged_agg["grounded_predictions"]
        flagged_agg["num_invalid_flagged"] = flagged_agg["unmapped_predictions"]
        scene_setting_df = scene_setting_df.merge(
            flagged_agg,
            on=["scene_family_display", "display_setting"],
            how="left",
        )
        if "auto_verdict" in flagged_df.columns:
            unmapped_with_verdict = flagged_df[flagged_df["grounding_label"].map(_text) == "Unmapped candidate"].copy()
            if not unmapped_with_verdict.empty:
                verdict_scene_agg = (
                    unmapped_with_verdict.assign(
                        auto_verified_candidates=unmapped_with_verdict["auto_verdict"].map(_text).eq("auto_verified_extra").astype(int),
                        auto_likely_hallucination_candidates=unmapped_with_verdict["auto_verdict"].map(_text).eq("auto_likely_hallucination").astype(int),
                    )
                    .groupby(["scene_family_display", "display_setting"], as_index=False)
                    .agg(
                        auto_verified_candidates=("auto_verified_candidates", "sum"),
                        auto_likely_hallucination_candidates=("auto_likely_hallucination_candidates", "sum"),
                    )
                )
                scene_setting_df = scene_setting_df.merge(
                    verdict_scene_agg,
                    on=["scene_family_display", "display_setting"],
                    how="left",
                )

        grounded_slots = flagged_df[
            flagged_df["annotation_grounded"] & flagged_df["matched_slot"].isin(list(RECOVERY_SLOT_SPECS.keys()))
        ][["scene_family_display", "display_setting", "image_id", "matched_slot"]].drop_duplicates()
        if not grounded_slots.empty:
            recovered_slot_agg = (
                grounded_slots.groupby(["scene_family_display", "display_setting"], as_index=False)
                .agg(
                    recovered_slots=("matched_slot", "size"),
                    recovered_primary_slots=("matched_slot", lambda s: int((pd.Series(s).astype(str) == "primary").sum())),
                    recovered_secondary_slots=("matched_slot", lambda s: int((pd.Series(s).astype(str) == "secondary").sum())),
                    recovered_ternary_slots=("matched_slot", lambda s: int((pd.Series(s).astype(str) == "ternary").sum())),
                    recovered_background_slots=("matched_slot", lambda s: int((pd.Series(s).astype(str) == "background").sum())),
                )
            )
            scene_setting_df = scene_setting_df.merge(
                recovered_slot_agg,
                on=["scene_family_display", "display_setting"],
                how="left",
            )

            recovered_image_counts = (
                grounded_slots.groupby(["scene_family_display", "display_setting", "image_id"], as_index=False)
                .agg(recovered_slot_count=("matched_slot", "nunique"))
            )
        else:
            recovered_image_counts = pd.DataFrame(columns=["scene_family_display", "display_setting", "image_id", "recovered_slot_count"])

        settings_only = settings_df[["display_setting"]].drop_duplicates().copy()
        target_image_status = target_images_df[["scene_family_display", "image_id", "available_slot_count"]].copy()
        target_image_status["__join_key"] = 1
        settings_only["__join_key"] = 1
        image_status_df = target_image_status.merge(settings_only, on="__join_key", how="inner").drop(columns=["__join_key"])
        image_status_df = image_status_df.merge(
            recovered_image_counts,
            on=["scene_family_display", "display_setting", "image_id"],
            how="left",
        )
        image_status_df["recovered_slot_count"] = pd.to_numeric(
            image_status_df["recovered_slot_count"], errors="coerce"
        ).fillna(0).astype(int)
        image_status_df["any_slot_recovered"] = image_status_df["recovered_slot_count"] > 0
        image_status_df["full_slot_recovered"] = (
            image_status_df["available_slot_count"] > 0
        ) & (image_status_df["recovered_slot_count"] >= image_status_df["available_slot_count"])
        image_status_agg = (
            image_status_df.groupby(["scene_family_display", "display_setting"], as_index=False)
            .agg(
                any_slot_recovered_images=("any_slot_recovered", "sum"),
                full_slot_recovered_images=("full_slot_recovered", "sum"),
            )
        )
        scene_setting_df = scene_setting_df.merge(
            image_status_agg,
            on=["scene_family_display", "display_setting"],
            how="left",
        )

    for col in [
        "flagged_pairs",
        "unique_flagged_images",
        "num_flagged",
        "grounded_predictions",
        "unmapped_predictions",
        "auto_verified_candidates",
        "auto_likely_hallucination_candidates",
        "num_valid_flagged",
        "num_invalid_flagged",
        "recovered_slots",
        "recovered_primary_slots",
        "recovered_secondary_slots",
        "recovered_ternary_slots",
        "recovered_background_slots",
        "any_slot_recovered_images",
        "full_slot_recovered_images",
    ]:
        if col not in scene_setting_df.columns:
            scene_setting_df[col] = 0
        scene_setting_df[col] = pd.to_numeric(scene_setting_df[col], errors="coerce").fillna(0).astype(int)

    image_denom = scene_setting_df["target_images"].where(scene_setting_df["target_images"] > 0)
    slot_denom = scene_setting_df["total_available_slots"].where(scene_setting_df["total_available_slots"] > 0)
    scene_setting_df["GroundedPredictionRate"] = (
        scene_setting_df["grounded_predictions"] / scene_setting_df["num_flagged"].where(scene_setting_df["num_flagged"] > 0)
    ).fillna(0.0)
    scene_setting_df["GroundedPredictionYield"] = (scene_setting_df["grounded_predictions"] / image_denom).fillna(0.0)
    scene_setting_df["OpenWorldCandidateYield"] = (scene_setting_df["unmapped_predictions"] / image_denom).fillna(0.0)
    scene_setting_df["SlotRecall"] = (scene_setting_df["recovered_slots"] / slot_denom).fillna(0.0)
    scene_setting_df["AnyRecoveryRate"] = (scene_setting_df["any_slot_recovered_images"] / image_denom).fillna(0.0)
    scene_setting_df["FullRecoveryRate"] = (scene_setting_df["full_slot_recovered_images"] / image_denom).fillna(0.0)
    scene_setting_df["AvgRecoveredSlotsPerImage"] = (scene_setting_df["recovered_slots"] / image_denom).fillna(0.0)
    scene_setting_df = _apply_net_discovery_metrics(
        scene_setting_df,
        recovered_col="recovered_slots",
        verified_col="auto_verified_candidates",
        hallucination_col="auto_likely_hallucination_candidates",
        image_denom=image_denom,
    )
    scene_setting_df["flagged_pair_rate"] = (scene_setting_df["flagged_pairs"] / image_denom).fillna(0.0)
    scene_setting_df["avg_flagged_per_image"] = (scene_setting_df["num_flagged"] / image_denom).fillna(0.0)
    scene_setting_df["SVR"] = scene_setting_df["GroundedPredictionRate"]
    scene_setting_df["ESY"] = scene_setting_df["GroundedPredictionYield"]
    scene_setting_df["HCE"] = scene_setting_df["OpenWorldCandidateYield"]
    return scene_setting_df


def _build_scene_overview_df(scene_setting_df: pd.DataFrame, flagged_df: pd.DataFrame) -> pd.DataFrame:
    if scene_setting_df.empty:
        return pd.DataFrame()
    overview_df = (
        scene_setting_df.groupby("scene_family_display", as_index=False)
        .agg(
            target_images=("target_images", "first"),
            total_available_slots=("total_available_slots", "first"),
            settings_compared=("display_setting", "nunique"),
            total_flagged_pairs=("flagged_pairs", "sum"),
            total_flagged_attributes=("num_flagged", "sum"),
            total_grounded_predictions=("grounded_predictions", "sum"),
            total_unmapped_predictions=("unmapped_predictions", "sum"),
            total_recovered_slots=("recovered_slots", "sum"),
            total_auto_verified_candidates=("auto_verified_candidates", "sum"),
            total_auto_likely_hallucination_candidates=("auto_likely_hallucination_candidates", "sum"),
            mean_NetSemanticDiscoveryYield=("NetSemanticDiscoveryYield", "mean"),
            mean_NetSemanticDiscoveryYieldConservative=("NetSemanticDiscoveryYieldConservative", "mean"),
            mean_SlotRecall=("SlotRecall", "mean"),
            mean_AnyRecoveryRate=("AnyRecoveryRate", "mean"),
            mean_FullRecoveryRate=("FullRecoveryRate", "mean"),
            mean_GroundedPredictionRate=("GroundedPredictionRate", "mean"),
            mean_OpenWorldCandidateYield=("OpenWorldCandidateYield", "mean"),
        )
    )
    if not flagged_df.empty:
        unique_flagged_images = (
            flagged_df.groupby("scene_family_display", as_index=False)["image_id"]
            .nunique()
            .rename(columns={"image_id": "unique_flagged_images"})
        )
        overview_df = overview_df.merge(unique_flagged_images, on="scene_family_display", how="left")
    else:
        overview_df["unique_flagged_images"] = 0
    overview_df["unique_flagged_images"] = overview_df["unique_flagged_images"].fillna(0).astype(int)
    return overview_df.sort_values(
        ["mean_NetSemanticDiscoveryYield", "mean_SlotRecall", "scene_family_display"],
        ascending=[False, False, True],
    )


def _scene_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scene_family_display",
        "target_images",
        "total_available_slots",
        "unique_flagged_images",
        "settings_compared",
        "total_flagged_pairs",
        "total_flagged_attributes",
        "total_recovered_slots",
        "total_grounded_predictions",
        "total_unmapped_predictions",
        "total_auto_verified_candidates",
        "total_auto_likely_hallucination_candidates",
        "mean_NetSemanticDiscoveryYield",
        "mean_NetSemanticDiscoveryYieldConservative",
        "mean_SlotRecall",
        "mean_AnyRecoveryRate",
        "mean_FullRecoveryRate",
        "mean_GroundedPredictionRate",
        "mean_OpenWorldCandidateYield",
    ]
    view = df[[col for col in columns if col in df.columns]].copy()
    view = view.rename(
        columns={
            "scene_family_display": "Scene Family",
            "target_images": "Target Images",
            "total_available_slots": "Annotated Slots",
            "unique_flagged_images": "Unique Flagged Images",
            "settings_compared": "Settings Compared",
            "total_flagged_pairs": "Flagged Pairs",
            "total_flagged_attributes": "Flagged Attributes",
            "total_recovered_slots": "Recovered Slots",
            "total_grounded_predictions": "Grounded Predictions",
            "total_unmapped_predictions": "Unmapped Candidates",
            "total_auto_verified_candidates": "Auto-Verified Extra",
            "total_auto_likely_hallucination_candidates": "Likely Hallucinations",
            "mean_NetSemanticDiscoveryYield": "Mean Net Discovery / Image",
            "mean_NetSemanticDiscoveryYieldConservative": "Mean Conservative Net / Image",
            "mean_SlotRecall": "Mean Excess Slot Recall",
            "mean_AnyRecoveryRate": "Mean Any Excess Recovery",
            "mean_FullRecoveryRate": "Mean Full Excess Recovery",
            "mean_GroundedPredictionRate": "Mean Excess Grounded Rate",
            "mean_OpenWorldCandidateYield": "Mean Unmapped Candidates / Image",
        }
    )
    for col in ["Mean Excess Slot Recall", "Mean Any Excess Recovery", "Mean Full Excess Recovery", "Mean Excess Grounded Rate"]:
        if col in view.columns:
            view[col] = view[col].map(_format_percent)
    for col in ["Mean Net Discovery / Image", "Mean Conservative Net / Image", "Mean Unmapped Candidates / Image"]:
        if col in view.columns:
            view[col] = view[col].map(_format_float)
    return view


def _scene_setting_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    columns = [
        "scene_family_display",
        "display_setting",
        "target_images",
        "total_available_slots",
        "recovered_slots",
        "SlotRecall",
        "AnyRecoveryRate",
        "FullRecoveryRate",
        "AvgRecoveredSlotsPerImage",
        "GroundedPredictionRate",
        "OpenWorldCandidateYield",
        "AutoVerifiedExtraYield",
        "HallucinationYield",
        "NetSemanticDiscoveryYield",
        "NetSemanticDiscoveryYieldConservative",
        "flagged_pairs",
        "flagged_pair_rate",
        "num_flagged",
        "grounded_predictions",
        "unmapped_predictions",
    ]
    view = df[[col for col in columns if col in df.columns]].copy()
    view = view.rename(
        columns={
            "scene_family_display": "Scene Family",
            "display_setting": _display_entity_name(mode),
            "target_images": "Target Images",
            "total_available_slots": "Annotated Slots",
            "recovered_slots": "Recovered Non-Task Slots",
            "SlotRecall": "Excess Slot Recall",
            "AnyRecoveryRate": "Any Excess Recovery",
            "FullRecoveryRate": "Full Excess Recovery",
            "AvgRecoveredSlotsPerImage": "Excess Recovered / Image",
            "GroundedPredictionRate": "Excess Grounded Rate",
            "OpenWorldCandidateYield": "Unmapped Candidates / Image",
            "AutoVerifiedExtraYield": "Auto-Verified Extra / Image",
            "HallucinationYield": "Likely Hallucinations / Image",
            "NetSemanticDiscoveryYield": "Net Discovery / Image",
            "NetSemanticDiscoveryYieldConservative": "Conservative Net / Image",
            "flagged_pairs": "Flagged Pairs",
            "flagged_pair_rate": "Pair Coverage",
            "num_flagged": "Flagged Attributes",
            "grounded_predictions": "Excess Grounded Predictions",
            "unmapped_predictions": "Unmapped Candidates",
        }
    )
    for col in ["Excess Slot Recall", "Any Excess Recovery", "Full Excess Recovery", "Excess Grounded Rate", "Pair Coverage"]:
        if col in view.columns:
            view[col] = view[col].map(_format_percent)
    for col in [
        "Excess Recovered / Image",
        "Unmapped Candidates / Image",
        "Auto-Verified Extra / Image",
        "Likely Hallucinations / Image",
        "Net Discovery / Image",
        "Conservative Net / Image",
    ]:
        if col in view.columns:
            view[col] = view[col].map(_format_float)
    return view


def _chart_or_info(chart, *, fallback_message: str = "Chart unavailable.") -> None:
    if chart is None:
        st.info(fallback_message)
    else:
        st.altair_chart(chart, width="stretch")


def build_overall_discovery_chart(summary_df: pd.DataFrame, mode: str):
    if alt is None or summary_df.empty or "NetSemanticDiscoveryYield" not in summary_df.columns:
        return None
    data = summary_df.copy()
    color = (
        alt.Color("display_setting:N", legend=None)
        if mode == "rq1"
        else alt.Color("setting_type:N", title="Ablation")
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("display_setting:N", sort=_sort_settings(data["display_setting"].dropna().unique().tolist()), title=None),
            y=alt.Y("NetSemanticDiscoveryYield:Q", title="Net discovery / image"),
            color=color,
            tooltip=[
                alt.Tooltip("display_setting:N", title=_display_entity_name(mode)),
                alt.Tooltip("llm_name:N", title="LLM"),
                alt.Tooltip("K:Q", title="K"),
                alt.Tooltip("NetSemanticDiscoveryYield:Q", title="Net discovery / image", format=".3f"),
                alt.Tooltip("NetSemanticDiscoveryYieldConservative:Q", title="Conservative net / image", format=".3f"),
                alt.Tooltip("SlotRecall:Q", title="Excess slot recall", format=".1%"),
                alt.Tooltip("AutoVerifiedExtraYield:Q", title="Auto-verified extra / image", format=".3f"),
                alt.Tooltip("HallucinationYield:Q", title="Likely hallucinations / image", format=".3f"),
            ],
        )
        .properties(height=320, title=f"Net Semantic Discovery Yield by {_display_entity_name(mode)}")
    )


def build_svr_chart(summary_df: pd.DataFrame, mode: str):
    if alt is None or summary_df.empty:
        return None
    data = summary_df.copy()
    color = (
        alt.Color("display_setting:N", legend=None)
        if mode == "rq1"
        else alt.Color("setting_type:N", title="Ablation")
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("display_setting:N", sort=_sort_settings(data["display_setting"].dropna().unique().tolist()), title=None),
            y=alt.Y("SlotRecall:Q", title="Excess slot recall", axis=alt.Axis(format="%")),
            color=color,
            tooltip=[
                alt.Tooltip("display_setting:N", title=_display_entity_name(mode)),
                alt.Tooltip("llm_name:N", title="LLM"),
                alt.Tooltip("K:Q", title="K"),
                alt.Tooltip("SlotRecall:Q", title="Excess slot recall", format=".1%"),
                alt.Tooltip("AnyRecoveryRate:Q", title="Any excess recovery", format=".1%"),
                alt.Tooltip("FullRecoveryRate:Q", title="Full excess recovery", format=".1%"),
                alt.Tooltip("AvgRecoveredSlotsPerImage:Q", title="Excess recovered slots / image", format=".3f"),
                alt.Tooltip("total_flagged_attributes:Q", title="Flagged"),
            ],
        )
        .properties(height=320, title=f"Excess Slot Recall by {_display_entity_name(mode)}")
    )


def build_tradeoff_chart(summary_df: pd.DataFrame, mode: str):
    required = ["NetSemanticDiscoveryYield", "OpenWorldCandidateYield", "AvgRecoveredSlotsPerImage"]
    if alt is None or summary_df.empty or any(col not in summary_df.columns for col in required):
        return None
    data = summary_df.copy()
    color = (
        alt.Color("display_setting:N", title=_display_entity_name(mode))
        if mode == "rq1"
        else alt.Color("setting_type:N", title="Ablation")
    )
    return (
        alt.Chart(data)
        .mark_circle(size=180, opacity=0.85)
        .encode(
            x=alt.X("OpenWorldCandidateYield:Q", title="Unmapped candidates / image"),
            y=alt.Y("NetSemanticDiscoveryYield:Q", title="Net discovery / image"),
            size=alt.Size("AvgRecoveredSlotsPerImage:Q", title="Excess recovered slots / image"),
            color=color,
            tooltip=[
                alt.Tooltip("display_setting:N", title=_display_entity_name(mode)),
                alt.Tooltip("llm_name:N", title="LLM"),
                alt.Tooltip("K:Q", title="K"),
                alt.Tooltip("NetSemanticDiscoveryYield:Q", title="Net discovery / image", format=".3f"),
                alt.Tooltip("NetSemanticDiscoveryYieldConservative:Q", title="Conservative net / image", format=".3f"),
                alt.Tooltip("SlotRecall:Q", title="Excess slot recall", format=".1%"),
                alt.Tooltip("AnyRecoveryRate:Q", title="Any excess recovery", format=".1%"),
                alt.Tooltip("FullRecoveryRate:Q", title="Full excess recovery", format=".1%"),
                alt.Tooltip("GroundedPredictionRate:Q", title="Excess grounded rate", format=".1%"),
                alt.Tooltip("OpenWorldCandidateYield:Q", title="Unmapped candidates / image", format=".3f"),
            ],
        )
        .properties(height=320, title=f"{_display_entity_name(mode)} Trade-off: Net Discovery vs. Unmapped Candidate Yield")
    )


def build_validity_stack_chart(summary_df: pd.DataFrame, mode: str):
    if alt is None or summary_df.empty:
        return None
    required = ["display_setting", "total_task_aligned_predictions", "total_grounded_predictions", "total_unmapped_predictions"]
    if any(col not in summary_df.columns for col in required):
        return None
    plot_df = summary_df[required].melt(
        id_vars="display_setting",
        var_name="bucket",
        value_name="count",
    )
    plot_df["bucket"] = plot_df["bucket"].map(
        {
            "total_task_aligned_predictions": "Task-aligned recovery",
            "total_grounded_predictions": "Excess grounded recovery",
            "total_unmapped_predictions": "Unmapped candidate",
        }
    )
    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("display_setting:N", sort=_sort_settings(plot_df["display_setting"].dropna().unique().tolist()), title=None),
            y=alt.Y("count:Q", title="Flagged attributes"),
            color=alt.Color(
                "bucket:N",
                title="Prediction type",
                scale=alt.Scale(
                    domain=["Task-aligned recovery", "Excess grounded recovery", "Unmapped candidate"],
                    range=["#4C78A8", "#2F855A", "#C53030"],
                ),
            ),
            tooltip=["display_setting:N", "bucket:N", alt.Tooltip("count:Q", title="Count")],
        )
        .properties(height=320, title="Task-Aligned vs. Excess vs. Unmapped Flagged Attributes")
    )


def build_auto_verification_chart(summary_df: pd.DataFrame, mode: str):
    if alt is None or summary_df.empty:
        return None
    required = [
        "display_setting",
        "auto_verified_candidates",
        "auto_likely_hallucination_candidates",
        "auto_uncertain_candidates",
    ]
    if any(col not in summary_df.columns for col in required):
        return None
    if float(pd.to_numeric(summary_df["auto_verified_candidates"], errors="coerce").fillna(0).sum()) <= 0 and float(
        pd.to_numeric(summary_df["auto_likely_hallucination_candidates"], errors="coerce").fillna(0).sum()
    ) <= 0 and float(pd.to_numeric(summary_df["auto_uncertain_candidates"], errors="coerce").fillna(0).sum()) <= 0:
        return None

    plot_df = summary_df[required].melt(
        id_vars="display_setting",
        var_name="bucket",
        value_name="count",
    )
    plot_df["bucket"] = plot_df["bucket"].map(
        {
            "auto_verified_candidates": "Auto-verified extra",
            "auto_likely_hallucination_candidates": "Likely hallucination",
            "auto_uncertain_candidates": "Auto-uncertain",
        }
    )
    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("display_setting:N", sort=_sort_settings(plot_df["display_setting"].dropna().unique().tolist()), title=None),
            y=alt.Y("count:Q", title="Verified unmapped candidates"),
            color=alt.Color(
                "bucket:N",
                title="Auto verifier",
                scale=alt.Scale(
                    domain=["Auto-verified extra", "Likely hallucination", "Auto-uncertain"],
                    range=["#2F855A", "#C53030", "#DD6B20"],
                ),
            ),
            tooltip=["display_setting:N", "bucket:N", alt.Tooltip("count:Q", title="Count")],
        )
        .properties(height=320, title=f"Automatic Verification of Unmapped Candidates by {_display_entity_name(mode)}")
    )


def build_scene_chart(image_df: pd.DataFrame, mode: str):
    if alt is None or image_df.empty:
        return None
    plot_df = (
        image_df.groupby(["scene_family_display", "display_setting"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    scene_order = (
        plot_df.groupby("scene_family_display")["count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y("scene_family_display:N", sort=scene_order, title=None),
            x=alt.X("count:Q", title=f"Flagged {_display_pair_name(mode)}"),
            color=alt.Color("display_setting:N", title=_display_entity_name(mode)),
            tooltip=["scene_family_display:N", "display_setting:N", alt.Tooltip("count:Q", title="Count")],
        )
        .properties(height=360, title="Flagged Image Coverage by Scene Family")
    )


def build_scene_metric_heatmap(scene_setting_df: pd.DataFrame, metric: str, mode: str):
    if alt is None or scene_setting_df.empty:
        return None
    spec = SCENE_ANALYSIS_METRICS.get(metric, {})
    scene_order = (
        scene_setting_df.groupby("scene_family_display")["target_images"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    setting_order = _sort_settings(scene_setting_df["display_setting"].dropna().unique().tolist())
    return (
        alt.Chart(scene_setting_df)
        .mark_rect()
        .encode(
            x=alt.X("display_setting:N", sort=setting_order, title=None),
            y=alt.Y("scene_family_display:N", sort=scene_order, title=None),
            color=alt.Color(f"{metric}:Q", title=spec.get("label", metric)),
            tooltip=[
                alt.Tooltip("scene_family_display:N", title="Scene Family"),
                alt.Tooltip("display_setting:N", title=_display_entity_name(mode)),
                alt.Tooltip("target_images:Q", title="Target Images"),
                alt.Tooltip("total_available_slots:Q", title="Annotated Slots"),
                alt.Tooltip("recovered_slots:Q", title="Recovered Slots"),
                alt.Tooltip("num_flagged:Q", title="Flagged Attributes"),
                alt.Tooltip("grounded_predictions:Q", title="Grounded Predictions"),
                alt.Tooltip("unmapped_predictions:Q", title="Unmapped Candidates"),
                alt.Tooltip(f"{metric}:Q", title=spec.get("label", metric), format=spec.get("tooltip_format")),
            ],
        )
        .properties(height=max(260, len(scene_order) * 28), title=f"{spec.get('title', metric)} by Scene Family and {_display_entity_name(mode)}")
    )


def build_scene_focus_chart(scene_setting_df: pd.DataFrame, metric: str, focus_setting: str, mode: str):
    if alt is None or scene_setting_df.empty or not focus_setting:
        return None
    spec = SCENE_ANALYSIS_METRICS.get(metric, {})
    plot_df = scene_setting_df[scene_setting_df["display_setting"] == focus_setting].copy()
    if plot_df.empty:
        return None
    scene_order = (
        plot_df.sort_values([metric, "scene_family_display"], ascending=[False, True])["scene_family_display"]
        .tolist()
    )
    return (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            y=alt.Y("scene_family_display:N", sort=scene_order, title=None),
            x=alt.X(f"{metric}:Q", title=spec.get("title", metric)),
            color=alt.Color(f"{metric}:Q", title=spec.get("label", metric)),
            tooltip=[
                alt.Tooltip("scene_family_display:N", title="Scene Family"),
                alt.Tooltip("target_images:Q", title="Target Images"),
                alt.Tooltip("total_available_slots:Q", title="Annotated Slots"),
                alt.Tooltip("recovered_slots:Q", title="Recovered Slots"),
                alt.Tooltip("num_flagged:Q", title="Flagged Attributes"),
                alt.Tooltip("grounded_predictions:Q", title="Grounded Predictions"),
                alt.Tooltip("unmapped_predictions:Q", title="Unmapped Candidates"),
                alt.Tooltip(f"{metric}:Q", title=spec.get("label", metric), format=spec.get("tooltip_format")),
            ],
        )
        .properties(height=max(260, len(scene_order) * 28), title=f"{focus_setting}: {spec.get('title', metric)} by Scene Family")
    )


def build_top_attributes_chart(flagged_df: pd.DataFrame, top_n: int):
    if alt is None or flagged_df.empty:
        return None
    plot_df = (
        flagged_df.groupby(["attribute_name", "support_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    top_names = (
        plot_df.groupby("attribute_name")["count"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    )
    plot_df = plot_df[plot_df["attribute_name"].isin(top_names)]
    attribute_order = (
        plot_df.groupby("attribute_name")["count"].sum().sort_values(ascending=False).index.tolist()
    )
    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y("attribute_name:N", sort=attribute_order, title=None),
            x=alt.X("count:Q", title="Occurrences"),
            color=alt.Color(
                "support_label:N",
                title="Prediction type",
                scale=alt.Scale(domain=SUPPORT_LABEL_COLOR_DOMAIN, range=SUPPORT_LABEL_COLOR_RANGE),
            ),
            tooltip=["attribute_name:N", "support_label:N", alt.Tooltip("count:Q", title="Count")],
        )
        .properties(height=max(260, top_n * 22), title=f"Top {top_n} Flagged Attributes")
    )


def build_evidence_chart(flagged_df: pd.DataFrame, mode: str):
    if alt is None or flagged_df.empty:
        return None, None
    plot_df = flagged_df.copy()
    plot_df = plot_df[(plot_df["excess_kl"].notna()) & (plot_df["excess_kl"] >= 0)].copy()
    finite_ratio_mask = (~plot_df["ratio_is_infinite"]) & plot_df["excess_to_task_kl_ratio"].notna()
    plot_df = plot_df[finite_ratio_mask & (plot_df["excess_to_task_kl_ratio"] > 0)].copy()
    if plot_df.empty:
        return None, None

    clip_value = float(plot_df["excess_to_task_kl_ratio"].quantile(0.99))
    if clip_value <= 0:
        clip_value = float(plot_df["excess_to_task_kl_ratio"].max())
    plot_df["ratio_plot"] = plot_df["excess_to_task_kl_ratio"].clip(upper=clip_value)

    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=70, opacity=0.55)
        .encode(
            x=alt.X("excess_kl:Q", title="Excess KL"),
            y=alt.Y("ratio_plot:Q", title="Excess / task KL ratio", scale=alt.Scale(type="log")),
            color=alt.Color(
                "support_label:N",
                title="Prediction type",
                scale=alt.Scale(domain=SUPPORT_LABEL_COLOR_DOMAIN, range=SUPPORT_LABEL_COLOR_RANGE),
            ),
            tooltip=[
                "image_id:N",
                "display_setting:N",
                "attribute_name:N",
                alt.Tooltip("support_label:N", title="Type"),
                alt.Tooltip("confirmation_status:N", title="Confirmation"),
                alt.Tooltip("excess_kl:Q", title="Excess KL", format=".4f"),
                alt.Tooltip("task_kl:Q", title="Task KL", format=".4f"),
                alt.Tooltip("excess_to_task_kl_ratio:Q", title="Ratio", format=".4f"),
            ],
        )
        .properties(height=360, title="Evidence Strength per Flagged Attribute")
    )
    return chart, clip_value


def build_field_breakdown_chart(summary_df: pd.DataFrame, mode: str):
    if alt is None or summary_df.empty:
        return None
    field_specs = [
        ("primary_recall", "Primary", "recovered_primary_slots", "available_primary_slots"),
        ("secondary_recall", "Secondary", "recovered_secondary_slots", "available_secondary_slots"),
        ("ternary_recall", "Ternary", "recovered_ternary_slots", "available_ternary_slots"),
        ("background_recall", "Background", "recovered_background_slots", "available_background_slots"),
    ]
    frames: list[pd.DataFrame] = []
    for recall_col, label, recovered_col, available_col in field_specs:
        if recall_col not in summary_df.columns:
            continue
        cols = [col for col in ["display_setting", recall_col, recovered_col, available_col] if col in summary_df.columns]
        frame = summary_df[cols].copy()
        frame["field"] = label
        frame["recall"] = pd.to_numeric(frame.get(recall_col), errors="coerce").fillna(0.0)
        frame["recovered"] = pd.to_numeric(frame.get(recovered_col), errors="coerce").fillna(0).astype(int)
        frame["available"] = pd.to_numeric(frame.get(available_col), errors="coerce").fillna(0).astype(int)
        frames.append(frame[["display_setting", "field", "recall", "recovered", "available"]])
    if not frames:
        return None
    plot_df = pd.concat(frames, ignore_index=True)
    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("display_setting:N", sort=_sort_settings(plot_df["display_setting"].dropna().unique().tolist()), title=None),
            y=alt.Y("recall:Q", title="Field recall", axis=alt.Axis(format="%")),
            color=alt.Color("field:N", title="Recovered slot"),
            tooltip=[
                "display_setting:N",
                "field:N",
                alt.Tooltip("recall:Q", title="Recall", format=".1%"),
                alt.Tooltip("recovered:Q", title="Recovered slots"),
                alt.Tooltip("available:Q", title="Annotated slots"),
            ],
        )
        .properties(height=320, title="Excess Slot Recall by Ground-Truth Field")
    )


def _resolve_image(record: Mapping[str, Any], bundle: ResultsBundle) -> Path | None:
    try:
        return resolve_image_path_from_record(record, bundle.cfg, bundle.project_root)
    except Exception:
        image_path = _text(record.get("image_path"))
        return Path(image_path).expanduser().resolve() if image_path else None


def _attribute_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "attribute_name",
        "support_label",
        "auto_verdict_display",
        "auto_valid_probability",
        "matched_field",
        "confirmation_status",
        "support_score",
        "excess_kl",
        "task_kl",
        "excess_to_task_kl_ratio",
        "attribute_description",
    ]
    view = df[[col for col in columns if col in df.columns]].copy()
    for col in ["support_score", "excess_kl", "task_kl", "excess_to_task_kl_ratio", "auto_valid_probability"]:
        if col in view.columns:
            view[col] = view[col].map(_format_float)
    return view


def _mode_results_caption(mode: str) -> str:
    if mode == "rq1":
        return "Model-level inference-attack report for `eval_rq1.*`, `flagged_attributes_rq1.csv`, and `flagged_images_rq1.jsonl`, centered on annotated slot recovery and unmapped discovery yield."
    return "Ablation-level inference-attack report for `eval_rq2.*`, `flagged_attributes_rq2.csv`, and `flagged_images_rq2.jsonl`, centered on annotated slot recovery and unmapped discovery yield."


st.set_page_config(page_title="Results Dashboard", layout="wide")
st.title("Semantic Leakage Results Dashboard")
st.caption("Switch between RQ1 and RQ2 results from the sidebar.")

with st.sidebar:
    st.header("Inputs")
    selected_mode_label = st.radio("Result set", ["RQ1", "RQ2"], horizontal=True)
    selected_mode = selected_mode_label.lower()
    output_dir = Path(st.text_input("Output directory", str(DEFAULT_OUTPUT_DIR))).expanduser()
    auto_merge_preserved_runs = False
    if selected_mode == "rq2":
        auto_merge_preserved_runs = st.checkbox(
            "Auto-merge preserved RQ2 runs",
            value=True,
            help="Reconstruct RQ2 directly from preserved output/rq2_* run directories instead of relying on the root aggregate files.",
        )
        if auto_merge_preserved_runs:
            st.caption("When enabled, the app ignores the root `eval_rq2.*` files and rebuilds the dashboard from preserved run directories.")
    summary_path = st.text_input(
        "Summary CSV",
        str(output_dir / f"eval_{selected_mode}.csv"),
        key=f"summary_path_{selected_mode}",
    )
    eval_json_path = st.text_input(
        "Evaluation JSON",
        str(output_dir / f"eval_{selected_mode}.json"),
        key=f"eval_json_path_{selected_mode}",
    )
    flagged_csv_path = st.text_input(
        "Flagged attributes CSV",
        str(output_dir / f"flagged_attributes_{selected_mode}.csv"),
        key=f"flagged_csv_path_{selected_mode}",
    )
    flagged_images_path = st.text_input(
        "Flagged images JSONL",
        str(output_dir / f"flagged_images_{selected_mode}.jsonl"),
        key=f"flagged_images_path_{selected_mode}",
    )
    verification_csv_path = st.text_input(
        "Unmapped verification CSV",
        str(output_dir / f"flagged_attributes_{selected_mode}.unmapped_verification.csv"),
        key=f"verification_csv_path_{selected_mode}",
    )
    verification_summary_path = st.text_input(
        "Unmapped verification summary JSON",
        str(output_dir / f"flagged_attributes_{selected_mode}.unmapped_verification.summary.json"),
        key=f"verification_summary_path_{selected_mode}",
    )
    if st.button("Reload data"):
        load_results_bundle.clear()
        load_unmapped_verification.clear()

try:
    bundle = load_results_bundle(
        selected_mode,
        summary_path,
        eval_json_path,
        flagged_csv_path,
        flagged_images_path,
        auto_merge_preserved_runs=auto_merge_preserved_runs,
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

summary_df = _prepare_summary_df(bundle.summary_df.copy(), bundle.mode)
flagged_df = _prepare_flagged_df(bundle.flagged_df.copy())
verification_df_raw, verification_meta = load_unmapped_verification(
    verification_csv_path,
    verification_summary_path,
    bundle.mode,
)
verification_df = _prepare_verification_df(verification_df_raw, bundle.mode)
flagged_df = _attach_verification_to_flagged(flagged_df, verification_df)
image_df = _prepare_image_df(bundle.image_df.copy(), flagged_df)
target_df = _ensure_target_slot_columns(bundle.target_df.copy())
summary_df = _backfill_summary_attack_metrics(summary_df, flagged_df, image_df, target_df)
summary_df = _backfill_summary_unmapped_verification_metrics(summary_df, flagged_df)
setting_options = _sort_settings(summary_df["display_setting"].dropna().unique().tolist())
llm_options = sorted([value for value in flagged_df["llm_name"].dropna().unique().tolist() if _text(value)])
k_options = sorted({int(value) for value in flagged_df["K"].dropna().tolist()})

st.subheader(bundle.mode.upper())
st.caption(_mode_results_caption(bundle.mode))
if bundle.data_source == "preserved_runs":
    message = f"Loaded {bundle.discovered_run_count} preserved RQ2 runs from `{bundle.output_dir}`."
    if bundle.eval_meta.get("includes_rq1_reference"):
        labels = [label for label in bundle.eval_meta.get("rq1_reference_labels", []) if _text(label)]
        if labels:
            message += f" Added RQ1 reference views: {', '.join(f'`{label}`' for label in labels)}."
    st.info(message)
if not verification_df.empty:
    verifier_message = f"Loaded {len(verification_df)} automatic unmapped-verification rows"
    clip_model = _text(verification_meta.get("clip_model_name"))
    if clip_model:
        verifier_message += f" using `{clip_model}`"
    verifier_message += "."
    st.info(verifier_message)

with st.sidebar:
    st.header(f"{_display_entity_name(bundle.mode)} Filter")
    selected_settings = st.multiselect(
        f"{_display_entity_name(bundle.mode)}s",
        setting_options,
        default=setting_options,
    )
    st.header("Row Filters")
    if len(llm_options) > 1:
        selected_llms = st.multiselect("LLMs", llm_options, default=llm_options)
    else:
        selected_llms = llm_options
        if llm_options:
            st.caption(f"LLM: `{llm_options[0]}`")
    if len(k_options) > 1:
        selected_k_values = st.multiselect("K values", k_options, default=k_options)
    else:
        selected_k_values = k_options
        if k_options:
            st.caption(f"K: `{k_options[0]}`")
    scene_options_source = target_df if not target_df.empty else flagged_df
    scene_options = sorted(scene_options_source["scene_family_display"].dropna().unique().tolist()) if "scene_family_display" in scene_options_source.columns else []
    confirmation_options = sorted(flagged_df["confirmation_status"].dropna().unique().tolist())
    support_options = sorted(flagged_df["support_label"].dropna().unique().tolist())
    auto_verdict_options = sorted(
        [value for value in flagged_df["auto_verdict_display"].dropna().unique().tolist() if _text(value)]
    ) if "auto_verdict_display" in flagged_df.columns else []
    matched_field_options = sorted(flagged_df["matched_field"].dropna().unique().tolist())
    selected_scenes = st.multiselect("Scene families", scene_options, default=scene_options)
    selected_confirmations = st.multiselect("Confirmation status", confirmation_options, default=confirmation_options)
    selected_support_labels = st.multiselect("Grounding status", support_options, default=support_options)
    selected_auto_verdicts = st.multiselect("Auto verification", auto_verdict_options, default=auto_verdict_options)
    selected_matched_fields = st.multiselect("Matched slot", matched_field_options, default=matched_field_options)
    search_text = st.text_input("Search rows", "")
    top_n = st.slider("Top attributes", min_value=5, max_value=30, value=15, step=1)
    st.caption(f"Row filters apply to breakdowns, tables, and the sample inspector. Summary metrics only use the {_display_entity_name(bundle.mode).lower()} filter.")

if not selected_settings:
    st.warning(f"Select at least one {_display_entity_name(bundle.mode).lower()}.")
    st.stop()

summary_filtered = summary_df[summary_df["display_setting"].isin(selected_settings)].copy()
flagged_filtered, image_filtered = _apply_row_filters(
    flagged_df,
    image_df,
    settings=selected_settings,
    llms=selected_llms,
    k_values=selected_k_values,
    scenes=selected_scenes,
    confirmation_statuses=selected_confirmations,
    support_labels=selected_support_labels,
    auto_verdicts=selected_auto_verdicts,
    matched_fields=selected_matched_fields,
    search_text=search_text,
)
if target_df.empty:
    target_filtered = target_df.copy()
elif selected_scenes:
    target_filtered = target_df[target_df["scene_family_display"].isin(selected_scenes)].copy()
else:
    target_filtered = target_df.iloc[0:0].copy()
summary_view_df = summary_filtered.copy()
scene_setting_df = _build_scene_setting_df(summary_filtered, flagged_filtered, target_filtered)
scene_setting_view_df = scene_setting_df.copy()
scene_overview_df = _build_scene_overview_df(scene_setting_view_df, flagged_filtered)

meta_col1, meta_col2, meta_col3, meta_col4, meta_col5, meta_col6, meta_col7, meta_col8, meta_col9 = st.columns(9)
meta_col1.metric(f"{_display_entity_name(bundle.mode)}s", f"{len(summary_view_df)}")
meta_col2.metric(f"Flagged {_display_pair_name(bundle.mode)}", f"{len(image_filtered)}")
meta_col3.metric("Unique flagged images", f"{flagged_filtered['image_id'].nunique()}")
meta_col4.metric("Flagged attributes", f"{len(flagged_filtered)}")
net_metric_label = "Net discovery / image" if len(summary_view_df) == 1 else "Best net discovery / image"
net_metric_value = _format_float(float(pd.to_numeric(summary_view_df["NetSemanticDiscoveryYield"], errors="coerce").max())) if (
    not summary_view_df.empty and "NetSemanticDiscoveryYield" in summary_view_df.columns
) else "-"
meta_col5.metric(net_metric_label, net_metric_value)
meta_col6.metric(
    "Conservative net / image",
    _format_float(float(pd.to_numeric(summary_view_df["NetSemanticDiscoveryYieldConservative"], errors="coerce").max())) if (
        not summary_view_df.empty and "NetSemanticDiscoveryYieldConservative" in summary_view_df.columns
    ) else "-",
)
meta_col7.metric(
    "Excess grounded rate",
    _format_percent(float(flagged_filtered["annotation_grounded"].mean())) if not flagged_filtered.empty else "-",
)
unmapped_filtered = flagged_filtered[flagged_filtered["grounding_label"].map(_text) == "Unmapped candidate"].copy()
reviewed_unmapped_filtered = unmapped_filtered[unmapped_filtered["auto_verdict"].map(_text).ne("")].copy() if "auto_verdict" in unmapped_filtered.columns else unmapped_filtered.iloc[0:0].copy()
decided_unmapped_filtered = reviewed_unmapped_filtered[
    reviewed_unmapped_filtered["auto_verdict"].isin(["auto_verified_extra", "auto_likely_hallucination"])
].copy() if not reviewed_unmapped_filtered.empty else reviewed_unmapped_filtered
meta_col8.metric(
    "Auto-Verified / Reviewed",
    _format_percent(
        float((reviewed_unmapped_filtered["auto_verdict"] == "auto_verified_extra").mean())
    ) if not reviewed_unmapped_filtered.empty else "-",
)
meta_col9.metric(
    "Likely Hallucination / Decided",
    _format_percent(
        float((decided_unmapped_filtered["auto_verdict"] == "auto_likely_hallucination").mean())
    ) if not decided_unmapped_filtered.empty else "-",
)
st.caption("Net discovery / image = recovered non-task slots + auto-verified extra discoveries - likely hallucinations, measured per target image. Conservative net discovery doubles the hallucination penalty.")

with st.expander("Run Metadata", expanded=False):
    verification_model = _text(verification_meta.get("clip_model_name"))
    st.write(
        {
            "generated_at_utc": bundle.eval_meta.get("generated_at_utc"),
            "mode": bundle.eval_meta.get("mode"),
            "tau_excess_kl": bundle.eval_meta.get("tau_excess_kl"),
            "tau_ratio": bundle.eval_meta.get("tau_ratio"),
            "output_dir": str(bundle.output_dir),
            "config_path": str(bundle.config_path) if bundle.config_path else "",
            "verification_csv_path": verification_csv_path if _text(verification_csv_path) else "",
            "verification_model": verification_model or "Not run",
            "verification_valid_threshold": verification_meta.get("valid_threshold"),
            "verification_hallucination_threshold": verification_meta.get("hallucination_threshold"),
        }
    )

overview_tab, breakdown_tab, scene_tab, inspector_tab, data_tab = st.tabs(
    ["Overview", "Breakdowns", "Scene Families", "Sample Inspector", "Data"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        _chart_or_info(build_overall_discovery_chart(summary_view_df, bundle.mode), fallback_message="No summary data available.")
    with right:
        _chart_or_info(build_tradeoff_chart(summary_view_df, bundle.mode), fallback_message="No summary data available.")

    left, right = st.columns(2)
    with left:
        _chart_or_info(build_svr_chart(summary_view_df, bundle.mode), fallback_message="No summary data available.")
    with right:
        _chart_or_info(build_field_breakdown_chart(summary_view_df, bundle.mode))

    _chart_or_info(build_validity_stack_chart(summary_view_df, bundle.mode))
    _chart_or_info(
        build_auto_verification_chart(summary_view_df, bundle.mode),
        fallback_message="No automatic unmapped-verification results are available for the current selection.",
    )

    st.subheader("Summary Table")
    st.dataframe(_summary_table(summary_view_df, bundle.mode), width="stretch", hide_index=True)

with breakdown_tab:
    if flagged_filtered.empty:
        st.info("No flagged attributes match the current row filters.")
    else:
        left, right = st.columns(2)
        with left:
            _chart_or_info(build_top_attributes_chart(flagged_filtered, top_n))
        with right:
            _chart_or_info(build_scene_chart(image_filtered, bundle.mode))

        evidence_chart, ratio_clip = build_evidence_chart(flagged_filtered, bundle.mode)
        _chart_or_info(evidence_chart, fallback_message="No positive finite ratios are available for the evidence plot.")
        if ratio_clip is not None:
            st.caption(f"Evidence plot clips the top 1% of ratio values at {_format_float(ratio_clip)} to keep the scale readable.")

        st.subheader("Attribute Table")
        st.dataframe(
            _attribute_table_for_display(
                flagged_filtered.sort_values(["display_setting", "image_id", "excess_kl"], ascending=[True, True, False])
            ),
            width="stretch",
            hide_index=True,
        )

with scene_tab:
    if target_df.empty:
        st.info("Scene-family analysis is unavailable because the target manifest could not be loaded for this run.")
    elif not selected_scenes:
        st.info("Select at least one scene family to populate the scene-family analysis.")
    else:
        st.caption(
            "Scene-family metrics use only non-task slots as the denominator. In this dataset that excludes the primary slot because `task_label=primary_object`."
        )
        scene_metric = st.selectbox(
            "Scene metric",
            list(SCENE_ANALYSIS_METRICS.keys()),
            index=0,
            format_func=lambda key: SCENE_ANALYSIS_METRICS[key]["label"],
        )
        focus_setting_options = _sort_settings(scene_setting_view_df["display_setting"].dropna().unique().tolist()) if not scene_setting_view_df.empty else []
        focus_setting = st.selectbox(
            f"Focus {_display_entity_name(bundle.mode).lower()}",
            focus_setting_options,
            index=0 if focus_setting_options else None,
        ) if focus_setting_options else ""

        if scene_setting_view_df.empty:
            st.info("No scene-family rows are available under the current filters.")
        else:
            left, right = st.columns(2)
            with left:
                _chart_or_info(
                    build_scene_metric_heatmap(scene_setting_view_df, scene_metric, bundle.mode),
                    fallback_message="No scene-family heatmap is available.",
                )
            with right:
                _chart_or_info(
                    build_scene_focus_chart(scene_setting_view_df, scene_metric, focus_setting, bundle.mode),
                    fallback_message="No focus-setting scene-family chart is available.",
                )

            st.subheader("Scene Overview")
            st.dataframe(_scene_overview_table(scene_overview_df), width="stretch", hide_index=True)

            st.subheader("Scene x Setting Table")
            st.dataframe(
                _scene_setting_table(
                    scene_setting_view_df.sort_values([scene_metric, "scene_family_display", "display_setting"], ascending=[False, True, True]),
                    bundle.mode,
                ),
                width="stretch",
                hide_index=True,
            )

with inspector_tab:
    if image_filtered.empty:
        st.info(f"No {_display_pair_name(bundle.mode)} match the current row filters.")
    else:
        selector_df = (
            image_filtered.groupby(
                [
                    "image_id",
                    "scene_family_display",
                    "primary_label",
                    "background_label",
                ],
                as_index=False,
            )
            .agg(
                matched_settings=("display_setting", "nunique"),
                flagged_pairs=("pair_key", "nunique"),
                total_flagged=("num_flagged", "sum"),
            )
            .sort_values(["total_flagged", "matched_settings", "image_id"], ascending=[False, False, True])
        )
        label_key = "models" if bundle.mode == "rq1" else "settings"
        option_map = {
            f"{row.image_id} | {row.scene_family_display} | {label_key}={int(row.matched_settings)} | attrs={int(row.total_flagged)}": row.image_id
            for row in selector_df.itertuples(index=False)
        }
        selected_image_label = st.selectbox("Image", list(option_map.keys()))
        selected_image_id = option_map[selected_image_label]

        image_setting_df = image_filtered[image_filtered["image_id"] == selected_image_id].copy()
        image_setting_df = image_setting_df.sort_values(
            ["num_flagged", "num_valid_flagged", "display_setting"], ascending=[False, False, True]
        )
        st.dataframe(
            image_setting_df[
                [
                    col
                    for col in [
                        "display_setting",
                        "K",
                        "llm_name",
                        "available_slot_count",
                        "recovered_slot_count",
                        "slot_recall_image",
                        "num_flagged",
                        "num_task_aligned",
                        "num_valid_flagged",
                        "num_invalid_flagged",
                        "task_label",
                    ]
                    if col in image_setting_df.columns
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        selected_setting = st.selectbox(
            f"{_display_entity_name(bundle.mode)} for detail view",
            image_setting_df["display_setting"].tolist(),
            index=0,
        )

        selected_row = image_df[
            (image_df["image_id"] == selected_image_id) & (image_df["display_setting"] == selected_setting)
        ].iloc[0]
        detail_df = flagged_df[
            (flagged_df["image_id"] == selected_image_id) & (flagged_df["display_setting"] == selected_setting)
        ].copy()
        detail_df = detail_df.sort_values(["semantic_supported", "excess_kl"], ascending=[False, False])

        left, right = st.columns([1.1, 1.2])
        with left:
            image_path = _resolve_image(selected_row.to_dict(), bundle)
            if image_path is not None and image_path.exists():
                st.image(str(image_path), caption=selected_image_id, width="stretch")
            else:
                st.warning("Image file could not be resolved from the current metadata.")
        with right:
            st.markdown(f"**Image ID**: `{selected_image_id}`")
            st.markdown(f"**{_display_entity_name(bundle.mode)}**: `{selected_setting}`")
            st.markdown(f"**Embedding Model**: {_text(selected_row.get('model_name')) or '-'}")
            st.markdown(f"**LLM**: {_text(selected_row.get('llm_name')) or '-'}")
            st.markdown(f"**K**: {_normalize_numeric_token(selected_row.get('K')) or '-'}")
            st.markdown(f"**Scene Family**: {selected_row.get('scene_family_display', 'Unknown')}")
            st.markdown(f"**Task Label**: {_text(selected_row.get('task_label')) or '-'}")
            st.markdown(
                f"**Excess Slots Recovered**: {int(selected_row.get('recovered_slot_count') or 0)} / "
                f"{int(selected_row.get('available_slot_count') or 0)} "
                f"({_format_percent(selected_row.get('slot_recall_image'))})"
            )
            st.markdown(
                f"**Flagged Attributes**: {int(selected_row.get('num_flagged') or 0)} "
                f"({int(selected_row.get('num_task_aligned') or 0)} task-aligned, "
                f"{int(selected_row.get('num_valid_flagged') or 0)} excess grounded, "
                f"{int(selected_row.get('num_invalid_flagged') or 0)} unmapped)"
            )
            if not detail_df.empty and "auto_verdict" in detail_df.columns:
                unmapped_detail = detail_df[detail_df["grounding_label"].map(_text) == "Unmapped candidate"].copy()
                reviewed_detail = unmapped_detail[unmapped_detail["auto_verdict"].map(_text).ne("")].copy()
                decided_detail = reviewed_detail[
                    reviewed_detail["auto_verdict"].isin(["auto_verified_extra", "auto_likely_hallucination"])
                ].copy()
                st.markdown(
                    f"**Auto Verification**: {int((reviewed_detail['auto_verdict'] == 'auto_verified_extra').sum()) if not reviewed_detail.empty else 0} verified, "
                    f"{int((reviewed_detail['auto_verdict'] == 'auto_likely_hallucination').sum()) if not reviewed_detail.empty else 0} likely hallucination, "
                    f"{int((reviewed_detail['auto_verdict'] == 'auto_uncertain').sum()) if not reviewed_detail.empty else 0} uncertain"
                )
                if not decided_detail.empty:
                    st.markdown(
                        f"**Auto-Verified / Decided**: {_format_percent(float((decided_detail['auto_verdict'] == 'auto_verified_extra').mean()))}"
                    )
            semantic_parts = semantic_parts_from_record(selected_row.to_dict())
            if semantic_parts:
                st.markdown("**Semantic Labels**")
                for label, value in semantic_parts:
                    st.write(f"{label}: {value}")
            semantic_text = _text(selected_row.get("semantic_text"))
            if semantic_text:
                st.markdown("**Semantic Text**")
                st.write(semantic_text)

        st.subheader(f"All Flagged Attributes for This Image/{_display_entity_name(bundle.mode)}")
        st.caption(f"The detail table shows all flagged attributes for the selected image/{_display_entity_name(bundle.mode).lower()} pair, even if some are outside the current row filters.")
        st.dataframe(_attribute_table_for_display(detail_df), width="stretch", hide_index=True)

        attribute_options = detail_df["attribute_name"].tolist()
        if attribute_options:
            selected_attribute = st.selectbox("Attribute detail", attribute_options)
            attr_row = detail_df[detail_df["attribute_name"] == selected_attribute].iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Description**: {_text(attr_row.get('attribute_description')) or '-'}")
                st.markdown(f"**Matched Slot**: {_text(attr_row.get('matched_slot') or attr_row.get('matched_field')) or '-'}")
                st.markdown(f"**Grounding**: {_text(attr_row.get('support_label'))}")
                st.markdown(f"**Auto Verification**: {_text(attr_row.get('auto_verdict_display')) or '-'}")
                st.markdown(f"**Confirmation**: {_text(attr_row.get('confirmation_status'))}")
            with col2:
                st.markdown(f"**Support Score**: {_format_float(attr_row.get('support_score'))}")
                st.markdown(f"**Excess KL**: {_format_float(attr_row.get('excess_kl'), 4)}")
                st.markdown(f"**Task KL**: {_format_float(attr_row.get('task_kl'), 4)}")
                st.markdown(f"**Auto Valid Probability**: {_format_float(attr_row.get('auto_valid_probability'), 4)}")
                ratio_value = attr_row.get("excess_to_task_kl_ratio")
                st.markdown(
                    f"**Excess / Task KL Ratio**: {'inf' if bool(attr_row.get('ratio_is_infinite')) else _format_float(ratio_value, 4)}"
                )
            if _text(attr_row.get("positive_patterns")):
                st.markdown(f"**Positive Patterns**: {_text(attr_row.get('positive_patterns'))}")
            if _text(attr_row.get("matched_terms")):
                st.markdown(f"**Matched Terms**: {_text(attr_row.get('matched_terms'))}")
            if _text(attr_row.get("auto_best_prompt")):
                st.markdown(f"**Verifier Prompt**: {_text(attr_row.get('auto_best_prompt'))}")

with data_tab:
    st.subheader("Downloads")
    download_col1, download_col2, download_col3 = st.columns(3)
    with download_col1:
        st.download_button(
            "Download summary CSV",
            data=_download_csv(summary_filtered),
            file_name=f"{bundle.mode}_summary_filtered.csv",
            mime="text/csv",
        )
    with download_col2:
        st.download_button(
            "Download filtered attributes CSV",
            data=_download_csv(flagged_filtered),
            file_name=f"{bundle.mode}_flagged_attributes_filtered.csv",
            mime="text/csv",
        )
    with download_col3:
        st.download_button(
            f"Download filtered {_display_pair_name(bundle.mode)} CSV",
            data=_download_csv(image_filtered),
            file_name=f"{bundle.mode}_flagged_images_filtered.csv",
            mime="text/csv",
        )

    st.subheader("Raw Summary")
    st.dataframe(summary_filtered, width="stretch", hide_index=True)
    st.subheader("Raw Filtered Attributes")
    st.dataframe(flagged_filtered, width="stretch", hide_index=True)
    st.subheader("Raw Filtered Image Pairs")
    st.dataframe(image_filtered, width="stretch", hide_index=True)
