from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runner import EvaluationRunner
from evaluation.semantic_support import normalize_text


@dataclass(frozen=True)
class MetricSpec:
    column: str
    title: str
    is_percent: bool
    color: str


MODEL_ORDER = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_base_patch16_224",
]

MODEL_LABELS = {
    "mobilenet_v3_small": "MNV3-S",
    "mobilenet_v3_large": "MNV3-L",
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EffNet-B0",
    "convnext_tiny": "ConvNeXt-T",
    "vit_base_patch16_224": "ViT-B/16",
}

LLM_LABELS = {
    "LLM=GPT-4o-mini": "GPT-4o-mini",
    "LLM=GPT-4.1": "GPT-4.1",
    "LLM=GPT-5.4-mini": "GPT-5.4-mini",
    "LLM=o3": "o3",
    "LLM=GPT-5.4": "GPT-5.4",
}

METRICS = [
    MetricSpec("AnyRecoveryRate", "Success Rate", True, "#72B7B2"),
    MetricSpec("AutoVerifiedRateAll", "Semantic Validity Rate", True, "#F58518"),
    MetricSpec("NetSemanticDiscoveryYield", "Net Discovery / Image", False, "#2F855A"),
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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    root = _project_root()
    output_dir = root / "output"
    parser = argparse.ArgumentParser(description="Plot REVEAL evaluation figures as PDF.")
    parser.add_argument("--rq1-csv", type=Path, default=output_dir / "eval_rq1.csv")
    parser.add_argument("--rq1-json", type=Path, default=output_dir / "eval_rq1.json")
    parser.add_argument("--rq1-flagged-csv", type=Path, default=output_dir / "flagged_attributes_rq1.csv")
    parser.add_argument("--rq1-flagged-images-jsonl", type=Path, default=output_dir / "flagged_images_rq1.jsonl")
    parser.add_argument(
        "--rq1-verification-csv",
        type=Path,
        default=output_dir / "flagged_attributes_rq1.unmapped_verification.csv",
    )
    parser.add_argument("--rq2-csv", type=Path, default=output_dir / "eval_rq2.csv")
    parser.add_argument("--rq2-json", type=Path, default=output_dir / "eval_rq2.json")
    parser.add_argument("--rq2-flagged-csv", type=Path, default=output_dir / "flagged_attributes_rq2.csv")
    parser.add_argument("--rq2-flagged-images-jsonl", type=Path, default=output_dir / "flagged_images_rq2.jsonl")
    parser.add_argument(
        "--rq2-verification-csv",
        type=Path,
        default=output_dir / "flagged_attributes_rq2.unmapped_verification.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir / "figures")
    return parser.parse_args()


def _configure_matplotlib() -> None:
    lm_font_dir = Path("/usr/share/texmf/fonts/opentype/public/lm")
    if lm_font_dir.exists():
        for font_path in lm_font_dir.glob("*.otf"):
            try:
                font_manager.fontManager.addfont(str(font_path))
            except Exception:
                pass
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "Latin Modern Roman",
            "font.serif": ["Latin Modern Roman"],
            "mathtext.fontset": "cm",
            "font.size": 30,
            "axes.labelsize": 34,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _format_scene_family(record: Mapping[str, Any]) -> str:
    label = _text(record.get("scene_family_label"))
    return label or _text(record.get("scene_family")) or "Unknown"


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


def _task_field_from_record(record: Mapping[str, Any]) -> str:
    return TASK_LABEL_TO_FIELD.get(_text(record.get("task_label")), "")


def _available_slot_fields_from_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    available: list[str] = []
    for field_name, columns in RECOVERY_SLOT_SPECS.items():
        if any(_text(record.get(column)) for column in columns):
            available.append(field_name)
    return tuple(available)


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
    flagged.loc[verdict_mask, "auto_verdict_display"] = flagged.loc[verdict_mask, "auto_verdict"].map(
        lambda value: AUTO_VERDICT_DISPLAY.get(value, value)
    )
    return flagged


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

    existing_display = labeled["display_setting"].map(_text) if "display_setting" in labeled.columns else pd.Series("", index=labeled.index)
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
    if "pair_key" in labeled.columns:
        labeled["display_pair_key"] = _text_series(labeled["pair_key"]) + "||" + _text_series(labeled["display_setting"])
    return labeled


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
    summary["OpenWorldCandidateYield"] = (summary["total_unmapped_predictions"] / image_denom).fillna(0.0)
    summary["SlotRecall"] = (summary["total_recovered_slots"] / slot_denom).fillna(0.0)
    summary["AnyRecoveryRate"] = (summary["any_slot_recovered_images"] / image_denom).fillna(0.0)
    summary["FullRecoveryRate"] = (summary["full_slot_recovered_images"] / image_denom).fillna(0.0)
    summary["AvgRecoveredSlotsPerImage"] = (summary["total_recovered_slots"] / image_denom).fillna(0.0)
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

    summary["AutoVerificationCoverage"] = summary["auto_reviewed_unmapped_candidates"] / total_unmapped_denom
    summary["AutoVerifiedRateAll"] = (summary["auto_verified_candidates"] / total_unmapped_denom).where(reviewed_denom.notna())
    summary["AutoHallucinationRateAll"] = (summary["auto_likely_hallucination_candidates"] / total_unmapped_denom).where(reviewed_denom.notna())
    summary["AutoUncertainRateAll"] = (summary["auto_uncertain_candidates"] / total_unmapped_denom).where(reviewed_denom.notna())
    summary["AutoVerifiedRateDecided"] = summary["auto_verified_candidates"] / decided_denom
    return _apply_net_discovery_metrics(
        summary,
        recovered_col="total_recovered_slots",
        verified_col="auto_verified_candidates",
        hallucination_col="auto_likely_hallucination_candidates",
        image_denom=image_denom,
    )


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


def _safe_metric_slug(metric: MetricSpec) -> str:
    return (
        metric.column.replace("AnyRecoveryRate", "success_rate")
        .replace("AutoVerifiedRateAll", "semantic_validity_rate")
        .replace("NetSemanticDiscoveryYield", "net_discovery")
    )


def _mode_summary(
    *,
    mode: str,
    summary_path: Path,
    eval_json_path: Path,
    flagged_csv_path: Path,
    flagged_images_path: Path,
    verification_csv_path: Path,
) -> pd.DataFrame:
    if not summary_path.exists() or not flagged_csv_path.exists():
        return pd.DataFrame()

    eval_meta = _load_json(eval_json_path)
    summary_df = _prepare_summary_df(_read_csv_or_empty(summary_path), mode)
    flagged_df = _prepare_flagged_df(_fill_display_setting(_read_csv_or_empty(flagged_csv_path), mode))
    target_df = _load_target_manifest_df(eval_meta, summary_path.parent)

    if flagged_images_path.exists():
        image_rows = _load_jsonl(flagged_images_path)
        image_df = _prepare_image_df(_fill_display_setting(pd.DataFrame(image_rows), mode), flagged_df)
    else:
        image_df = _derive_image_df_from_flagged(flagged_df)

    flagged_df = _attach_setting_labels(flagged_df, summary_df, mode)
    image_df = _attach_setting_labels(image_df, summary_df, mode)
    verification_df = _prepare_verification_df(_read_csv_or_empty(verification_csv_path), mode)
    flagged_df = _attach_verification_to_flagged(flagged_df, verification_df)
    image_df = _prepare_image_df(image_df, flagged_df)
    summary_df = _backfill_summary_attack_metrics(summary_df, flagged_df, image_df, target_df)
    summary_df = _backfill_summary_unmapped_verification_metrics(summary_df, flagged_df)
    return summary_df


def _ordered_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    frame = df[df["display_setting"].isin(MODEL_ORDER)].copy()
    frame["display_setting"] = pd.Categorical(frame["display_setting"], categories=MODEL_ORDER, ordered=True)
    frame = frame.sort_values("display_setting").reset_index(drop=True)
    frame["display_label"] = frame["display_setting"].map(MODEL_LABELS)
    return frame


def _k_value(label: str) -> int:
    text = _text(label)
    if not text.startswith("K="):
        raise ValueError(f"Invalid K label: {label}")
    return int(text.split("=", 1)[1].strip())


def _rq2_k_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    frame = df[df["display_setting"].map(_text).str.fullmatch(r"K=\d+")].copy()
    if frame.empty:
        return frame
    frame["k_numeric"] = frame["display_setting"].map(_k_value)
    frame = frame.sort_values("k_numeric").reset_index(drop=True)
    frame["display_label"] = frame["k_numeric"].map(lambda value: f"k={value}")
    return frame


def _rq2_llm_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    frame = df[df["display_setting"].map(_text).str.startswith("LLM=")].copy()
    if frame.empty:
        return frame
    frame["display_label"] = frame["display_setting"].map(lambda value: LLM_LABELS.get(_text(value), _text(value).split("=", 1)[-1]))
    return frame


def _sort_frame_for_metric(df: pd.DataFrame, metric: MetricSpec) -> pd.DataFrame:
    frame = df.copy()
    frame["_metric_value"] = pd.to_numeric(frame[metric.column], errors="coerce")
    frame = frame[frame["_metric_value"].notna()].copy()
    frame = frame.sort_values(["_metric_value", "display_label"], ascending=[True, True]).reset_index(drop=True)
    return frame.drop(columns=["_metric_value"])


def _format_percent_axis(value: float, _pos: int) -> str:
    return f"{value * 100:.0f}%"


def _value_label(value: float, *, is_percent: bool) -> str:
    return f"{value * 100:.2g}%" if is_percent else f"{value:.2f}"


def _annotate_bars(ax: plt.Axes, values: list[float], *, is_percent: bool) -> None:
    ymax = ax.get_ylim()[1]
    offset = ymax * 0.028
    for idx, value in enumerate(values):
        ax.text(idx, value + offset, _value_label(value, is_percent=is_percent), ha="center", va="bottom", fontsize=26)


def _annotate_points(ax: plt.Axes, xs: list[float], values: list[float], *, is_percent: bool) -> None:
    ymax = ax.get_ylim()[1]
    offset = ymax * 0.032
    for x, value in zip(xs, values):
        ax.text(x, value + offset, _value_label(value, is_percent=is_percent), ha="center", va="bottom", fontsize=26)


def _metric_axis(ax: plt.Axes, df: pd.DataFrame, metric: MetricSpec) -> None:
    values = pd.to_numeric(df[metric.column], errors="coerce").fillna(0.0).tolist()
    ax.bar(df["display_label"], values, color=metric.color, width=0.72)
    ax.set_ylabel(metric.title)
    ax.grid(axis="y", alpha=0.25, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=20)
    if metric.is_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_percent_axis))
    ymax = max(values) if values else 0.0
    if ymax <= 0:
        ax.set_ylim(0, 1.0)
    elif metric.is_percent:
        ax.set_ylim(0, ymax * 1.24 + 0.01)
    else:
        ax.set_ylim(0, ymax * 1.24 + 0.025)
    _annotate_bars(ax, values, is_percent=metric.is_percent)


def _line_metric_axis(ax: plt.Axes, df: pd.DataFrame, metric: MetricSpec) -> None:
    plot_df = df.copy()
    xs = plot_df["k_numeric"].astype(float).tolist()
    values = pd.to_numeric(plot_df[metric.column], errors="coerce").fillna(0.0).tolist()
    ax.plot(xs, values, color=metric.color, marker="o", linewidth=6.0, markersize=14.0)
    ax.set_ylabel(metric.title)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"k={int(x)}" for x in xs])
    ax.set_xlim(3, 20)
    ax.grid(axis="y", alpha=0.25, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if metric.is_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_percent_axis))
    ymax = max(values) if values else 0.0
    if ymax <= 0:
        ax.set_ylim(0, 1.0)
    elif metric.is_percent:
        ax.set_ylim(0, ymax * 1.24 + 0.01)
    else:
        ax.set_ylim(0, ymax * 1.24 + 0.025)
    _annotate_points(ax, xs, values, is_percent=metric.is_percent)


def _plot_bar_metric(df: pd.DataFrame, metric: MetricSpec) -> plt.Figure:
    plot_df = _sort_frame_for_metric(df, metric)
    fig, ax = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)
    _metric_axis(ax, plot_df, metric)
    return fig


def _plot_k_metric(df: pd.DataFrame, metric: MetricSpec) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)
    _line_metric_axis(ax, df, metric)
    return fig


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def main() -> None:
    args = _parse_args()
    _configure_matplotlib()

    rq1_summary = _mode_summary(
        mode="rq1",
        summary_path=args.rq1_csv,
        eval_json_path=args.rq1_json,
        flagged_csv_path=args.rq1_flagged_csv,
        flagged_images_path=args.rq1_flagged_images_jsonl,
        verification_csv_path=args.rq1_verification_csv,
    )
    rq2_summary = _mode_summary(
        mode="rq2",
        summary_path=args.rq2_csv,
        eval_json_path=args.rq2_json,
        flagged_csv_path=args.rq2_flagged_csv,
        flagged_images_path=args.rq2_flagged_images_jsonl,
        verification_csv_path=args.rq2_verification_csv,
    )

    outputs: list[tuple[Path, plt.Figure]] = []
    output_dir = args.output_dir

    if not rq1_summary.empty:
        rq1_frame = _ordered_model_frame(rq1_summary)
        if not rq1_frame.empty:
            for metric in METRICS:
                pdf_path = output_dir / f"models_{_safe_metric_slug(metric)}.pdf"
                fig = _plot_bar_metric(rq1_frame, metric)
                _save_figure(fig, pdf_path)
                outputs.append((pdf_path, fig))

    if not rq2_summary.empty:
        rq2_k_frame = _rq2_k_frame(rq2_summary)
        rq2_llm_frame = _rq2_llm_frame(rq2_summary)

        if not rq2_k_frame.empty:
            for metric in METRICS:
                pdf_path = output_dir / f"k_{_safe_metric_slug(metric)}.pdf"
                fig = _plot_k_metric(rq2_k_frame, metric)
                _save_figure(fig, pdf_path)
                outputs.append((pdf_path, fig))

        if not rq2_llm_frame.empty:
            for metric in METRICS:
                pdf_path = output_dir / f"llm_{_safe_metric_slug(metric)}.pdf"
                fig = _plot_bar_metric(rq2_llm_frame, metric)
                _save_figure(fig, pdf_path)
                outputs.append((pdf_path, fig))

    if not outputs:
        raise FileNotFoundError("No plottable evaluation summaries were found in output/.")

    bundle_pdf = output_dir / "selected_metric_plots.pdf"
    with PdfPages(bundle_pdf) as pdf:
        for _, fig in outputs:
            pdf.savefig(fig, bbox_inches="tight")

    for _, fig in outputs:
        plt.close(fig)

    for path, _ in outputs:
        print(f"Wrote {path}")
    print(f"Wrote {bundle_pdf}")


if __name__ == "__main__":
    main()
