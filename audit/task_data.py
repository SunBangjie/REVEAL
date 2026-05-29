from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from audit.config import resolve_optional
from audit.utils import normalize_rows

PRED_LABEL_ALIASES = (
    "pred_label",
    "predicted_label",
    "pred",
    "pred_idx",
    "pred_id",
    "class_id",
    "class_idx",
    "class_index",
    "label_id",
    "label_idx",
    "label_index",
    "y",
    "Y",
    "label",
)


def _npz_has_key(npz: Any, key: str) -> bool:
    files = getattr(npz, "files", None)
    if files is not None:
        return key in files
    return key in npz


def _empty_nan_series(length: int) -> pd.Series:
    return pd.Series(np.nan, index=range(length), dtype=np.float32)


def _class_name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("name", "label", "class_name"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
    return str(value)


def find_pred_label_column(columns: Iterable[Any]) -> str | None:
    available = {str(col): str(col) for col in columns}
    for alias in PRED_LABEL_ALIASES:
        if alias in available:
            return available[alias]
    return None


def load_class_name_map(class_name_json: str | Path | None, *, warn_missing: bool = False) -> dict[str, str]:
    if class_name_json is None:
        return {}
    path = Path(class_name_json)
    if not path.exists():
        if warn_missing:
            print(f"[WARN] class_name_json not found at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(i): _class_name(value) for i, value in enumerate(data)}
    if isinstance(data, dict):
        return {str(key): _class_name(value) for key, value in data.items()}
    return {}


def extract_logits_from_csv(df: pd.DataFrame) -> tuple[np.ndarray | None, list[str]]:
    logit_cols = [col for col in df.columns if str(col).startswith("logit_")]
    if logit_cols:
        return df[logit_cols].to_numpy(dtype=np.float32), logit_cols
    if "logits" not in df.columns:
        return None, []

    rows: list[list[float]] = []
    max_len = 0
    for cell in df["logits"].fillna("[]"):
        try:
            parsed = json.loads(cell) if isinstance(cell, str) else list(cell)
        except Exception:
            parsed = []
        vals = [float(value) for value in parsed]
        rows.append(vals)
        max_len = max(max_len, len(vals))

    if max_len == 0:
        return None, []

    arr = np.zeros((len(rows), max_len), dtype=np.float32)
    for idx, vals in enumerate(rows):
        arr[idx, : len(vals)] = vals
    return arr, [f"logit_{idx}" for idx in range(max_len)]


def load_task_outputs(
    cfg: dict[str, Any],
    project_root: Path,
    emb_npz: Any,
    ids: np.ndarray,
    *,
    error_on_missing_image_id: bool = False,
) -> pd.DataFrame:
    out = pd.DataFrame({"image_id": np.asarray(ids).astype(str)})

    if _npz_has_key(emb_npz, "pred_labels"):
        out["pred_label"] = np.asarray(emb_npz["pred_labels"]).astype(np.int64)
    if _npz_has_key(emb_npz, "logits"):
        logits = np.asarray(emb_npz["logits"])
        if logits.ndim == 2 and logits.shape[0] == len(out):
            for idx in range(logits.shape[1]):
                out[f"logit_{idx}"] = logits[:, idx].astype(np.float32)
            if "pred_label" not in out.columns:
                out["pred_label"] = np.argmax(logits, axis=1).astype(np.int64)

    task_csv = resolve_optional((cfg.get("paths") or {}).get("task_outputs_csv"), project_root)
    if task_csv is None or not task_csv.exists():
        return out

    ext = pd.read_csv(task_csv)
    if "image_id" not in ext.columns:
        if error_on_missing_image_id:
            raise KeyError(f"task outputs csv must contain image_id column: {task_csv}")
        return out

    ext = ext.copy()
    ext["image_id"] = ext["image_id"].astype(str)
    pred_label_col = find_pred_label_column(ext.columns)
    if pred_label_col and pred_label_col != "pred_label":
        ext["pred_label"] = ext[pred_label_col]

    logits_arr, logit_cols = extract_logits_from_csv(ext)
    keep_cols = ["image_id"]
    if "pred_label" in ext.columns:
        ext["pred_label"] = pd.to_numeric(ext["pred_label"], errors="coerce")
        keep_cols.append("pred_label")
    ext_keep = ext[keep_cols].copy()
    if logits_arr is not None:
        for idx, col in enumerate(logit_cols):
            ext_keep[col] = logits_arr[:, idx]

    merged = out.merge(ext_keep, on="image_id", how="left", suffixes=("_npz", "_csv"))
    if "pred_label_csv" in merged.columns or "pred_label_npz" in merged.columns:
        pred_csv = merged.get("pred_label_csv", _empty_nan_series(len(merged)))
        pred_npz = merged.get("pred_label_npz", _empty_nan_series(len(merged)))
        merged["pred_label"] = pred_csv.combine_first(pred_npz)

    csv_logit_cols = [col for col in merged.columns if col.endswith("_csv") and col.startswith("logit_")]
    npz_logit_cols = [col for col in merged.columns if col.endswith("_npz") and col.startswith("logit_")]
    base_names = sorted({col[:-4] for col in csv_logit_cols + npz_logit_cols})
    out_cols = ["image_id"]
    if "pred_label" in merged.columns:
        out_cols.append("pred_label")
    for base in base_names:
        csv_col = f"{base}_csv"
        npz_col = f"{base}_npz"
        col_csv = merged.get(csv_col, _empty_nan_series(len(merged)))
        col_npz = merged.get(npz_col, _empty_nan_series(len(merged)))
        merged[base] = col_csv.combine_first(col_npz)
        out_cols.append(base)
    return merged[out_cols].copy()


def task_output_matrix(task_outputs: pd.DataFrame) -> tuple[np.ndarray | None, list[str]]:
    logit_cols = [col for col in task_outputs.columns if str(col).startswith("logit_")]
    if not logit_cols:
        return None, []
    matrix = task_outputs[logit_cols].to_numpy(dtype=np.float32)
    if np.isnan(matrix).all():
        return None, []
    return normalize_rows(np.nan_to_num(matrix, nan=0.0)), logit_cols


def safe_pred_label(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None
