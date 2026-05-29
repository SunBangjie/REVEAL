from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve(path_str: str | Path, project_root: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (Path(project_root) / p).resolve()


def resolve_optional(path_str: str | Path | None, project_root: str | Path) -> Path | None:
    if not path_str:
        return None
    return resolve(path_str, project_root)


def _stringify_path(path_like: Any) -> str:
    if path_like is None:
        return ""
    try:
        if path_like != path_like:
            return ""
    except Exception:
        pass
    text = str(path_like).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def semantic_parts_from_record(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    field_specs = [
        ("Scene Family", "scene_family_label", "scene_family", None),
        ("Primary", "primary_label", "primary_object", "primary_attributes"),
        ("Secondary", "secondary_label", "secondary_object", "secondary_attributes"),
        ("Ternary", "ternary_label", "ternary_object", "ternary_attributes"),
        ("Background", "background_label", "background_scene", "background_attributes"),
    ]
    parts: list[tuple[str, str]] = []
    for label, preferred_key, fallback_key, attrs_key in field_specs:
        value = _stringify_path(record.get(preferred_key)) or _stringify_path(record.get(fallback_key))
        attrs = _stringify_path(record.get(attrs_key)) if attrs_key else ""
        if attrs:
            attrs_text = attrs.replace("|", ", ")
            value = f"{value} ({attrs_text})" if value else attrs_text
        if value:
            parts.append((label, value))
    return parts


def semantic_text_from_record(record: Mapping[str, Any]) -> str:
    parts = semantic_parts_from_record(record)
    return " ".join(f"{label}: {value}." for label, value in parts).strip()


def normalize_manifest_df(df: Any, cfg: Mapping[str, Any], project_root: str | Path):
    if df is None:
        return df
    if getattr(df, "empty", False):
        return df.copy()

    out = df.copy()
    report_cfg = cfg.get("report") or {}
    dataset_cfg = cfg.get("dataset") or {}
    image_root = resolve_optional(
        report_cfg.get("markdown_image_root") or report_cfg.get("image_root") or dataset_cfg.get("photos_root"),
        project_root,
    )

    output_file_series = out["output_file"].apply(_stringify_path) if "output_file" in out.columns else None

    if "image_id" in out.columns:
        image_id_series = out["image_id"].apply(_stringify_path)
    elif output_file_series is not None:
        image_id_series = output_file_series.copy()
    else:
        image_id_series = None
    if image_id_series is not None and output_file_series is not None:
        image_id_series = image_id_series.mask(image_id_series.eq(""), output_file_series)
        out["image_id"] = image_id_series

    if "image_relpath" in out.columns:
        image_relpath_series = out["image_relpath"].apply(_stringify_path)
    elif output_file_series is not None:
        image_relpath_series = output_file_series.copy()
    else:
        image_relpath_series = None
    if image_relpath_series is not None and output_file_series is not None:
        image_relpath_series = image_relpath_series.mask(image_relpath_series.eq(""), output_file_series)
        out["image_relpath"] = image_relpath_series

    if "image_path" in out.columns:
        image_path_series = out["image_path"].apply(_stringify_path)
    else:
        image_path_series = None
    if image_root is not None and output_file_series is not None:
        fallback_paths = output_file_series.apply(lambda name: str((image_root / name).resolve()) if name else "")
        if image_path_series is None:
            image_path_series = fallback_paths
        else:
            image_path_series = image_path_series.mask(image_path_series.eq(""), fallback_paths)
        out["image_path"] = image_path_series

    semantic_series = out.apply(semantic_text_from_record, axis=1)
    semantics_available = semantic_series.apply(bool).any()
    if semantics_available:
        if "semantic_text" not in out.columns:
            out["semantic_text"] = semantic_series
        else:
            semantic_text_series = out["semantic_text"].apply(_stringify_path)
            out["semantic_text"] = semantic_text_series.mask(semantic_text_series.eq(""), semantic_series)

    if "prompt" in out.columns:
        prompt_series = out["prompt"].apply(_stringify_path)
    else:
        prompt_series = None

    if semantics_available:
        if "text_base" not in out.columns:
            out["text_base"] = semantic_series
        else:
            text_base_series = out["text_base"].apply(_stringify_path)
            out["text_base"] = text_base_series.mask(text_base_series.eq(""), semantic_series)

        full_text_series = semantic_series
        if prompt_series is not None:
            full_text_series = semantic_series + " Prompt: " + prompt_series
        if "text_full" not in out.columns:
            out["text_full"] = full_text_series
        else:
            text_full_series = out["text_full"].apply(_stringify_path)
            out["text_full"] = text_full_series.mask(text_full_series.eq(""), full_text_series)
    elif prompt_series is not None:
        if "text_base" not in out.columns:
            out["text_base"] = prompt_series
        if "text_full" not in out.columns:
            out["text_full"] = prompt_series

    return out


def resolve_image_path_from_record(
    record: Mapping[str, Any],
    cfg: Mapping[str, Any],
    project_root: str | Path,
) -> Path | None:
    image_path = _stringify_path(record.get("image_path"))
    image_relpath = _stringify_path(record.get("image_relpath"))
    output_file = _stringify_path(record.get("output_file"))
    dataset_cfg = cfg.get("dataset") or {}
    report_cfg = cfg.get("report") or {}

    photos_root = resolve_optional(dataset_cfg.get("photos_root"), project_root)
    markdown_image_root = resolve_optional(
        report_cfg.get("markdown_image_root") or report_cfg.get("image_root"),
        project_root,
    )

    candidates: list[Path] = []
    if image_relpath and markdown_image_root is not None:
        candidates.append(markdown_image_root / image_relpath)
    if output_file and markdown_image_root is not None:
        candidates.append(markdown_image_root / output_file)
    if image_path:
        p = Path(image_path)
        candidates.append(p if p.is_absolute() else resolve(p, project_root))
    if image_relpath and photos_root is not None:
        candidates.append(photos_root / image_relpath)
    if output_file and photos_root is not None:
        candidates.append(photos_root / output_file)

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            continue
    return candidates[0].resolve() if candidates else None


def infer_project_root_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    parent = config_path.parent
    if parent.name == "configs":
        return parent.parent
    return parent
