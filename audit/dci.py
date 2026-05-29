from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from audit.utils import safe_text


def _resolve_image_path(image_field: str, photos_root: Path) -> Path:
    """Resolve the actual photo path using the DCI JSON `image` field.

    DCI stores the alignment inside each annotation JSON via `image`, and the
    explorer / dataset loader then open `photos_root / image_field`.
    We preserve that behavior and only fall back to basename matching if the
    exact relative path is missing locally.
    """
    p = Path(image_field)
    if p.is_absolute() and p.exists():
        return p.resolve()
    direct = (photos_root / p)
    if direct.exists():
        return direct.resolve()
    fallback = photos_root / p.name
    return fallback.resolve()


def _iter_mask_items(mask_data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(mask_data, dict):
        for v in mask_data.values():
            if isinstance(v, dict):
                yield v
    elif isinstance(mask_data, list):
        for v in mask_data:
            if isinstance(v, dict):
                yield v


def _mask_quality_score(item: dict[str, Any]) -> tuple[bool, float]:
    """Return (keep, score) for sorting / filtering masks.

    DCI uses `mask_quality` with semantics:
      0 = fine, 1 = low_quality, 2 = unusable.
    Some derived exports may instead use `quality` or `score`.
    """
    if "mask_quality" in item:
        try:
            mq = int(item.get("mask_quality", 2))
        except Exception:
            mq = 2
        if mq >= 2:
            return False, 0.0
        # fine > low_quality
        return True, 1.0 if mq == 0 else 0.5

    for key in ("quality", "score"):
        if key in item:
            try:
                q = float(item.get(key, 0.0))
            except Exception:
                q = 0.0
            return True, q

    return True, 0.0


def _format_mask_caption(item: dict[str, Any]) -> str:
    """Approximate DCI's _extract_caption semantics for manifest text.

    - unusable masks are filtered earlier
    - low-quality masks return the label only
    - fine masks return `label: caption` when both are present
    """
    label = safe_text(item.get("label", ""))
    caption = safe_text(item.get("caption", ""))

    if "mask_quality" in item:
        try:
            mq = int(item.get("mask_quality", 2))
        except Exception:
            mq = 2
        if mq >= 2:
            return ""
        if mq == 1:
            return label or caption

    if label and caption:
        return f"{label}: {caption}"
    return label or caption


def flatten_mask_captions(mask_data: Any, max_mask_captions: int = 8, min_quality: float = 0.0) -> list[str]:
    items: list[tuple[float, str]] = []
    for item in _iter_mask_items(mask_data):
        keep, score = _mask_quality_score(item)
        if not keep or score < min_quality:
            continue
        txt = _format_mask_caption(item)
        if txt:
            items.append((score, txt))
    items = sorted(items, key=lambda x: x[0], reverse=True)[:max_mask_captions]
    return [t for _, t in items]


def parse_dci_record(
    record: dict[str, Any],
    source_json: Path,
    photos_root: Path,
    max_mask_captions: int,
    min_mask_quality: float,
) -> dict[str, Any] | None:
    image_field = record.get("image") or record.get("image_path") or record.get("photo")
    if not image_field:
        return None

    image_path = _resolve_image_path(str(image_field), photos_root)
    short_caption = safe_text(record.get("short_caption", ""))
    extra_caption = safe_text(record.get("extra_caption", ""))
    summaries = safe_text(record.get("summaries", ""))
    negatives = safe_text(record.get("negatives", ""))
    mask_caps = flatten_mask_captions(
        record.get("mask_data", []),
        max_mask_captions=max_mask_captions,
        min_quality=min_mask_quality,
    )

    text_base = " ".join([t for t in [short_caption, extra_caption] if t]).strip()
    text_full = " ".join([t for t in [short_caption, extra_caption, " ".join(mask_caps), summaries] if t]).strip()

    entry_key = source_json.name
    image_id = safe_text(record.get("image_id") or entry_key)

    return {
        "image_id": image_id,
        "entry_key": entry_key,
        "source_json": str(source_json.resolve()),
        "source_dir": source_json.parent.name,
        "image_relpath": str(image_field),
        "image_path": str(image_path),
        "image_exists": int(image_path.exists()),
        "short_caption": short_caption,
        "extra_caption": extra_caption,
        "summaries": summaries,
        "negatives": negatives,
        "mask_captions": " || ".join(mask_caps),
        "text_base": text_base,
        "text_full": text_full,
    }
