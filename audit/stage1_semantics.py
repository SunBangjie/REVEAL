from __future__ import annotations

import json
import re
from typing import Any, Iterable, Sequence


DEFAULT_GENERIC_TERMS = {
    "image",
    "images",
    "photo",
    "photos",
    "picture",
    "pictures",
    "photograph",
    "photographs",
    "depiction",
    "depictions",
    "visual",
    "content",
    "thing",
    "things",
    "stuff",
    "object",
    "objects",
    "item",
    "items",
    "entity",
    "entities",
}

MEDIA_TAUTOLOGY_PATTERNS = (
    r"\bimage\s+(?:of|about|showing)\b",
    r"\bphoto(?:graph)?\s+(?:of|about|showing)\b",
    r"\bpicture\s+(?:of|about|showing)\b",
    r"\bvisual\s+depiction\s+of\b",
)

MEDIA_GENERIC_TERMS = {
    "image",
    "images",
    "photo",
    "photos",
    "picture",
    "pictures",
    "photograph",
    "photographs",
    "depiction",
    "depictions",
}


def _find_json_key_value_start(text: str, key: str) -> int | None:
    match = re.search(rf'"{re.escape(str(key))}"\s*:\s*', text)
    if not match:
        return None
    return match.end()


def _extract_json_string_field(text: str, key: str) -> str | None:
    start = _find_json_key_value_start(text, key)
    if start is None:
        return None
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != '"':
        return None

    in_str = True
    esc = False
    for idx in range(start + 1, len(text)):
        ch = text[idx]
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            try:
                value = json.loads(text[start : idx + 1])
            except Exception:
                return None
            return value if isinstance(value, str) else None
    return None


def _extract_json_object_list_field(text: str, key: str) -> list[dict[str, Any]] | None:
    start = _find_json_key_value_start(text, key)
    if start is None:
        return None
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "[":
        return None

    items: list[dict[str, Any]] = []
    depth = 0
    in_str = False
    esc = False
    obj_start: int | None = None

    for idx in range(start + 1, len(text)):
        ch = text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    value = json.loads(text[obj_start : idx + 1])
                except Exception:
                    value = None
                if isinstance(value, dict):
                    items.append(value)
                obj_start = None
        elif ch == "]" and depth == 0:
            return items

    return items


def _extract_json_string_list_field(text: str, key: str) -> list[str] | None:
    start = _find_json_key_value_start(text, key)
    if start is None:
        return None
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "[":
        return None

    decoder = json.JSONDecoder()
    items: list[str] = []
    idx = start + 1

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        if text[idx] == "]":
            return items
        if text[idx] == ",":
            idx += 1
            continue

        try:
            value, idx = decoder.raw_decode(text, idx)
        except Exception:
            break
        if isinstance(value, str):
            items.append(value)

    return items


def _recover_stage1_json_object(text: str) -> dict[str, Any] | None:
    recovered: dict[str, Any] = {}

    summary = _extract_json_string_field(text, "summary")
    if summary is not None:
        recovered["summary"] = summary

    shared_attributes = _extract_json_object_list_field(text, "shared_attributes")
    if shared_attributes:
        recovered["shared_attributes"] = shared_attributes

    predicted_label_covers = _extract_json_string_field(text, "predicted_label_covers")
    if predicted_label_covers is not None:
        recovered["predicted_label_covers"] = predicted_label_covers

    excess_secrets = _extract_json_object_list_field(text, "excess_secrets")
    if excess_secrets:
        recovered["excess_secrets"] = excess_secrets

    rejected_generic_terms = _extract_json_string_list_field(text, "rejected_generic_terms")
    if rejected_generic_terms:
        recovered["rejected_generic_terms"] = rejected_generic_terms

    if not recovered:
        return None
    recovered["_partial_parse_recovered"] = True
    return recovered


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response")
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")

    depth = 0
    in_str = False
    esc = False
    for idx, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : idx + 1])
                if isinstance(parsed, dict):
                    return parsed
    recovered = _recover_stage1_json_object(text)
    if recovered is not None:
        return recovered
    raise ValueError("Could not parse JSON object")


def build_stage1_system_prompt(generic_terms: Sequence[str] | None = None) -> str:
    terms = list(generic_terms) if generic_terms is not None else sorted(DEFAULT_GENERIC_TERMS)
    bad_terms = ", ".join(terms[:60])
    return (
        "You analyze retrieved image captions to discover ALL semantic meanings "
        "that a machine-learning representation may encode. "
        "Work in two passes. Pass 1: ignore the task label and recover the full "
        "set of caption-supported semantic attributes. Pass 2: if a predicted "
        "task label is provided, compare each discovered attribute against that "
        "label and judge how much the label already explains it. "
        "The predicted label is for comparison only. Do not let it suppress, "
        "hide, or narrow the discovered attribute set. "
        "Include both broad and specific semantics when they are genuinely "
        "supported by the captions: objects, fine-grained categories, scene "
        "type, activities, relations, materials, colors, weather, lighting, "
        "time of day, season, indoor/outdoor setting, demographics, mood, "
        "location cues, and other contextual details are all allowed. "
        "Prefer high recall among caption-supported semantics. Missing a real "
        "attribute is worse than including a broad but genuine one. "
        "Do not describe artistic style or media-format tautologies. "
        f"Reject only vacuous placeholder terms such as: {bad_terms}. "
        "For each discovered attribute, provide task_relevance so downstream "
        "code can decide whether it is beyond the task label. "
        "Prefer attributes that could later be operationalized into "
        "text-labeling rules using short keyword or phrase patterns. "
        "Return JSON only."
    )


def stage1_response_schema(*, include_excess_secrets: bool) -> dict[str, Any]:
    shared_spec = {
        "name": "str",
        "description": "str",
        "evidence": ["str"],
        "specificity": "float",
        "relevance": "float",
        "privacy_relevance": "float",
        "task_relevance": "float",
        "positive_patterns": ["str"],
        "negative_patterns": ["str"],
    }
    schema: dict[str, Any] = {
        "summary": "str",
        "shared_attributes": [shared_spec],
        "predicted_label_covers": "str",
        "rejected_generic_terms": ["str"],
    }
    if include_excess_secrets:
        schema["excess_secrets"] = [{
            "name": "str",
            "description": "str",
            "why_excess": "str",
            "positive_patterns": ["str"],
            "negative_patterns": ["str"],
            "privacy_relevance": "float",
            "specificity": "float",
            "task_relevance": "float",
        }]
        schema["excess_task_relevance_threshold"] = "float"
    return schema


def _schema_json(include_excess_secrets: bool) -> str:
    schema = stage1_response_schema(include_excess_secrets=include_excess_secrets)
    return json.dumps(schema, ensure_ascii=False)


def build_stage1_user_prompt(
    target_image_id: str,
    predicted_label_desc: str | None,
    captions: Sequence[str],
    heuristic_phrases: Sequence[str],
    retrieved_top_k: int,
    *,
    max_attributes: int = 10,
    max_excess_secrets: int | None = None,
    include_excess_secrets: bool = True,
) -> str:
    lines = [f"Target image id: {target_image_id}"]
    predicted_text = predicted_label_desc or "None"
    lines.append(f"Model predicted label for this image: {predicted_text}")
    lines.append(
        "Important: do not anchor on this label while discovering semantics. "
        "First discover attributes from the captions alone. Only after you have "
        "the full list should you score how much each attribute is already "
        "explained by the predicted label."
    )
    lines.append(f"Retrieved top-{retrieved_top_k} neighbor captions:")
    for idx, caption in enumerate(captions, start=1):
        lines.append(f"[{idx}] {caption}")
    if heuristic_phrases:
        lines.append("")
        lines.append("Heuristic top phrases (weak hints only):")
        lines.append(", ".join(str(v) for v in heuristic_phrases if str(v).strip()))

    lines.extend([
        "",
        "Return a JSON object with this schema:",
        _schema_json(include_excess_secrets),
        "Rules:",
        f"- Keep at most {int(max_attributes)} shared_attributes.",
        "- name should be concise snake_case if possible.",
        "- Include broad concepts too when they are meaningful: color, weather, time of day, season, indoor/outdoor, material, crowd level, scene type, activity, mood, or environmental conditions are allowed.",
        "- evidence should be short phrases copied or lightly normalized from the captions.",
        "- specificity: how specific vs generic the concept is (0-1).",
        "- relevance: how substantively informative the concept is about image content (0-1).",
        "- privacy_relevance: how revealing or sensitive the concept could be (0-1).",
        "- task_relevance: how much the predicted label already explains the concept (0-1). High means the concept is close to or heavily implied by the task label. Low means it adds meaning beyond the task label.",
        "- positive_patterns should be short keyword phrases (1-4 words) that could later be used for caption matching.",
        "- negative_patterns are optional short exclusion phrases to reduce false positives.",
        "- predicted_label_covers should briefly summarize what the predicted label already explains.",
        "- shared_attributes must be the full discovered set, not only concepts beyond the task label.",
        "- If a concept is genuine but largely explained by the task label, still include it and give it a higher task_relevance instead of deleting it.",
        "- Do not reject a concept merely because it is generic, visually common, or correlated with the task label.",
        "- Reject only media-format tautologies or vacuous placeholders.",
    ])
    if include_excess_secrets:
        limit = int(max_excess_secrets if max_excess_secrets is not None else max_attributes)
        lines.extend([
            f"- Keep at most {limit} excess_secrets.",
            "- excess_secrets should be the subset of shared_attributes whose task_relevance is low enough that they go beyond the predicted label, not mere restatements of it.",
            "- If there is no meaningful excess secret beyond the predicted label, return an empty excess_secrets list.",
        ])
    lines.append("- Return JSON only. No prose outside JSON.")
    return "\n".join(lines)


def build_stage1_chat_prompt(
    target_image_id: str,
    predicted_label_desc: str | None,
    captions: Sequence[str],
    heuristic_phrases: Sequence[str],
    retrieved_top_k: int,
    *,
    generic_terms: Sequence[str] | None = None,
    max_attributes: int = 10,
    max_excess_secrets: int | None = None,
    include_excess_secrets: bool = True,
) -> str:
    system_prompt = build_stage1_system_prompt(generic_terms)
    user_prompt = build_stage1_user_prompt(
        target_image_id,
        predicted_label_desc,
        captions,
        heuristic_phrases,
        retrieved_top_k,
        max_attributes=max_attributes,
        max_excess_secrets=max_excess_secrets,
        include_excess_secrets=include_excess_secrets,
    )
    return (
        "Please follow the instruction block and then answer the request. "
        "Return JSON only, with no markdown fence and no extra commentary.\n\n"
        "[SYSTEM INSTRUCTION]\n"
        f"{system_prompt}\n\n"
        "[USER REQUEST]\n"
        f"{user_prompt}\n"
    )


def snakeify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80]


def narrow_generic_blacklist(values: Iterable[Any] | None) -> set[str]:
    narrowed = set(MEDIA_GENERIC_TERMS)
    for value in values or []:
        text = str(value or "").strip().lower()
        if text in MEDIA_GENERIC_TERMS:
            narrowed.add(text)
    return narrowed


def looks_like_media_tautology(name: str, description: str) -> bool:
    hay = f"{name} {description}".strip().lower()
    if not hay:
        return False
    if hay in MEDIA_GENERIC_TERMS:
        return True
    return any(re.search(pattern, hay) for pattern in MEDIA_TAUTOLOGY_PATTERNS)


def dedupe_secret_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.get("name") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if number != number:
            return default
        return number
    except Exception:
        return default


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _clean_pattern_list(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = _clean_text(str(value))
        if text and text not in out:
            out.append(text)
    return out


def derive_excess_from_shared(
    shared_items: list[dict[str, Any]],
    *,
    max_task_relevance: float,
    min_privacy_relevance: float,
    max_items: int = 10,
) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    for item in shared_items:
        task_rel = _safe_float(item.get("task_relevance"), 1.0)
        if task_rel > max_task_relevance:
            continue
        privacy_rel = _safe_float(item.get("privacy_relevance", item.get("relevance")), _safe_float(item.get("relevance"), 0.0))
        if privacy_rel < min_privacy_relevance:
            continue
        derived.append({
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "why_excess": f"task_relevance={task_rel:.2f} <= threshold {max_task_relevance:.2f}",
            "positive_patterns": list(item.get("positive_patterns") or []),
            "negative_patterns": list(item.get("negative_patterns") or []),
            "privacy_relevance": privacy_rel,
            "specificity": _safe_float(item.get("specificity"), 0.0),
            "task_relevance": task_rel,
            "relevance": _safe_float(item.get("relevance"), 0.0),
        })
    return dedupe_secret_items(derived)[:max_items]


def sanitize_stage1_result(
    raw: dict[str, Any],
    *,
    min_specificity: float,
    min_relevance: float,
    min_privacy_relevance: float,
    generic_blacklist: set[str],
    max_task_relevance: float,
    max_shared_attributes: int = 10,
    max_excess_secrets: int = 10,
    merge_explicit_excess: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "summary": str(raw.get("summary") or "").strip(),
        "predicted_label_covers": str(raw.get("predicted_label_covers") or "").strip(),
        "rejected_generic_terms": [str(x).strip() for x in (raw.get("rejected_generic_terms") or []) if str(x).strip()],
        "shared_attributes": [],
        "excess_secrets": [],
        "excess_task_relevance_threshold": float(max_task_relevance),
    }

    consolidated_shared: list[dict[str, Any]] = []
    for item in (raw.get("shared_attributes") or [])[:max_shared_attributes]:
        name = snakeify(str(item.get("name") or ""))
        desc = str(item.get("description") or "").strip()
        spec = _safe_float(item.get("specificity"), 0.0)
        rel = _safe_float(item.get("relevance"), 0.0)
        priv = _safe_float(item.get("privacy_relevance", item.get("relevance")), rel)
        task_rel = _safe_float(item.get("task_relevance"), 1.0)
        evidence = [str(x).strip() for x in (item.get("evidence") or []) if str(x).strip()][:10]
        pos = _clean_pattern_list(item.get("positive_patterns") or [])[:10]
        neg = _clean_pattern_list(item.get("negative_patterns") or [])[:10]
        if not name or name in generic_blacklist or looks_like_media_tautology(name, desc):
            continue
        if spec < min_specificity or rel < min_relevance:
            continue
        if not pos:
            pos = [name.replace("_", " ")]
        consolidated_shared.append({
            "name": name,
            "description": desc,
            "evidence": evidence,
            "specificity": spec,
            "relevance": rel,
            "privacy_relevance": priv,
            "task_relevance": task_rel,
            "positive_patterns": pos,
            "negative_patterns": neg,
        })

    if merge_explicit_excess:
        for item in (raw.get("excess_secrets") or [])[:max_excess_secrets]:
            name = snakeify(str(item.get("name") or ""))
            desc = str(item.get("description") or "").strip()
            spec = _safe_float(item.get("specificity"), 0.0)
            rel = _safe_float(item.get("relevance", item.get("privacy_relevance")), 0.5)
            priv = _safe_float(item.get("privacy_relevance", item.get("relevance")), rel)
            task_rel = _safe_float(item.get("task_relevance"), 0.25)
            pos = _clean_pattern_list(item.get("positive_patterns") or [])[:10]
            neg = _clean_pattern_list(item.get("negative_patterns") or [])[:10]
            if not name or name in generic_blacklist or looks_like_media_tautology(name, desc):
                continue
            if spec < min_specificity or rel < min_relevance:
                continue
            if not pos:
                pos = [name.replace("_", " ")]
            consolidated_shared.append({
                "name": name,
                "description": desc,
                "evidence": [],
                "specificity": spec,
                "relevance": rel,
                "privacy_relevance": priv,
                "task_relevance": task_rel,
                "positive_patterns": pos,
                "negative_patterns": neg,
                "why_excess": str(item.get("why_excess") or "").strip(),
            })

    out["shared_attributes"] = dedupe_secret_items(consolidated_shared)[:max_shared_attributes]
    out["excess_secrets"] = derive_excess_from_shared(
        out["shared_attributes"],
        max_task_relevance=float(max_task_relevance),
        min_privacy_relevance=float(min_privacy_relevance),
        max_items=max_excess_secrets,
    )
    return out
