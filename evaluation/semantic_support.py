from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import yaml


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "near",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

FIELD_SPECS: dict[str, tuple[str, ...]] = {
    "primary": ("primary_label", "primary_object", "primary_semantic_type", "primary_attributes"),
    "secondary": ("secondary_label", "secondary_object", "secondary_semantic_type", "secondary_attributes"),
    "ternary": ("ternary_label", "ternary_object", "ternary_semantic_type", "ternary_attributes"),
    "background": (
        "background_label",
        "background_scene",
        "background_semantic_type",
        "background_attributes",
        "scene_family",
        "scene_family_label",
    ),
}


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(value))
    except Exception:
        return False


def _safe_text(value: Any) -> str:
    if value is None or _is_nan(value):
        return ""
    return str(value).strip()


def normalize_text(text: str) -> str:
    text = _safe_text(text).lower()
    if not text:
        return ""
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str, stopwords: set[str]) -> tuple[str, ...]:
    normalized = normalize_text(text)
    if not normalized:
        return ()
    out: list[str] = []
    for token in normalized.split():
        if token in stopwords:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token:
            out.append(token)
    return tuple(out)


def _split_metadata_values(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    if "|" in text:
        return [item.strip() for item in text.split("|") if item.strip()]
    if ";" in text:
        return [item.strip() for item in text.split(";") if item.strip()]
    return [text]


def _load_raw_mapping(path: Path) -> dict[str, list[str]]:
    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle) or {}
        elif suffix == ".json":
            payload = json.load(handle) or {}
        else:
            raise ValueError(f"Unsupported synonym map format: {path}")
    if not isinstance(payload, dict):
        raise ValueError("Synonym map must be a mapping.")
    out: dict[str, list[str]] = {}
    for key, values in payload.items():
        if isinstance(values, list):
            out[str(key)] = [str(item) for item in values if _safe_text(item)]
        elif values is None:
            out[str(key)] = []
        else:
            out[str(key)] = [str(values)]
    return out


@dataclass(frozen=True)
class SupportDecision:
    supported: bool
    matched_field: str
    support_score: float
    matched_terms: tuple[str, ...]
    matched_candidate: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": bool(self.supported),
            "matched_field": str(self.matched_field),
            "support_score": float(self.support_score),
            "matched_terms": list(self.matched_terms),
            "matched_candidate": str(self.matched_candidate),
        }


@dataclass(frozen=True)
class _FieldEntry:
    field_name: str
    source_text: str
    normalized: str
    variants: frozenset[str]
    tokens: frozenset[str]


class SemanticSupportChecker:
    def __init__(
        self,
        synonym_map: Mapping[str, Sequence[str]] | None = None,
        *,
        minimum_score: float = 0.75,
        stopwords: Iterable[str] | None = None,
    ) -> None:
        self.minimum_score = float(minimum_score)
        self.stopwords = set(stopwords or DEFAULT_STOPWORDS)
        self.variant_to_canonical: dict[str, str] = {}
        self.canonical_to_variants: dict[str, set[str]] = {}
        for canonical, aliases in dict(synonym_map or {}).items():
            canonical_norm = normalize_text(str(canonical))
            if not canonical_norm:
                continue
            variants = {canonical_norm}
            variants.update(normalize_text(str(alias)) for alias in aliases or [])
            variants = {item for item in variants if item}
            self.canonical_to_variants[canonical_norm] = variants
            for item in variants:
                self.variant_to_canonical[item] = canonical_norm

    @classmethod
    def from_path(
        cls,
        path: str | Path | None,
        *,
        minimum_score: float = 0.75,
        stopwords: Iterable[str] | None = None,
    ) -> "SemanticSupportChecker":
        mapping = _load_raw_mapping(Path(path)) if path else {}
        return cls(mapping, minimum_score=minimum_score, stopwords=stopwords)

    def _expand_variants(self, text: str) -> set[str]:
        normalized = normalize_text(text)
        if not normalized:
            return set()
        variants = {normalized}
        canonical = self.variant_to_canonical.get(normalized)
        if canonical:
            variants.update(self.canonical_to_variants.get(canonical, {canonical}))
        elif normalized in self.canonical_to_variants:
            variants.update(self.canonical_to_variants[normalized])
        return variants

    def _build_field_entries(self, metadata: Mapping[str, Any]) -> list[_FieldEntry]:
        entries: list[_FieldEntry] = []
        for field_name, columns in FIELD_SPECS.items():
            for column in columns:
                for raw_value in _split_metadata_values(metadata.get(column)):
                    normalized = normalize_text(raw_value)
                    if not normalized:
                        continue
                    entries.append(
                        _FieldEntry(
                            field_name=field_name,
                            source_text=raw_value,
                            normalized=normalized,
                            variants=frozenset(self._expand_variants(normalized)),
                            tokens=frozenset(_tokenize(normalized, self.stopwords)),
                        )
                    )
        return entries

    def _score_match(self, candidate_text: str, entry: _FieldEntry) -> float:
        candidate_norm = normalize_text(candidate_text)
        if not candidate_norm:
            return 0.0
        candidate_variants = self._expand_variants(candidate_norm)
        if candidate_variants & entry.variants:
            return 1.0

        if len(candidate_norm) >= 4 and (
            candidate_norm in entry.normalized or entry.normalized in candidate_norm
        ):
            return 0.92

        candidate_tokens = frozenset(_tokenize(candidate_norm, self.stopwords))
        if not candidate_tokens or not entry.tokens:
            return 0.0
        overlap = candidate_tokens & entry.tokens
        if not overlap:
            return 0.0

        containment = len(overlap) / max(len(candidate_tokens), 1)
        jaccard = len(overlap) / len(candidate_tokens | entry.tokens)
        if containment >= 1.0:
            return 0.88
        if jaccard >= 0.67:
            return 0.80
        if jaccard >= 0.50:
            return 0.72
        return 0.0

    def check(
        self,
        attribute_text: str,
        metadata: Mapping[str, Any],
        *,
        extra_texts: Sequence[str] | None = None,
    ) -> SupportDecision:
        candidates = [attribute_text, *(extra_texts or [])]
        entries = self._build_field_entries(metadata)
        best_score = 0.0
        best_field = "none"
        best_terms: tuple[str, ...] = ()
        best_candidate = ""

        for candidate in candidates:
            candidate_text = _safe_text(candidate)
            if not candidate_text:
                continue
            for entry in entries:
                score = self._score_match(candidate_text, entry)
                if score <= best_score:
                    continue
                best_score = float(score)
                best_field = entry.field_name
                best_terms = (entry.source_text,)
                best_candidate = candidate_text

        return SupportDecision(
            supported=best_score >= self.minimum_score,
            matched_field=best_field if best_score >= self.minimum_score else "none",
            support_score=float(best_score),
            matched_terms=best_terms if best_score > 0 else (),
            matched_candidate=best_candidate,
        )


def build_support_text_candidates(
    attribute_name: str,
    *,
    description: str = "",
    positive_patterns: Sequence[str] | None = None,
) -> list[str]:
    candidates: list[str] = []
    for value in [attribute_name, description, *(positive_patterns or [])]:
        text = _safe_text(value)
        if text and text not in candidates:
            candidates.append(text)
    return candidates
