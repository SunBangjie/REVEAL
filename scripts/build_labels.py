from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from tqdm.auto import tqdm

from audit.config import load_config, normalize_manifest_df, resolve
from audit.ontology import SECRET_PATTERNS
from audit.utils import ensure_parent


DEFAULT_TEXT_COLUMN = "text_full"
DEFAULT_ID_COLUMN = "image_id"
DEFAULT_STAGE1_FIELD = "shared_attributes"


def _load_external_schema(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        if suffix == ".json":
            return json.load(f) or {}
        if suffix == ".jsonl":
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return {"records": rows}
        raise ValueError(f"Unsupported operationalization schema format: {path}")


def _load_jsonl_or_json(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            if isinstance(obj.get("records"), list):
                return [x for x in obj["records"] if isinstance(x, dict)]
            return [obj]
    raise ValueError(f"Unsupported Stage 1 format: {path}")


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _normalize_secret_name(name: str) -> str:
    name = str(name or "").strip().lower().replace("-", "_")
    name = re.sub(r"[^a-z0-9_\s]+", " ", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _looks_like_regex_pattern(phrase: str) -> bool:
    phrase = str(phrase or "").strip()
    if not phrase:
        return False
    if phrase.startswith("re:"):
        return True
    regex_markers = (
        r"\b", r"\B", r"\d", r"\D", r"\s", r"\S", r"\w", r"\W",
        "(?:", "(?=", "(?!", "(?<=", "(?<!", r"\A", r"\Z",
    )
    return any(marker in phrase for marker in regex_markers)


def _phrase_to_safe_regex(phrase: str) -> str:
    phrase = str(phrase or "").strip()
    if not phrase:
        return ""
    if phrase.startswith("re:"):
        return phrase[3:].strip()
    if _looks_like_regex_pattern(phrase):
        return phrase

    escaped = re.escape(phrase)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"(?:-|\s+)")
    has_word = any(ch.isalnum() for ch in phrase)
    if has_word:
        return rf"\b{escaped}\b"
    return escaped


def _normalize_pattern_list(values: Iterable[str]) -> list[str]:
    return _dedupe_keep_order(_phrase_to_safe_regex(v) for v in values)


def _normalize_schema(raw: Any) -> dict[str, dict[str, list[str]]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Operationalization schema must be a mapping.")
    if "operationalization" in raw and isinstance(raw["operationalization"], dict):
        raw = raw["operationalization"]
    if "secrets" in raw and isinstance(raw["secrets"], dict):
        raw = raw["secrets"]

    normalized: dict[str, dict[str, list[str]]] = {}
    for secret, spec in raw.items():
        secret_name = _normalize_secret_name(secret) or str(secret)
        if isinstance(spec, list):
            normalized[secret_name] = {
                "positive_patterns": _normalize_pattern_list(str(x) for x in spec),
                "negative_patterns": [],
            }
            continue
        if isinstance(spec, dict):
            pos = spec.get("positive_patterns", spec.get("patterns", []))
            neg = spec.get("negative_patterns", [])
            normalized[secret_name] = {
                "positive_patterns": _normalize_pattern_list(str(x) for x in (pos or [])),
                "negative_patterns": _normalize_pattern_list(str(x) for x in (neg or [])),
            }
            continue
        raise ValueError(f"Unsupported schema entry for secret '{secret}': {type(spec)!r}")
    return normalized


def _iter_stage1_secret_items(record: dict[str, Any], field: str, max_task_relevance: float | None = None) -> list[tuple[str, dict[str, Any]]]:
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
        shared = _coerce_items("shared_attributes")
        if shared:
            return shared
        return []

    items = record.get(field, []) or []
    return [(field, item) for item in items if isinstance(item, dict)]


def _get_stage1_target_id(record: dict[str, Any], id_column: str) -> str:
    for key in (id_column, "target_image_id", "image_id", "id"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _derive_positive_patterns(item: dict[str, Any]) -> list[str]:
    pos = [str(x) for x in (item.get("positive_patterns") or []) if str(x).strip()]
    if pos:
        return _normalize_pattern_list(pos)

    evidence = [str(x) for x in (item.get("evidence") or []) if str(x).strip()]
    if evidence:
        return _normalize_pattern_list(evidence)

    name = _normalize_secret_name(item.get("name") or "")
    if name:
        return _normalize_pattern_list([name.replace("_", " ")])

    desc = str(item.get("description") or "").strip()
    if desc:
        bits = re.findall(r"[A-Za-z][A-Za-z\- ]{2,}", desc)
        return _normalize_pattern_list(bits[:2])
    return []


def _stage1_item_metrics(item: dict[str, Any]) -> tuple[float, float]:
    try:
        specificity = float(item.get("specificity", 0.0))
    except Exception:
        specificity = 0.0
    try:
        privacy = float(item.get("privacy_relevance", item.get("relevance", 0.0)))
    except Exception:
        privacy = 0.0
    return specificity, privacy


def _build_schema_from_stage1(
    stage1_records: list[dict[str, Any]],
    *,
    target_id_column: str,
    id_column: str,
    field: str,
    mode: str,
    min_specificity: float,
    min_privacy_relevance: float,
    min_support: int,
    max_task_relevance: float | None,
) -> tuple[dict[str, Any], str]:
    if mode not in {"per_target", "global", "global_consensus"}:
        raise ValueError(f"Unsupported Stage 1 operationalization mode: {mode}")

    diagnostics = {
        "records": len(stage1_records),
        "missing_target_id": 0,
        "items_seen": 0,
        "missing_name": 0,
        "filtered_specificity": 0,
        "filtered_privacy": 0,
        "filtered_task_relevance": 0,
        "filtered_patterns": 0,
    }

    if mode == "per_target":
        target_schemas: dict[str, Any] = {}
        kept_targets = 0
        kept_secrets = 0
        for record in stage1_records:
            target_id = _get_stage1_target_id(record, target_id_column)
            if not target_id:
                diagnostics["missing_target_id"] += 1
                continue
            secrets_out: dict[str, Any] = {}
            for source_kind, item in _iter_stage1_secret_items(record, field, max_task_relevance):
                diagnostics["items_seen"] += 1
                name = _normalize_secret_name(item.get("name") or "")
                if not name:
                    diagnostics["missing_name"] += 1
                    continue
                specificity, privacy = _stage1_item_metrics(item)
                if specificity < min_specificity:
                    diagnostics["filtered_specificity"] += 1
                    continue
                if privacy < min_privacy_relevance:
                    diagnostics["filtered_privacy"] += 1
                    continue
                task_rel = item.get("task_relevance", None)
                if max_task_relevance is not None and source_kind != "excess_secrets" and task_rel is not None:
                    try:
                        if float(task_rel) > float(max_task_relevance):
                            diagnostics["filtered_task_relevance"] += 1
                            continue
                    except Exception:
                        pass
                pos = _derive_positive_patterns(item)
                if not pos:
                    diagnostics["filtered_patterns"] += 1
                    continue
                neg = _normalize_pattern_list(item.get("negative_patterns") or [])
                prev = secrets_out.get(name, {})
                secrets_out[name] = {
                    "description": str(item.get("description") or prev.get("description") or "").strip(),
                    "positive_patterns": _dedupe_keep_order([*(prev.get("positive_patterns") or []), *pos]),
                    "negative_patterns": _dedupe_keep_order([*(prev.get("negative_patterns") or []), *neg]),
                    "specificity": max(float(prev.get("specificity", 0.0)), specificity),
                    "privacy_relevance": max(float(prev.get("privacy_relevance", 0.0)), privacy),
                    "source_field": source_kind,
                }
            if secrets_out:
                target_schemas[target_id] = {
                    "predicted_label": record.get("predicted_label"),
                    "predicted_label_covers": record.get("predicted_label_covers", ""),
                    "secrets": secrets_out,
                }
                kept_targets += 1
                kept_secrets += len(secrets_out)

        payload = {
            "mode": "per_target",
            "text_column": DEFAULT_TEXT_COLUMN,
            "id_column": id_column,
            "target_id_column": target_id_column,
            "target_key": id_column,
            "generated_from": "stage1_jsonl",
            "target_schemas": target_schemas,
        }
        summary = (
            f"stage1 per_target | records={diagnostics['records']} | targets={kept_targets} | secrets={kept_secrets} "
            f"| items_seen={diagnostics['items_seen']} | missing_target_id={diagnostics['missing_target_id']} "
            f"| filtered_specificity={diagnostics['filtered_specificity']} | filtered_privacy={diagnostics['filtered_privacy']} "
            f"| filtered_task_relevance={diagnostics['filtered_task_relevance']} | filtered_patterns={diagnostics['filtered_patterns']}"
        )
        return payload, summary

    agg: dict[str, dict[str, Any]] = {}
    for record in stage1_records:
        target_id = _get_stage1_target_id(record, target_id_column)
        if not target_id:
            diagnostics["missing_target_id"] += 1
        for source_kind, item in _iter_stage1_secret_items(record, field, max_task_relevance):
            diagnostics["items_seen"] += 1
            name = _normalize_secret_name(item.get("name") or "")
            if not name:
                diagnostics["missing_name"] += 1
                continue
            specificity, privacy = _stage1_item_metrics(item)
            if specificity < min_specificity:
                diagnostics["filtered_specificity"] += 1
                continue
            if privacy < min_privacy_relevance:
                diagnostics["filtered_privacy"] += 1
                continue
            task_rel = item.get("task_relevance", None)
            if max_task_relevance is not None and source_kind != "excess_secrets" and task_rel is not None:
                try:
                    if float(task_rel) > float(max_task_relevance):
                        diagnostics["filtered_task_relevance"] += 1
                        continue
                except Exception:
                    pass
            pos = _derive_positive_patterns(item)
            if not pos:
                diagnostics["filtered_patterns"] += 1
                continue
            neg = _normalize_pattern_list(item.get("negative_patterns") or [])
            slot = agg.setdefault(name, {
                "positive_patterns": [],
                "negative_patterns": [],
                "support_targets": set(),
                "support_count": 0,
                "specificities": [],
                "privacy_scores": [],
                "source_fields": set(),
            })
            slot["positive_patterns"] = _dedupe_keep_order([*slot["positive_patterns"], *pos])
            slot["negative_patterns"] = _dedupe_keep_order([*slot["negative_patterns"], *neg])
            if target_id and target_id not in slot["support_targets"]:
                slot["support_targets"].add(target_id)
                slot["support_count"] += 1
            slot["specificities"].append(specificity)
            slot["privacy_scores"].append(privacy)
            slot["source_fields"].add(source_kind)

    normalized: dict[str, dict[str, list[str]]] = {}
    kept = 0
    for name, slot in sorted(agg.items()):
        support_count = int(slot["support_count"])
        if support_count < min_support:
            continue
        normalized[name] = {
            "positive_patterns": slot["positive_patterns"],
            "negative_patterns": slot["negative_patterns"],
        }
        kept += 1

    payload = normalized
    summary = (
        f"stage1 {mode} | records={diagnostics['records']} | secrets={kept} | min_support={min_support} "
        f"| items_seen={diagnostics['items_seen']} | missing_target_id={diagnostics['missing_target_id']} "
        f"| filtered_specificity={diagnostics['filtered_specificity']} | filtered_privacy={diagnostics['filtered_privacy']} "
        f"| filtered_task_relevance={diagnostics['filtered_task_relevance']} | filtered_patterns={diagnostics['filtered_patterns']}"
    )
    return payload, summary


def _load_operationalization(cfg: dict[str, Any], project_root: Path) -> tuple[dict[str, Any], str, str, str, str]:
    op_cfg = cfg.get("operationalization") or {}
    text_column = str(op_cfg.get("text_column", DEFAULT_TEXT_COLUMN))
    id_column = str(op_cfg.get("id_column", DEFAULT_ID_COLUMN))
    schema_source = ""
    mode = "global"
    schema_payload: dict[str, Any] = {}

    source_mode = str(op_cfg.get("source", "auto")).strip().lower()
    stage1_path_cfg = op_cfg.get("stage1_path") or (cfg.get("paths") or {}).get("stage1_jsonl")
    from_stage1_field = str(op_cfg.get("from_stage1_field", DEFAULT_STAGE1_FIELD)).strip().lower()
    from_stage1_mode = str(op_cfg.get("from_stage1_mode", "per_target")).strip().lower()
    target_id_column = str(op_cfg.get("target_id_column", id_column))
    min_specificity = float(op_cfg.get("min_specificity", 0.4))
    has_explicit_priv = "min_privacy_relevance" in op_cfg
    default_priv = 0.15 if from_stage1_field in {"excess_secrets", "both"} else 0.0
    min_privacy_relevance = float(op_cfg.get("min_privacy_relevance", default_priv))
    min_support = int(op_cfg.get("min_support", 1 if from_stage1_mode == "per_target" else 2))
    max_task_relevance_cfg = op_cfg.get("max_task_relevance", None)
    max_task_relevance = None if max_task_relevance_cfg in {None, "", "null"} else float(max_task_relevance_cfg)

    should_try_stage1 = source_mode in {"stage1", "auto"} and bool(stage1_path_cfg)
    if should_try_stage1:
        stage1_path = resolve(stage1_path_cfg, project_root)
        if stage1_path.exists():
            stage1_records = _load_jsonl_or_json(stage1_path)
            if max_task_relevance is None:
                for rec in stage1_records:
                    val = rec.get("excess_task_relevance_threshold", rec.get("stage1_excess_max_task_relevance")) if isinstance(rec, dict) else None
                    if val not in {None, "", "null"}:
                        try:
                            max_task_relevance = float(val)
                            break
                        except Exception:
                            pass
            schema_payload, stage1_summary = _build_schema_from_stage1(
                stage1_records,
                target_id_column=target_id_column,
                id_column=id_column,
                field=from_stage1_field,
                mode=from_stage1_mode,
                min_specificity=min_specificity,
                min_privacy_relevance=min_privacy_relevance,
                min_support=min_support,
                max_task_relevance=max_task_relevance,
            )
            if from_stage1_mode == "per_target" and not ((schema_payload or {}).get("target_schemas") or {}) and from_stage1_field in {"excess_secrets", "both"} and not has_explicit_priv:
                relaxed_priv = 0.0
                schema_payload, stage1_summary = _build_schema_from_stage1(
                    stage1_records,
                    target_id_column=target_id_column,
                    id_column=id_column,
                    field=from_stage1_field,
                    mode=from_stage1_mode,
                    min_specificity=min_specificity,
                    min_privacy_relevance=relaxed_priv,
                    min_support=min_support,
                )
                stage1_summary += f" | auto_relaxed_min_privacy={relaxed_priv}"
            mode = "per_target" if from_stage1_mode == "per_target" else "global"
            schema_source = f"stage1-derived schema: {stage1_path_cfg} | field={from_stage1_field} | target_id_column={target_id_column} | min_specificity={min_specificity} | min_privacy_relevance={min_privacy_relevance} | max_task_relevance={max_task_relevance} | {stage1_summary}"
            return schema_payload, text_column, id_column, schema_source, mode
        if source_mode == "stage1":
            raise FileNotFoundError(f"operationalization.stage1_path not found: {stage1_path}")

    schema_path = op_cfg.get("schema_path") or op_cfg.get("path")
    if schema_path:
        external = _load_external_schema(resolve(schema_path, project_root))
        wrapper = external.get("operationalization", external) if isinstance(external, dict) else {}
        if isinstance(wrapper, dict) and "target_schemas" in wrapper:
            mode = str(wrapper.get("mode", "per_target"))
            schema_payload = wrapper
            text_column = str(wrapper.get("text_column", text_column))
            id_column = str(wrapper.get("id_column", id_column))
        else:
            schema_payload = _normalize_schema(external)
            mode = "global"
        schema_source = f"external schema: {schema_path}"
    elif "secrets" in op_cfg:
        schema_payload = _normalize_schema(op_cfg)
        schema_source = "config operationalization.secrets"
        mode = "global"
    else:
        schema_payload = _normalize_schema(SECRET_PATTERNS)
        schema_source = "audit.ontology.SECRET_PATTERNS fallback"
        mode = "global"

    return schema_payload, text_column, id_column, schema_source, mode


def _resolve_label_scope(cfg: dict[str, Any], args: argparse.Namespace, mode: str) -> tuple[bool, str]:
    op_cfg = cfg.get("operationalization") or {}
    cli_label_split = getattr(args, "label_split", None)
    cfg_label_split = op_cfg.get("label_split")
    explicit_full_manifest = op_cfg.get("label_on_full_manifest", None)

    if getattr(args, "full_manifest", False):
        use_full_manifest = True
    elif cli_label_split is not None or cfg_label_split is not None:
        # An explicit split selection should always restrict the labeling pool,
        # even if the base config defaults per-target labeling to the full manifest.
        use_full_manifest = False
    elif explicit_full_manifest is not None:
        use_full_manifest = bool(explicit_full_manifest)
    elif mode == "per_target":
        # Per-target excess confirmation needs broad support by default. Restricting
        # labels to the audited split frequently starves q(S|Y) and forces Stage 3
        # into blanket abstention.
        use_full_manifest = True
    else:
        use_full_manifest = False

    label_split = (
        cli_label_split
        or cfg_label_split
        or (cfg.get("audit") or {}).get("target_split")
        or (cfg.get("probe") or {}).get("target_split")
        or "audited"
    )
    return use_full_manifest, str(label_split)


def _restrict_manifest_to_label_scope(
    manifest: pd.DataFrame,
    cfg: dict[str, Any],
    project_root: Path,
    id_column: str,
    use_full_manifest: bool,
    label_split: str,
) -> tuple[pd.DataFrame, str]:
    if use_full_manifest:
        return manifest.copy(), "full_manifest"

    splits_path_cfg = (cfg.get("paths") or {}).get("splits_csv")
    if not splits_path_cfg:
        print("[WARN] paths.splits_csv is not configured; falling back to full manifest for labeling.", file=sys.stderr)
        return manifest.copy(), "full_manifest_fallback_no_splits_path"

    splits_path = resolve(splits_path_cfg, project_root)
    if not splits_path.exists():
        print(f"[WARN] splits file not found at {splits_path}; falling back to full manifest for labeling.", file=sys.stderr)
        return manifest.copy(), "full_manifest_fallback_missing_splits"

    splits = pd.read_csv(splits_path)
    if id_column not in splits.columns or "split" not in splits.columns:
        print(
            f"[WARN] splits file {splits_path} does not contain required columns '{id_column}' and 'split'; "
            "falling back to full manifest for labeling.",
            file=sys.stderr,
        )
        return manifest.copy(), "full_manifest_fallback_bad_splits"

    split_counts = splits["split"].astype(str).value_counts().to_dict()
    if label_split not in set(splits["split"].astype(str)):
        print(
            f"[WARN] requested label split '{label_split}' not found in {splits_path}. "
            f"Available splits: {sorted(set(splits['split'].astype(str)))}. Falling back to full manifest.",
            file=sys.stderr,
        )
        return manifest.copy(), "full_manifest_fallback_unknown_split"

    scoped_ids = splits.loc[splits["split"].astype(str) == label_split, [id_column]].drop_duplicates()
    scoped_manifest = manifest.merge(scoped_ids, on=id_column, how="inner")
    print(f"[INFO] Restricting label generation to split '{label_split}' from {splits_path} with counts {split_counts}")
    return scoped_manifest, f"split:{label_split}"


class RegexLabeler:
    def __init__(self, schema: dict[str, dict[str, list[str]]]) -> None:
        self.compiled: dict[str, tuple[list[re.Pattern[str]], list[re.Pattern[str]]]] = {}
        for secret, spec in schema.items():
            pos = [re.compile(p, flags=re.IGNORECASE) for p in spec.get("positive_patterns", []) if str(p).strip()]
            neg = [re.compile(p, flags=re.IGNORECASE) for p in spec.get("negative_patterns", []) if str(p).strip()]
            if not pos:
                raise ValueError(f"Secret '{secret}' has no positive_patterns.")
            self.compiled[secret] = (pos, neg)

    def assign(self, text: str) -> dict[str, int]:
        text = str(text or "")
        labels: dict[str, int] = {}
        for secret, (pos_patterns, neg_patterns) in self.compiled.items():
            pos_hit = any(p.search(text) for p in pos_patterns)
            neg_hit = any(p.search(text) for p in neg_patterns)
            labels[secret] = int(pos_hit and not neg_hit)
        return labels


def _best_secret_name(names: Iterable[str]) -> str:
    unique = _dedupe_keep_order(names)
    if not unique:
        return ""
    return sorted(unique, key=lambda x: (len(x), x))[0]


def _label_signature(values: Iterable[int]) -> str:
    bits = "".join("1" if int(v) else "0" for v in values)
    digest = hashlib.sha1(bits.encode("utf-8")).hexdigest()[:16]
    return f"n{len(bits)}_p{bits.count('1')}_{digest}"


def _build_diagnostics_path(out_csv: Path) -> Path:
    return out_csv.with_name(out_csv.stem + ".label_diagnostics.json")


def _write_diagnostics(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _collapse_duplicate_global_labels(
    df: pd.DataFrame, id_column: str, *, collapse: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return df, {
            "mode": "global",
            "deduplication_enabled": bool(collapse),
            "exact_duplicate_groups": [],
            "constant_columns": [],
            "n_duplicate_groups": 0,
            "n_detected_aliases": 0,
            "n_collapsed_aliases": 0,
        }

    secret_cols = [c for c in df.columns if c != id_column]
    signature_to_cols: dict[str, list[str]] = defaultdict(list)
    signature_meta: dict[str, dict[str, Any]] = {}
    for secret in secret_cols:
        values = df[secret].fillna(0).astype(int).tolist()
        signature = _label_signature(values)
        signature_to_cols[signature].append(secret)
        signature_meta[signature] = {
            "positive_count": int(sum(values)),
            "negative_count": int(len(values) - sum(values)),
            "n_rows": int(len(values)),
            "positive_rate": float(sum(values) / len(values)) if values else 0.0,
        }

    groups = []
    keep_cols = [id_column]
    constant_columns = []
    for signature, cols in sorted(signature_to_cols.items(), key=lambda kv: (_best_secret_name(kv[1]), kv[0])):
        canonical = _best_secret_name(cols)
        aliases = [c for c in cols if c != canonical]
        keep_cols.append(canonical)
        meta = signature_meta[signature]
        if meta["positive_count"] in {0, meta["n_rows"]}:
            constant_columns.append({"secret": canonical, **meta})
        if aliases:
            groups.append({
                "signature": signature,
                "canonical_secret": canonical,
                "alias_secrets": aliases,
                **meta,
            })

    if collapse:
        out = df[keep_cols].copy()
        ordered_cols = [id_column] + sorted([c for c in out.columns if c != id_column])
        n_collapsed_aliases = int(sum(len(g["alias_secrets"]) for g in groups))
    else:
        out = df.copy()
        ordered_cols = [id_column] + sorted(secret_cols)
        n_collapsed_aliases = 0
    diagnostics = {
        "mode": "global",
        "deduplication_enabled": bool(collapse),
        "n_input_secrets": len(secret_cols),
        "n_output_secrets": len(ordered_cols) - 1,
        "n_duplicate_groups": len(groups),
        "n_detected_aliases": int(sum(len(g["alias_secrets"]) for g in groups)),
        "n_collapsed_aliases": n_collapsed_aliases,
        "exact_duplicate_groups": groups,
        "constant_columns": constant_columns,
    }
    return out[ordered_cols], diagnostics


def _collapse_duplicate_per_target_labels(
    df: pd.DataFrame, id_column: str, *, collapse: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return df, {
            "mode": "per_target",
            "deduplication_enabled": bool(collapse),
            "targets": [],
            "n_duplicate_groups": 0,
            "n_detected_aliases": 0,
            "n_collapsed_aliases": 0,
        }

    work = df.copy()
    work["target_image_id"] = work["target_image_id"].astype(str)
    work[id_column] = work[id_column].astype(str)
    work["secret"] = work["secret"].astype(str)
    work["label"] = work["label"].fillna(0).astype(int)

    all_targets = sorted(work["target_image_id"].unique())
    keep_pairs: set[tuple[str, str]] = set()
    target_reports: list[dict[str, Any]] = []
    total_aliases = 0
    total_groups = 0

    for target_id in all_targets:
        sub = work[work["target_image_id"] == target_id].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(index=id_column, columns="secret", values="label", aggfunc="max", fill_value=0)
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)

        signature_to_cols: dict[str, list[str]] = defaultdict(list)
        signature_meta: dict[str, dict[str, Any]] = {}
        for secret in pivot.columns.tolist():
            values = pivot[secret].astype(int).tolist()
            signature = _label_signature(values)
            signature_to_cols[signature].append(secret)
            signature_meta[signature] = {
                "positive_count": int(sum(values)),
                "negative_count": int(len(values) - sum(values)),
                "n_rows": int(len(values)),
                "positive_rate": float(sum(values) / len(values)) if values else 0.0,
            }

        duplicate_groups = []
        constant_columns = []
        kept_secrets: list[str] = []
        for signature, cols in sorted(signature_to_cols.items(), key=lambda kv: (_best_secret_name(kv[1]), kv[0])):
            canonical = _best_secret_name(cols)
            aliases = [c for c in cols if c != canonical]
            kept_secrets.append(canonical)
            keep_pairs.add((target_id, canonical))
            meta = signature_meta[signature]
            if meta["positive_count"] in {0, meta["n_rows"]}:
                constant_columns.append({"secret": canonical, **meta})
            if aliases:
                duplicate_groups.append({
                    "signature": signature,
                    "canonical_secret": canonical,
                    "alias_secrets": aliases,
                    **meta,
                })

        detected_aliases = int(sum(len(g["alias_secrets"]) for g in duplicate_groups))
        total_groups += len(duplicate_groups)
        total_aliases += detected_aliases
        target_reports.append({
            "target_image_id": target_id,
            "n_input_secrets": int(sub["secret"].nunique()),
            "n_output_secrets": int(len(kept_secrets) if collapse else sub["secret"].nunique()),
            "n_duplicate_groups": int(len(duplicate_groups)),
            "n_detected_aliases": detected_aliases,
            "n_collapsed_aliases": int(detected_aliases if collapse else 0),
            "exact_duplicate_groups": duplicate_groups,
            "constant_columns": constant_columns,
        })

    if collapse:
        filtered = work[work.apply(lambda row: (str(row["target_image_id"]), str(row["secret"])) in keep_pairs, axis=1)].copy()
        n_collapsed_aliases = int(total_aliases)
    else:
        filtered = work.copy()
        n_collapsed_aliases = 0
    filtered = filtered.sort_values(["target_image_id", "secret", id_column], kind="stable").reset_index(drop=True)
    diagnostics = {
        "mode": "per_target",
        "deduplication_enabled": bool(collapse),
        "n_targets": len(all_targets),
        "n_input_target_secret_pairs": int(work[["target_image_id", "secret"]].drop_duplicates().shape[0]),
        "n_output_target_secret_pairs": int(filtered[["target_image_id", "secret"]].drop_duplicates().shape[0]),
        "n_duplicate_groups": int(total_groups),
        "n_detected_aliases": int(total_aliases),
        "n_collapsed_aliases": n_collapsed_aliases,
        "targets": target_reports,
    }
    return filtered, diagnostics


def _build_global_labels(manifest: pd.DataFrame, id_column: str, text_column: str, schema: dict[str, Any]) -> pd.DataFrame:
    labeler = RegexLabeler(schema)
    records = list(manifest[[id_column, text_column]].fillna("").itertuples(index=False, name=None))
    rows = []
    with tqdm(total=len(records), desc="Build labels", unit="image") as pbar:
        for image_id, text in records:
            labels = labeler.assign(str(text))
            rows.append({id_column: image_id, **labels})
            pbar.update(1)
    df = pd.DataFrame(rows)
    ordered_cols = [id_column] + sorted([c for c in df.columns if c != id_column])
    return df[ordered_cols]


def _build_per_target_labels(manifest: pd.DataFrame, id_column: str, text_column: str, payload: dict[str, Any]) -> pd.DataFrame:
    target_schemas = payload.get("target_schemas") or {}
    target_items = [(target_id, target_spec) for target_id, target_spec in target_schemas.items() if (target_spec or {}).get("secrets")]
    records = list(manifest[[id_column, text_column]].fillna("").itertuples(index=False, name=None))
    total_work = len(target_items) * len(records)
    rows: list[dict[str, Any]] = []

    with tqdm(total=total_work, desc="Build labels", unit="target×image") as pbar:
        for target_id, target_spec in target_items:
            secrets = (target_spec or {}).get("secrets") or {}
            labeler = RegexLabeler(_normalize_schema({"secrets": secrets}))
            pbar.set_postfix_str(f"target={target_id}")
            for image_id, text in records:
                assigned = labeler.assign(str(text))
                for secret, label in assigned.items():
                    rows.append({
                        "target_image_id": str(target_id),
                        id_column: image_id,
                        "secret": str(secret),
                        "label": int(label),
                    })
                pbar.update(1)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--label-split",
        default=None,
        help="Split to label against. If omitted, per-target Stage 1 operationalization labels the full manifest by default; otherwise falls back to operationalization.label_split, audit.target_split, probe.target_split, then audited.",
    )
    ap.add_argument(
        "--full-manifest",
        action="store_true",
        help="Label the full manifest instead of restricting to the configured target split.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    manifest = pd.read_csv(resolve(cfg["paths"]["manifest_csv"], project_root))
    manifest = normalize_manifest_df(manifest, cfg, project_root)
    out_csv = resolve(cfg["paths"]["labels_csv"], project_root)
    ensure_parent(out_csv)

    schema_payload, text_column, id_column, schema_source, mode = _load_operationalization(cfg, project_root)
    collapse_identical_label_signatures = bool(
        ((cfg.get("operationalization") or {}).get("collapse_identical_label_signatures", False))
    )
    use_full_manifest, label_split = _resolve_label_scope(cfg, args, mode)
    if text_column not in manifest.columns:
        raise KeyError(f"Configured text_column '{text_column}' not found in manifest columns: {list(manifest.columns)}")
    if id_column not in manifest.columns:
        raise KeyError(f"Configured id_column '{id_column}' not found in manifest columns: {list(manifest.columns)}")

    scoped_manifest, label_scope = _restrict_manifest_to_label_scope(
        manifest, cfg, project_root, id_column, use_full_manifest, label_split
    )
    if scoped_manifest.empty:
        raise ValueError(f"No rows available for label generation under scope {label_scope}. Check splits.csv and the selected label split.")

    if mode == "per_target":
        target_schemas = (schema_payload or {}).get("target_schemas") or {}
        if not target_schemas:
            raise ValueError(
                "No per-target schemas were built from Stage 1. Loader summary: " + schema_source + "\n"
                + "Likely fixes:\n"
                + "- set operationalization.id_column=image_id and, if needed, operationalization.target_id_column=image_id\n"
                + "- set operationalization.from_stage1_field=shared_attributes (preferred) or excess_secrets\n"
                + "- lower operationalization.min_privacy_relevance to 0.0-0.15 if broad semantics are being filtered\n"
                + "- raise operationalization.max_task_relevance if too few excess secrets survive\n"
                + "- lower operationalization.min_specificity to 0.2-0.4\n"
            )
        raw_df = _build_per_target_labels(scoped_manifest, id_column, text_column, schema_payload)
        df, diagnostics = _collapse_duplicate_per_target_labels(
            raw_df,
            id_column,
            collapse=collapse_identical_label_signatures,
        )
    else:
        schema = schema_payload if isinstance(schema_payload, dict) else {}
        if not schema:
            raise ValueError(
                "No operationalization secrets found. Provide operationalization.secrets in the config, "
                "or set operationalization.schema_path / stage1_path, or keep audit.ontology.SECRET_PATTERNS as a fallback."
            )
        raw_df = _build_global_labels(scoped_manifest, id_column, text_column, schema)
        df, diagnostics = _collapse_duplicate_global_labels(
            raw_df,
            id_column,
            collapse=collapse_identical_label_signatures,
        )

    df.to_csv(out_csv, index=False)

    diagnostics_path = _build_diagnostics_path(out_csv)
    diagnostics_payload = {
        "labels_csv": str(out_csv),
        "schema_source": schema_source,
        "operationalization_mode": mode,
        "collapse_identical_label_signatures": collapse_identical_label_signatures,
        "text_column": text_column,
        "id_column": id_column,
        "label_scope": label_scope,
        "scoped_manifest_rows": int(len(scoped_manifest)),
        "label_rows": int(len(df)),
        "deduplication": diagnostics,
    }
    _write_diagnostics(diagnostics_path, diagnostics_payload)

    print(f"Saved labels to {out_csv}")
    print(f"[INFO] Using operationalization from: {schema_source}")
    print(f"[INFO] Operationalization mode: {mode}")
    print(f"[INFO] Text column: {text_column}")
    print(f"[INFO] Label scope: {label_scope}")
    print(f"[INFO] Scoped manifest rows: {len(scoped_manifest)}")
    print(f"[INFO] Collapse identical label signatures: {collapse_identical_label_signatures}")
    print(f"[INFO] Saved label diagnostics to {diagnostics_path}")
    if mode == "per_target":
        print(f"[INFO] Label table rows: {len(df)}")
        if not df.empty:
            print(f"[INFO] Target schemas: {df['target_image_id'].nunique()} | Secrets: {df['secret'].nunique()}")
            print(
                f"[INFO] Label-signature duplicates: duplicate_groups={diagnostics['n_duplicate_groups']} "
                f"| detected_aliases={diagnostics['n_detected_aliases']} "
                f"| collapsed_aliases={diagnostics['n_collapsed_aliases']}"
            )
    else:
        print(f"[INFO] Label columns: {', '.join(c for c in df.columns if c != id_column)}")
        print(
            f"[INFO] Label-signature duplicates: duplicate_groups={diagnostics['n_duplicate_groups']} "
            f"| detected_aliases={diagnostics['n_detected_aliases']} "
            f"| collapsed_aliases={diagnostics['n_collapsed_aliases']}"
        )


if __name__ == "__main__":
    main()
