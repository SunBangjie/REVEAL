"""Per-target audit reports — one Markdown file per audited target.

Reads Stage 1 discovery outputs and Stage 3 confirmation results
to produce human-readable per-target reports with natural-language
screening decisions and supporting evidence.

Usage:
    python scripts/report_targets.py --config configs/mvp.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.config import load_config, normalize_manifest_df, resolve, resolve_image_path_from_record
from audit.utils import ensure_parent, safe_float as _safe_float

SORT_PRIORITY_COLS = ["posterior_shift", "excess_kl_nats", "conditional_posterior", "p_secret_given_baseline_1", "p_secret_1"]


def _get_path(cfg: dict, project_root: Path, key: str, default_rel: str) -> Path:
    rel = cfg.get("paths", {}).get(key, default_rel)
    return resolve(rel, project_root)


def _safe_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    return [str(x)] if str(x).strip() else []


def _load_stage1(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_posteriors(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["image_id", "target_image_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _load_manifest(path: Path | None, cfg: dict, project_root: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = normalize_manifest_df(df, cfg, project_root)
    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str)
    return df


def _format_float(x: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "NA"


def _float_or_none(x: Any) -> float | None:
    try:
        value = float(x)
        if pd.isna(value):
            return None
        return value
    except Exception:
        return None


def _posterior_variant_col(df: pd.DataFrame) -> str:
    for col in SORT_PRIORITY_COLS:
        if col in df.columns:
            return col
    return "p_secret_1"


def _method_col(df: pd.DataFrame) -> str | None:
    for col in ["model", "method"]:
        if col in df.columns:
            return col
    return None


def _pick_top_posteriors(df: pd.DataFrame, method_filter: str | None, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    method_col = _method_col(work)
    if method_filter and method_col:
        work = work[work[method_col].astype(str) == str(method_filter)]
    if work.empty:
        return work
    if "confirmation_status" in work.columns:
        priority = {"confirmed": 0, "inconclusive": 1, "rejected": 2}
        work["__status_priority"] = work["confirmation_status"].map(priority).fillna(3).astype(int)
    sort_col = _posterior_variant_col(work)
    secondary = "conditional_posterior" if "conditional_posterior" in work.columns else (
        "p_secret_given_baseline_1" if "p_secret_given_baseline_1" in work.columns else "p_secret_1"
    )
    sort_cols = [sort_col, secondary]
    ascending = [False, False]
    if "__status_priority" in work.columns:
        sort_cols = ["__status_priority", *sort_cols]
        ascending = [True, *ascending]
    work = work.sort_values(by=sort_cols, ascending=ascending)
    if "__status_priority" in work.columns:
        work = work.drop(columns=["__status_priority"])
    return work.head(top_n)




def _safe_json_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
    except Exception:
        return [s]
    if isinstance(parsed, list):
        return [str(v) for v in parsed if str(v).strip()]
    return [str(parsed)]


def _safe_json_dict(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        parsed = json.loads(s)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_confirmation_thresholds(row: pd.Series) -> dict[str, Any]:
    thresholds = _safe_json_dict(row.get("thresholds_json"))
    for key in row.index:
        key_str = str(key)
        if not key_str.startswith("threshold_"):
            continue
        value = row.get(key)
        if _float_or_none(value) is None and str(value).strip().lower() in {"", "nan", "none"}:
            continue
        thresholds[key_str.removeprefix("threshold_")] = value
    return thresholds


def _confirmation_rule_lines(chosen: pd.DataFrame) -> list[str]:
    if chosen.empty:
        return []
    thresholds = _extract_confirmation_thresholds(chosen.iloc[0])
    min_shift = _float_or_none(thresholds.get("min_posterior_shift"))
    min_lift = _float_or_none(thresholds.get("min_excess_lift"))
    min_excess_kl = _float_or_none(thresholds.get("min_excess_kl"))
    min_neighbor_support = int(_float_or_none(thresholds.get("min_neighbor_support")) or 0)

    parts = []
    if min_shift is not None:
        parts.append(f"posterior_shift >= {min_shift:.3f}")
    if min_lift is not None:
        parts.append(f"excess_lift >= {min_lift:.2f}x")
    if min_excess_kl is not None:
        parts.append(f"excess_KL >= {min_excess_kl:.4f} nats")
    evidence_text = ", ".join(parts) if parts else "the configured evidence thresholds"

    lines = [
        "- Decision rule: a secret is confirmed only when all incremental-evidence checks pass and no hard support failure blocks confirmation.",
        f"- Evidence thresholds: {evidence_text}.",
    ]
    support_text = "baseline_available, support_status_ok, neighbor_support_ok"
    if min_neighbor_support > 0:
        support_text += f" (with at least {min_neighbor_support} supporting neighbors)"
    lines.append(f"- Hard support checks: {support_text}.")
    lines.append("- If the evidence thresholds pass but a hard support check fails, the result is marked inconclusive. If any evidence threshold fails, the result is rejected.")
    return lines


def _dedupe_identical_signature_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "label_signature" not in df.columns:
        return df
    work = df.copy()
    work["__dedup_key"] = work["label_signature"].fillna("").astype(str)
    empty_mask = work["__dedup_key"].eq("") | work["__dedup_key"].eq("nan")
    work.loc[empty_mask, "__dedup_key"] = [f"__row__{i}" for i in work.index[empty_mask]]
    sort_col = _posterior_variant_col(work)
    secondary = "p_secret_given_baseline_1" if "p_secret_given_baseline_1" in work.columns else "p_secret_1"
    work = work.sort_values(by=[sort_col, secondary], ascending=[False, False])
    work = work.drop_duplicates(subset=["__dedup_key"], keep="first")
    return work.drop(columns=["__dedup_key"])


def _alias_group_notes(df: pd.DataFrame) -> list[str]:
    if df.empty or "secret_alias_count" not in df.columns or "secret_aliases_json" not in df.columns:
        return []
    seen_groups: set[tuple[str, ...]] = set()
    notes: list[str] = []
    for _, row in df.iterrows():
        alias_count = int(_safe_float(row.get("secret_alias_count"), 1.0))
        aliases = sorted(set(_safe_json_list(row.get("secret_aliases_json"))))
        if alias_count <= 1 or len(aliases) <= 1:
            continue
        group_key = tuple(aliases)
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        notes.append("Identical operational label support: %s" % ", ".join("`%s`" % alias for alias in aliases))
    return notes


def _append_metric(parts: list[str], label: str, value: Any, *, ndigits: int = 4, suffix: str = "") -> None:
    parsed = _float_or_none(value)
    if parsed is None:
        return
    parts.append(f"{label}={parsed:.{ndigits}f}{suffix}")

def _stage1_attribute_names(item: dict) -> list[str]:
    attrs = []
    for a in item.get("shared_attributes", []) or []:
        if isinstance(a, dict):
            name = str(a.get("name", "")).strip()
            if name:
                attrs.append(name)
        elif str(a).strip():
            attrs.append(str(a).strip())
    attrs.extend(_safe_list(item.get("candidate_secrets")))
    seen = set()
    out = []
    for a in attrs:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _build_image_lookup(manifest_df: pd.DataFrame, cfg: dict, project_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    if manifest_df.empty or "image_id" not in manifest_df.columns:
        return lookup
    for _, row in manifest_df.iterrows():
        image_id = str(row.get("image_id", "")).strip()
        if not image_id or image_id in lookup:
            continue
        p = resolve_image_path_from_record(row, cfg, project_root)
        if p is not None:
            lookup[image_id] = p
    return lookup


def _to_report_relative(path: Path, report_path: Path) -> str:
    try:
        rel = os.path.relpath(str(path), start=str(report_path.parent))
        return rel.replace(os.sep, "/")
    except Exception:
        return path.as_posix()


def _image_block(image_id: str, image_lookup: dict[str, Path], report_path: Path, width: int, alt_prefix: str) -> list[str]:
    path = image_lookup.get(str(image_id))
    if path is None:
        return [f"- {alt_prefix} image: not found for `{image_id}`"]
    src = _to_report_relative(path, report_path)
    return [
        f"- {alt_prefix} image path: `{src}`",
        f'<img src="{src}" alt="{alt_prefix} {image_id}" width="{width}">',
    ]


def _slugify_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    s = s.strip("._-") or "target"
    return s + ".md"


def _resolve_output_dir(cfg: dict, project_root: Path, cli_output_dir: str | None) -> Path:
    if cli_output_dir:
        return resolve(cli_output_dir, project_root)
    target_report_dir = cfg.get("paths", {}).get("target_report_dir")
    if target_report_dir:
        return resolve(target_report_dir, project_root)
    target_report_md = cfg.get("paths", {}).get("target_report_md")
    if target_report_md:
        md_path = resolve(target_report_md, project_root)
        return md_path.parent / (md_path.stem + "_targets")
    return resolve("data/reports/target_reports", project_root)


# ── Natural-language interpretation of confirmation outcomes ──────────

def _kl_severity(kl_nats: float) -> str:
    """Qualitative label for a KL divergence value."""
    if kl_nats < 0.005:
        return "negligible"
    if kl_nats < 0.02:
        return "minor"
    if kl_nats < 0.10:
        return "moderate"
    if kl_nats < 0.50:
        return "substantial"
    return "very high"


def _pct_str(p: float) -> str:
    return f"{100.0 * p:.1f}%"


def _interpret_secret_row(row: pd.Series) -> str:
    """Generate a concise bottom-line screening summary for one attribute."""

    secret = str(row.get("attribute_name", row.get("secret", "this attribute")))
    status = str(row.get("confirmation_status", row.get("status", "inconclusive"))).strip().lower()
    baseline_p = _safe_float(row.get("baseline_posterior", row.get("p_baseline_1")), 0.5)
    conditional_p = _safe_float(row.get("conditional_posterior", row.get("p_secret_given_baseline_1")), baseline_p)
    shift = _safe_float(row.get("posterior_shift", row.get("delta_p")), 0.0)
    lift = _safe_float(row.get("excess_lift"), 1.0)
    excess_kl = _safe_float(row.get("excess_kl_nats", row.get("excess_local_kl_nats")), 0.0)
    confidence = str(row.get("confidence", ""))
    failures = _safe_json_list(row.get("support_failures_json"))
    rationale = str(row.get("rationale", "")).strip()

    evidence = (
        f"q(S|Y)={_pct_str(baseline_p)}, q(S|Z,Y)={_pct_str(conditional_p)}, "
        f"shift={shift:+.3f}, lift={lift:.2f}x, excess_KL={excess_kl:.4f} nats"
    )
    if confidence:
        evidence += f", confidence={confidence}"

    if status == "confirmed":
        return f"The embedding provides additional evidence for `{secret}` beyond the task label. {evidence}."
    if status == "rejected":
        if shift <= 0.01:
            return f"The observed signal for `{secret}` is largely explained by the task label. {evidence}."
        return f"`{secret}` is rejected because the incremental evidence beyond the task label does not clear the confirmation thresholds. {evidence}."

    if failures:
        return f"Evidence for `{secret}` is insufficient to confirm excess leakage. {evidence}. Reliability concerns: {', '.join(failures)}."
    if rationale:
        return rationale
    return f"Evidence for `{secret}` is insufficient to confirm excess leakage. {evidence}."


def _interpret_target_summary(chosen: pd.DataFrame) -> str:
    """Generate a short overall summary across screened attributes for one target."""
    if chosen.empty:
        return ""

    n_attrs = chosen["attribute_name"].nunique() if "attribute_name" in chosen.columns else (
        chosen["secret"].nunique() if "secret" in chosen.columns else len(chosen)
    )
    status_col = "confirmation_status" if "confirmation_status" in chosen.columns else "status"
    statuses = chosen[status_col].fillna("").astype(str) if status_col in chosen.columns else pd.Series(dtype=str)
    n_confirmed = int((statuses == "confirmed").sum()) if not statuses.empty else 0
    n_rejected = int((statuses == "rejected").sum()) if not statuses.empty else 0
    n_inconclusive = int((statuses == "inconclusive").sum()) if not statuses.empty else 0

    if n_confirmed == 0 and n_inconclusive == 0:
        return (
            f"**Overall**: Across {n_attrs} screened attribute(s), "
            "the observed signals are largely explained by the task label or too weak to matter."
        )

    parts = [f"**Overall**: Across {n_attrs} screened attribute(s)"]
    if n_confirmed:
        parts.append(f"{n_confirmed} confirmed as excess")
    if n_inconclusive:
        parts.append(f"{n_inconclusive} inconclusive")
    if n_rejected:
        parts.append(f"{n_rejected} rejected")
    lead = ", ".join(parts) + "."

    if "posterior_shift" in chosen.columns and n_confirmed:
        confirmed = chosen[statuses == "confirmed"].copy()
        if not confirmed.empty:
            top_idx = pd.to_numeric(confirmed["posterior_shift"], errors="coerce").idxmax()
            top_name = str(confirmed.loc[top_idx, "attribute_name"] if "attribute_name" in confirmed.columns else confirmed.loc[top_idx, "secret"])
            top_shift = _safe_float(confirmed.loc[top_idx, "posterior_shift"], 0.0)
            return lead + f" Largest confirmed posterior shift: `{top_name}` ({top_shift:+.3f})."
    return lead


# ── report rendering ──────────────────────────────────────────────────

def _render_target_report(
    item: dict,
    post_df: pd.DataFrame,
    image_lookup: dict[str, Path],
    report_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    lines: list[str] = []
    image_id = str(item.get("image_id", ""))
    auditable = bool(item.get("auditability", {}).get("auditable", False))

    lines.append(f"# {image_id}")
    lines.append("")
    lines.append(f"- Auditable: {auditable}")
    if item.get("target_split"):
        lines.append(f"- Target split: {item['target_split']}")
    if item.get("discovery_mode"):
        lines.append(f"- Discovery mode: {item['discovery_mode']}")
    lines.append("")
    lines.append("## Target image")
    lines.extend(_image_block(image_id, image_lookup, report_path, args.image_width, "Target"))

    auditability = item.get("auditability", {}) or {}
    if auditability:
        lines.append(
            "- Auditability: "
            f"kth_neighbor_similarity={_format_float(auditability.get('kth_neighbor_similarity'))}, "
            f"caption_cohesion={_format_float(auditability.get('caption_cohesion', auditability.get('concentration')))}, "
            f"stability={_format_float(auditability.get('stability'))}"
        )

    llm_summary = str(item.get("llm_summary", "")).strip()
    if llm_summary:
        lines.append(f"- LLM summary: {llm_summary}")

    shared_attrs = item.get("shared_attributes", []) or []
    if shared_attrs:
        lines.append("- Shared attributes discovered:")
        for a in shared_attrs[:5]:
            if isinstance(a, dict):
                nm = str(a.get("name", "")).strip() or "(unnamed)"
                desc = str(a.get("description", "")).strip()
                spec = a.get("specificity")
                prv = a.get("privacy_relevance")
                evidence = _safe_list(a.get("evidence"))[:5]
                row = f"  - **{nm}**"
                if desc:
                    row += f": {desc}"
                extras = []
                if spec is not None:
                    extras.append(f"specificity={_format_float(spec, 2)}")
                if prv is not None:
                    extras.append(f"privacy={_format_float(prv, 2)}")
                if extras:
                    row += f" ({', '.join(extras)})"
                lines.append(row)
                if evidence:
                    lines.append(f"    - Evidence terms: {', '.join(evidence)}")
            else:
                lines.append(f"  - {a}")

    cand = _safe_list(item.get("candidate_secrets"))
    if cand:
        lines.append(f"- Candidate secrets: {', '.join(cand)}")

    cues = _safe_list(item.get("common_cues"))
    if cues:
        lines.append(f"- Common cues: {', '.join(cues[:12])}")
    common_phrases = _safe_list(item.get("common_phrases"))
    if common_phrases:
        lines.append(f"- Common phrases: {', '.join(common_phrases[:12])}")

    rejected = _safe_list(item.get("rejected_generic_terms"))
    if rejected:
        lines.append(f"- Rejected generic terms: {', '.join(rejected[:12])}")

    nbrs = _safe_list(item.get("neighbor_ids"))
    if nbrs:
        lines.append(f"- Neighbor IDs: {', '.join(nbrs[:15])}")
        lines.append("")
        lines.append("## Neighbor images")
        for nb in nbrs[: args.max_neighbor_images]:
            lines.append(f"- Neighbor `{nb}`")
            lines.extend(_image_block(nb, image_lookup, report_path, args.image_width, "Neighbor"))
    else:
        lines.append("- Neighbor IDs: none")

    for i, s in enumerate(item.get("evidence_snippets", [])[:3], start=1):
        lines.append(f"  - Evidence snippet {i}: {str(s)[:500]}")

    # ── Stage 3 confirmation results ──
    img_post = pd.DataFrame()
    if not post_df.empty:
        key_col = "target_image_id" if "target_image_id" in post_df.columns else "image_id"
        img_post = post_df[post_df[key_col] == image_id].copy()

    if not img_post.empty:
        stage1_names = set(_stage1_attribute_names(item))
        filt = img_post[img_post["secret"].astype(str).isin(stage1_names)] if stage1_names and "secret" in img_post.columns else pd.DataFrame()
        chosen = _pick_top_posteriors(filt if not filt.empty else img_post, args.model, args.top_posteriors)
        if args.collapse_identical_signatures:
            chosen = _dedupe_identical_signature_rows(chosen)
        method_col_name = _method_col(chosen) if not chosen.empty else None

        if not chosen.empty:
            lines.append("")
            lines.append("## Confirmation results")
            lines.extend(_confirmation_rule_lines(chosen))
            lines.append("")

            # ── per-target overall summary ──
            target_summary = _interpret_target_summary(chosen)
            if target_summary:
                lines.append("")
                lines.append(target_summary)
                lines.append("")

            duplicate_alias_notes = _alias_group_notes(chosen)
            if duplicate_alias_notes:
                lines.append("- Note: some discovered concepts operationalized to identical binary label supports, so their confirmation evidence may be identical.")
                for note in duplicate_alias_notes[:8]:
                    lines.append(f"  - {note}")

            for _, row in chosen.iterrows():
                secret = str(row.get("attribute_name", row.get("secret", "")))
                method = str(row.get(method_col_name, "")) if method_col_name else ""
                status = str(row.get("confirmation_status", row.get("status", "inconclusive")))
                baseline = _format_float(row.get("baseline_posterior", row.get("p_baseline_1")))
                conditional = _format_float(row.get("conditional_posterior", row.get("p_secret_given_baseline_1")))
                shift = _format_float(row.get("posterior_shift", row.get("delta_p")))
                excess_kl = _format_float(row.get("excess_kl_nats", row.get("excess_local_kl_nats")))
                pred = row.get("pred_label")
                y_true = row.get("y_true")
                baseline_mode = str(row.get("baseline_mode", "NONE"))
                baseline_status = str(row.get("baseline_status", ""))
                method_txt = f" via `{method}`" if method else ""

                headline_parts = [
                    f"status={status}",
                    f"q(S|Y)={baseline}",
                    f"q(S|Z,Y)={conditional}",
                    f"shift={shift}",
                    f"excess_KL={excess_kl} nats",
                ]
                if not pd.isna(pred):
                    headline_parts.append(f"pred={pred}")
                if not pd.isna(y_true):
                    headline_parts.append(f"y_true={y_true}")
                lines.append(f"- **{secret}**{method_txt}: " + ", ".join(headline_parts))
                if baseline_mode and baseline_mode != "NONE":
                    baseline_parts: list[str] = []
                    _append_metric(baseline_parts, "q(S|Y)", row.get("baseline_posterior", row.get("p_baseline_1")))
                    _append_metric(baseline_parts, "q(S|Z,Y)", row.get("conditional_posterior", row.get("p_secret_given_baseline_1")))
                    _append_metric(baseline_parts, "excess_KL", row.get("excess_kl_nats", row.get("excess_local_kl_nats")), suffix=" nats")
                    if baseline_status:
                        baseline_parts.append(f"baseline_status={baseline_status}")
                    if baseline_parts:
                        lines.append(f"  - Task-relative evidence vs {baseline_mode}: " + ", ".join(baseline_parts))

                    decomp_parts: list[str] = []
                    _append_metric(decomp_parts, "task_KL", row.get("task_kl_nats"), suffix=" nats")
                    _append_metric(decomp_parts, "excess_KL", row.get("excess_kl_nats", row.get("excess_local_kl_nats")), suffix=" nats")
                    _append_metric(decomp_parts, "residual", row.get("decomposition_residual_nats"), suffix=" nats")
                    if decomp_parts:
                        lines.append("  - Supporting decomposition: " + ", ".join(decomp_parts))

                    derived_parts: list[str] = []
                    _append_metric(derived_parts, "excess_lift", row.get("excess_lift"), ndigits=3, suffix="x")
                    _append_metric(derived_parts, "posterior_shift", row.get("posterior_shift", row.get("delta_p")))
                    _append_metric(derived_parts, "support_score", row.get("support_score"), ndigits=3)
                    if derived_parts:
                        lines.append("  - Supporting metrics: " + ", ".join(derived_parts))

                failures = _safe_json_list(row.get("support_failures_json"))
                warnings = _safe_json_list(row.get("support_warnings_json"))
                if failures:
                    lines.append("  - Blocking reliability flags: " + ", ".join(failures))
                if warnings:
                    lines.append("  - Reliability warnings: " + ", ".join(warnings))
                interpretation = _interpret_secret_row(row)
                if interpretation:
                    lines.append(f"  - **Bottom line**: {interpretation}")
        else:
            lines.append("")
            lines.append("## Confirmation results")
            lines.append("- No confirmation results available after filtering.")
    else:
        lines.append("")
        lines.append("## Confirmation results")
        lines.append("- Not available for this target.")

    if item.get("llm_error"):
        lines.append(f"- LLM error/fallback info: {item['llm_error']}")

    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument(
        "--include-non-auditable",
        action="store_true",
        help="Include targets with auditability.auditable=False. By default only auditable targets are reported.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Optional posterior method/model filter, e.g. weighted_knn, logreg or mlp.",
    )
    ap.add_argument(
        "--top-posteriors",
        type=int,
        default=5,
        help="Maximum number of per-target confirmation rows to show per target.",
    )
    ap.add_argument(
        "--collapse-identical-signatures",
        action="store_true",
        help="Collapse attributes with identical operational label signatures into a single displayed row. By default all screened attributes are shown.",
    )
    ap.add_argument(
        "--max-neighbor-images",
        type=int,
        default=6,
        help="Maximum number of neighbor images to render per target.",
    )
    ap.add_argument(
        "--image-width",
        type=int,
        default=260,
        help="Rendered width in pixels for target and neighbor images.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for per-target markdown files. Defaults to paths.target_report_dir or a folder derived from paths.target_report_md.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    stage1_jsonl = resolve(cfg["paths"]["stage1_jsonl"], project_root)
    out_dir = _resolve_output_dir(cfg, project_root, args.output_dir)
    post_csv = _get_path(cfg, project_root, "target_confirmations_csv", "data/reports/target_confirmations.csv")
    if not post_csv.exists():
        post_csv = _get_path(cfg, project_root, "target_posteriors_csv", "data/reports/target_posteriors.csv")
    manifest_csv = _get_path(cfg, project_root, "manifest_csv", "data/manifests/manifest.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_records = _load_stage1(stage1_jsonl)
    post_df = _load_posteriors(post_csv)
    manifest_df = _load_manifest(manifest_csv, cfg, project_root)
    image_lookup = _build_image_lookup(manifest_df, cfg, project_root)

    count = 0
    skipped_non_auditable = 0
    index_lines = ["# Per-target Audit Reports", ""]
    index_lines.append("This folder contains one Markdown report per target.")
    index_lines.append("")
    if not post_df.empty:
        index_lines.append(f"Confirmation source: `{post_csv}`")
        if "baseline_mode" in post_df.columns:
            modes = sorted(post_df["baseline_mode"].dropna().astype(str).unique().tolist())
            index_lines.append(f"Stage 3 baseline modes present: {', '.join(f'`{m}`' for m in modes)}")
    if not manifest_df.empty:
        index_lines.append(f"Manifest source: `{manifest_csv}`")
    report_cfg = cfg.get("report", {}) or {}
    image_root_note = report_cfg.get("markdown_image_root") or report_cfg.get("image_root")
    if image_root_note:
        index_lines.append(f"Markdown image root override: `{image_root_note}`")
    index_lines.append(f"Resolved image paths available for `{len(image_lookup)}` image IDs.")
    index_lines.append("")

    for item in stage1_records:
        auditable = bool(item.get("auditability", {}).get("auditable", False))
        if not args.include_non_auditable and not auditable:
            skipped_non_auditable += 1
            continue
        if count >= args.limit:
            break

        image_id = str(item.get("image_id", "")).strip() or f"target_{count:04d}"
        report_path = out_dir / _slugify_filename(image_id)
        report_lines = _render_target_report(item, post_df, image_lookup, report_path, args)
        ensure_parent(report_path)
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        index_lines.append(f"- [{image_id}]({report_path.name})")
        count += 1

    if count == 0 and not args.include_non_auditable:
        index_lines.append("No auditable targets were found in the Stage 1 discovery file.")
    index_lines.append("")
    index_lines.append(f"_Reported targets: {count}_")
    if not args.include_non_auditable:
        index_lines.append(f"_Skipped non-auditable targets: {skipped_non_auditable}_")

    index_path = out_dir / "index.md"
    ensure_parent(index_path)
    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    print(f"Saved per-target reports to {out_dir}")
    print(f"Saved report index to {index_path}")
    if not args.include_non_auditable:
        print(f"[INFO] Reported auditable targets: {count}")
        print(f"[INFO] Skipped non-auditable targets: {skipped_non_auditable}")
    if post_df.empty:
        print(f"[INFO] No target confirmation CSV found at {post_csv}; reports contain Stage 1 outputs only.")
    if manifest_df.empty:
        print(f"[INFO] No manifest CSV found at {manifest_csv}; reports cannot render images.")
    elif not image_lookup:
        print(f"[INFO] Manifest loaded from {manifest_csv}, but no image paths could be resolved.")


if __name__ == "__main__":
    main()
