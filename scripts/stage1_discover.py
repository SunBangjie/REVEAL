"""Stage 1 — offline LLM discovery via exported prompt files.

This variant avoids direct API calls. It runs in two steps:

1) Export prompts and cache retrieval evidence:
   python scripts/stage1_discover_offline.py \
       --config configs/mvp.yaml \
       --action export_prompts \
       --work-dir outputs/stage1_offline

2) After manually getting model responses and saving them to files, ingest them:
   python scripts/stage1_discover_offline.py \
       --config configs/mvp.yaml \
       --action ingest_responses \
       --work-dir outputs/stage1_offline

Expected layout under --work-dir after export:
  prompts/
    system_prompt.txt
    prompt_manifest.jsonl
    000001__<imageid>.prompt.txt
    000001__<imageid>.prompt.json
  cache/
    stage1_records_cache.jsonl
    export_summary.json
  responses/               # user-created later
    000001__<imageid>.response.txt   (recommended)
    or .json / .md

The ingest step reads the cache plus any matching response files and writes the
final Stage 1 JSONL to cfg["paths"]["stage1_jsonl"]. Missing or invalid
responses fall back to heuristic-only outputs unless disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.config import load_config, normalize_manifest_df, resolve, resolve_optional
from audit.retrieval import CosineIndex
from audit.stage1_semantics import (
    build_stage1_chat_prompt,
    build_stage1_system_prompt,
    build_stage1_user_prompt,
    extract_json_object,
    narrow_generic_blacklist,
    sanitize_stage1_result,
    stage1_response_schema,
)
from audit.task_data import find_pred_label_column, load_class_name_map
from audit.text_features import caption_cohesion as _caption_cohesion, top_phrases as _top_phrases
from audit.utils import ensure_parent, jaccard


def _candidate_hypotheses(phrases: list[str], top_n: int = 3) -> list[str]:
    out: list[str] = []
    for phrase in phrases:
        if " " in phrase:
            out.append(f"presence of '{phrase}' context")
        else:
            out.append(f"content related to '{phrase}'")
        if len(out) >= top_n:
            break
    return out


def _global_coverage_threshold(kth_sims: np.ndarray, quantile: float) -> float:
    if kth_sims.size == 0:
        return -1.0
    quantile = float(np.clip(quantile, 0.0, 1.0))
    return float(np.quantile(kth_sims, 1.0 - quantile))


def _truncate_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _sanitize_offline_llm_result(raw: dict[str, Any], llm_cfg: dict[str, Any]) -> dict[str, Any]:
    generic_terms = narrow_generic_blacklist(llm_cfg.get("generic_blacklist") or [])
    max_attributes = int(llm_cfg.get("max_attributes", 5))
    min_specificity = float(llm_cfg.get("min_specificity", 0.55))
    min_relevance = float(llm_cfg.get("min_relevance", llm_cfg.get("min_privacy_relevance", 0.45)))
    max_secrets = int(llm_cfg.get("max_excess_secrets", max_attributes))
    max_task_relevance = float(llm_cfg.get("max_task_relevance", 0.75))
    min_privacy_relevance = float(llm_cfg.get("min_privacy_relevance", min_relevance))
    return sanitize_stage1_result(
        raw,
        min_specificity=min_specificity,
        min_relevance=min_relevance,
        min_privacy_relevance=min_privacy_relevance,
        generic_blacklist=generic_terms,
        max_task_relevance=max_task_relevance,
        max_shared_attributes=max_attributes,
        max_excess_secrets=max_secrets,
    )


# ── Target split resolution ──────────────────────────────────────────

def _resolve_target_split(cfg: dict[str, Any], args_target_split: str | None) -> str:
    if args_target_split:
        return str(args_target_split).strip()
    discovery_cfg = cfg.get("discovery") or {}
    audit_cfg = cfg.get("audit") or {}
    return str(discovery_cfg.get("target_split") or audit_cfg.get("target_split") or "audited").strip()


def _choose_existing_target_split(df: pd.DataFrame, preferred: str) -> tuple[str, bool]:
    available = [str(x) for x in df["split"].dropna().astype(str).unique().tolist()]
    if preferred in available:
        return preferred, False
    for name in ["audited", "discovery", "verification", "rubric", "corpus"]:
        if name in available:
            return name, True
    return (available[0], True) if available else (preferred, True)


def _coerce_label_index(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else int(value)

    text = str(value).strip()
    if not text:
        return None

    try:
        return int(text)
    except Exception:
        pass

    try:
        return int(float(text))
    except Exception:
        pass

    match = re.match(r"^\s*(-?\d+)\b", text)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _load_predicted_labels(cfg: dict[str, Any], project_root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    paths = cfg.get("paths") or {}

    task_csv = paths.get("task_outputs_csv")
    if task_csv:
        path = resolve(task_csv, project_root)
        if path.exists():
            df = pd.read_csv(path)
            if "image_id" in df.columns:
                label_col = find_pred_label_column(df.columns)
                if label_col:
                    for _, r in df[["image_id", label_col]].dropna().iterrows():
                        idx = _coerce_label_index(r[label_col])
                        if idx is None:
                            continue
                        out[str(r["image_id"])] = int(idx)
                    if out:
                        print(f"[INFO] Loaded predicted labels from task_outputs_csv ({len(out)} rows, column='{label_col}').")
                        return out

    emb_path = paths.get("embeddings_npz")
    if emb_path:
        path = resolve(emb_path, project_root)
        if path.exists():
            npz = np.load(path, allow_pickle=True)
            if "image_ids" in npz.files and "pred_labels" in npz.files:
                ids = npz["image_ids"].astype(str)
                labels = npz["pred_labels"].astype(int)
                out = {str(i): int(lbl) for i, lbl in zip(ids, labels)}
                if out:
                    print(f"[INFO] Loaded predicted labels from embeddings_npz ({len(out)} rows).")
    return out


# ── Offline file helpers ─────────────────────────────────────────────

def _safe_stem(image_id: str, max_len: int = 80) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(image_id)).strip("._")
    if not safe:
        safe = "image"
    if len(safe) > max_len:
        digest = hashlib.md5(str(image_id).encode("utf-8")).hexdigest()[:8]
        safe = safe[: max_len - 9] + "_" + digest
    return safe


def _recommended_base_filename(idx: int, image_id: str) -> str:
    return f"{idx:06d}__{_safe_stem(image_id)}"


def _response_candidates(base_name: str, responses_dir: Path) -> list[Path]:
    names = [
        f"{base_name}.response.txt",
        f"{base_name}.response.json",
        f"{base_name}.response.md",
        f"{base_name}.txt",
        f"{base_name}.json",
        f"{base_name}.md",
    ]
    return [responses_dir / n for n in names]


def _read_first_existing(paths: list[Path]) -> tuple[Path | None, str | None]:
    for p in paths:
        if p.exists() and p.is_file():
            return p, p.read_text(encoding="utf-8")
    return None, None


def _build_work_dirs(work_dir: Path) -> dict[str, Path]:
    return {
        "root": work_dir,
        "prompts": work_dir / "prompts",
        "responses": work_dir / "responses",
        "cache": work_dir / "cache",
        "system_prompt": work_dir / "prompts" / "system_prompt.txt",
        "manifest": work_dir / "prompts" / "prompt_manifest.jsonl",
        "cache_jsonl": work_dir / "cache" / "stage1_records_cache.jsonl",
        "summary_json": work_dir / "cache" / "export_summary.json",
        "readme": work_dir / "README.txt",
    }


def _default_work_dir(cfg: dict[str, Any], project_root: Path) -> Path:
    discovery_cfg = cfg.get("discovery") or {}
    offline_cfg = discovery_cfg.get("offline_llm") or {}
    if offline_cfg.get("work_dir"):
        return resolve(offline_cfg["work_dir"], project_root)
    stage1_path = resolve(cfg["paths"]["stage1_jsonl"], project_root)
    return stage1_path.parent / "stage1_offline"


# ── Core pass1 computation ───────────────────────────────────────────

def _compute_stage1_records(cfg: dict[str, Any], project_root: Path, target_split_arg: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = pd.read_csv(resolve(cfg["paths"]["manifest_csv"], project_root))
    manifest = normalize_manifest_df(manifest, cfg, project_root)
    splits = pd.read_csv(resolve(cfg["paths"]["splits_csv"], project_root))
    emb = np.load(resolve(cfg["paths"]["embeddings_npz"], project_root), allow_pickle=True)
    index_npz = np.load(resolve(cfg["paths"]["index_npz"], project_root), allow_pickle=True)

    df = manifest.merge(splits, on="image_id", how="inner")
    ids = emb["image_ids"].astype(str)
    embs = emb["embeddings"].astype(np.float32)
    id_to_pos = {k: i for i, k in enumerate(ids)}

    corpus_ids = index_npz["corpus_ids"].astype(str).tolist()
    corpus_embs = index_npz["corpus_embeddings"].astype(np.float32)
    index = CosineIndex(corpus_embs, corpus_ids)

    requested_target_split = _resolve_target_split(cfg, target_split_arg)
    target_split, used_fallback = _choose_existing_target_split(df, requested_target_split)
    if used_fallback:
        print(f"[WARN] Requested Stage 1 target split '{requested_target_split}' not found; using '{target_split}'.")
    else:
        print(f"[INFO] Using Stage 1 target split: {target_split}")

    target_df = df[df["split"] == target_split].reset_index(drop=True)
    if target_df.empty:
        print(f"[WARN] No rows found for Stage 1 target split '{target_split}'.")

    top_k = int(cfg["retrieval"]["top_k"])
    top_2k = max(top_k * 2, top_k)
    audit_cfg = cfg["retrieval"]["auditability"]
    coverage_quantile = float(audit_cfg["max_distance_quantile"])
    min_cohesion = float(audit_cfg["min_tag_concentration"])
    min_stability = float(audit_cfg["min_jaccard_stability"])

    pred_idx_map = _load_predicted_labels(cfg, project_root)
    class_name_map = load_class_name_map(
        resolve_optional((cfg.get("task_labels") or {}).get("class_name_json"), project_root),
        warn_missing=True,
    )
    image_id_to_label_desc: dict[str, str] = {}
    for img_id, idx in pred_idx_map.items():
        name = class_name_map.get(str(idx))
        image_id_to_label_desc[img_id] = f"{idx}: {name}" if name else str(idx)
    if image_id_to_label_desc:
        print("[INFO] Loaded predicted labels for prompt export.")

    records: list[dict[str, Any]] = []
    kth_sims: list[float] = []

    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Stage1 pass1"):
        image_id = str(row["image_id"])
        if image_id not in id_to_pos:
            continue
        q = embs[id_to_pos[image_id]][None, :]
        sims_k, nbrs_k = index.query(q, top_k=top_k)
        sims_2k, nbrs_2k = index.query(q, top_k=top_2k)
        nbr_ids = nbrs_k[0]
        nbr_df = df[df["image_id"].astype(str).isin(nbr_ids)].copy()
        nbr_df["_rank"] = nbr_df["image_id"].astype(str).map({nid: i for i, nid in enumerate(nbr_ids)})
        nbr_df = nbr_df.sort_values("_rank")
        text_col = "text_full" if "text_full" in nbr_df.columns else ("text_base" if "text_base" in nbr_df.columns else None)
        nbr_texts = nbr_df[text_col].fillna("").astype(str).tolist() if text_col else []

        top_sim = np.clip(sims_k[0], -1.0, 1.0)
        kth_sim = float(top_sim[-1]) if len(top_sim) else -1.0
        kth_sims.append(kth_sim)

        phrases = _top_phrases(nbr_texts, top_n=8)
        cohesion = _caption_cohesion(nbr_texts)
        stability = jaccard(nbrs_k[0], nbrs_2k[0][:top_k])

        records.append({
            "image_id": image_id,
            "target_split": target_split,
            "neighbor_ids": list(map(str, nbr_ids)),
            "neighbor_similarities": top_sim.tolist(),
            "neighbor_texts": nbr_texts,
            "common_phrases": phrases,
            "common_cues": phrases[:5],
            "candidate_secrets": _candidate_hypotheses(phrases, top_n=3),
            "auditability": {
                "caption_cohesion": float(cohesion),
                "concentration": float(cohesion),
                "stability": float(stability),
                "kth_neighbor_similarity": kth_sim,
            },
            "predicted_label": image_id_to_label_desc.get(image_id),
        })

    kth_sims_arr = np.asarray(kth_sims, dtype=np.float32)
    sim_threshold = _global_coverage_threshold(kth_sims_arr, coverage_quantile)

    for rec in records:
        cohesion = float(rec["auditability"]["caption_cohesion"])
        stability = float(rec["auditability"]["stability"])
        kth_sim = float(rec["auditability"]["kth_neighbor_similarity"])
        coverage_ok = kth_sim >= sim_threshold
        rec["auditability"]["coverage_similarity_threshold"] = float(sim_threshold)
        rec["auditability"]["coverage_ok"] = bool(coverage_ok)
        rec["auditability"]["auditable"] = bool(
            coverage_ok and cohesion >= min_cohesion and stability >= min_stability
        )

    meta = {
        "target_split": target_split,
        "sim_threshold": float(sim_threshold),
        "min_cohesion": min_cohesion,
        "min_stability": min_stability,
        "top_k": top_k,
        "num_records": len(records),
    }
    return records, meta


# ── Export / ingest ──────────────────────────────────────────────────

def export_prompts(cfg: dict[str, Any], project_root: Path, target_split_arg: str | None, work_dir: Path, limit: int | None = None) -> None:
    discovery_cfg = cfg.get("discovery") or {}
    llm_cfg = discovery_cfg.get("llm") or {}
    offline_cfg = discovery_cfg.get("offline_llm") or {}
    prompt_cfg = {**llm_cfg, **offline_cfg}
    only_auditable = bool(prompt_cfg.get("only_auditable", True))
    max_neighbor_texts = int(prompt_cfg.get("max_neighbor_texts", 8))
    max_chars_per_text = int(prompt_cfg.get("max_chars_per_text", 600))
    max_attributes = int(prompt_cfg.get("max_attributes", 10))
    max_excess_secrets = int(prompt_cfg.get("max_excess_secrets", max_attributes))

    dirs = _build_work_dirs(work_dir)
    for key in ["root", "prompts", "responses", "cache"]:
        dirs[key].mkdir(parents=True, exist_ok=True)

    records, meta = _compute_stage1_records(cfg, project_root, target_split_arg)
    if limit is not None:
        records = records[:limit]
        meta["num_records"] = len(records)
        meta["limit"] = limit

    generic_terms = sorted(narrow_generic_blacklist(llm_cfg.get("generic_blacklist") or []))
    system_prompt = build_stage1_system_prompt(generic_terms)
    dirs["system_prompt"].write_text(system_prompt + "\n", encoding="utf-8")

    exported = 0
    with open(dirs["cache_jsonl"], "w", encoding="utf-8") as f_cache, open(dirs["manifest"], "w", encoding="utf-8") as f_manifest:
        for idx, rec in enumerate(tqdm(records, desc="Export prompts"), start=1):
            f_cache.write(json.dumps(rec) + "\n")
            auditable = bool(rec["auditability"].get("auditable", False))
            should_export = auditable or not only_auditable
            base_name = _recommended_base_filename(idx, rec["image_id"])
            prompt_txt_rel = f"prompts/{base_name}.prompt.txt"
            prompt_json_rel = f"prompts/{base_name}.prompt.json"
            response_rel = f"responses/{base_name}.response.txt"

            manifest_row = {
                "index": idx,
                "image_id": rec["image_id"],
                "target_split": rec["target_split"],
                "auditable": auditable,
                "should_export_prompt": should_export,
                "prompt_txt": prompt_txt_rel,
                "prompt_json": prompt_json_rel,
                "recommended_response": response_rel,
                "predicted_label": rec.get("predicted_label"),
                "common_phrases": rec.get("common_phrases", []),
                "candidate_secrets": rec.get("candidate_secrets", []),
            }
            f_manifest.write(json.dumps(manifest_row) + "\n")

            if not should_export:
                continue

            trimmed = [_truncate_text(t, max_chars_per_text) for t in rec["neighbor_texts"][:max_neighbor_texts]]
            user_prompt = build_stage1_user_prompt(
                rec["image_id"],
                rec.get("predicted_label"),
                trimmed,
                rec["common_phrases"][:8],
                meta["top_k"],
                max_attributes=max_attributes,
                max_excess_secrets=max_excess_secrets,
                include_excess_secrets=True,
            )
            combo = build_stage1_chat_prompt(
                rec["image_id"],
                rec.get("predicted_label"),
                trimmed,
                rec["common_phrases"][:8],
                meta["top_k"],
                generic_terms=generic_terms,
                max_attributes=max_attributes,
                max_excess_secrets=max_excess_secrets,
                include_excess_secrets=True,
            )
            prompt_payload = {
                "image_id": rec["image_id"],
                "target_split": rec["target_split"],
                "auditable": auditable,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "chat_prompt": combo,
                "expected_response_schema": stage1_response_schema(include_excess_secrets=True),
                "recommended_response_path": response_rel,
            }
            (work_dir / prompt_txt_rel).write_text(combo + "\n", encoding="utf-8")
            with open(work_dir / prompt_json_rel, "w", encoding="utf-8") as f_prompt_json:
                json.dump(prompt_payload, f_prompt_json, ensure_ascii=False, indent=2)
            exported += 1

    readme = f"""Offline Stage 1 prompt export completed.

Work directory: {work_dir}

Files:
- {dirs['system_prompt'].name}: shared system instruction.
- prompts/prompt_manifest.jsonl: one line per target with image_id and recommended files.
- prompts/*.prompt.txt: paste this full text into Chat.
- responses/: save the model output here, preferably as *.response.txt with the recommended filename.
- cache/stage1_records_cache.jsonl: cached retrieval evidence used later during ingest.

Next step:
1. For each exported prompt file, send its content to Chat.
2. Save Chat's JSON-only reply into the matching file under responses/.
3. Run:
   python scripts/stage1_discover_offline.py --config <CFG> --action ingest_responses --work-dir {work_dir}
"""
    dirs["readme"].write_text(readme, encoding="utf-8")

    summary = {
        **meta,
        "work_dir": str(work_dir),
        "num_prompt_files": exported,
        "only_auditable": only_auditable,
    }
    with open(dirs["summary_json"], "w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, indent=2)

    print(f"[OK] Exported {exported} prompt files to {dirs['prompts']}")
    print(f"[OK] Cached pass1 records at {dirs['cache_jsonl']}")
    print(f"[OK] Instructions written to {dirs['readme']}")


def ingest_responses(
    cfg: dict[str, Any],
    project_root: Path,
    work_dir: Path,
    responses_dir: Path | None = None,
    disable_fallback: bool = False,
) -> None:
    discovery_cfg = cfg.get("discovery") or {}
    llm_cfg = (discovery_cfg.get("offline_llm") or discovery_cfg.get("llm") or {})
    dirs = _build_work_dirs(work_dir)
    responses_dir = responses_dir or dirs["responses"]
    cache_jsonl = dirs["cache_jsonl"]
    manifest_path = dirs["manifest"]
    if not cache_jsonl.exists():
        raise FileNotFoundError(f"Cached records not found: {cache_jsonl}. Run export_prompts first.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prompt manifest not found: {manifest_path}. Run export_prompts first.")

    manifests: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifests.append(json.loads(line))
    manifest_by_image = {str(m["image_id"]): m for m in manifests}

    records: list[dict[str, Any]] = []
    with open(cache_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    out_path = resolve(cfg["paths"]["stage1_jsonl"], project_root)
    ensure_parent(out_path)
    print(
        f"[INFO] Ingest config source: {'discovery.offline_llm' if discovery_cfg.get('offline_llm') else 'discovery.llm'}"
    )
    print(
        "[INFO] Offline ingest thresholds: "
        f"min_specificity={float(llm_cfg.get('min_specificity', 0.55))}, "
        f"min_relevance={float(llm_cfg.get('min_relevance', llm_cfg.get('min_privacy_relevance', 0.45)))}, "
        f"min_privacy_relevance={float(llm_cfg.get('min_privacy_relevance', llm_cfg.get('min_relevance', 0.45)))}, "
        f"max_task_relevance={float(llm_cfg.get('max_task_relevance', 0.75))}, "
        f"max_attributes={int(llm_cfg.get('max_attributes', 5))}, "
        f"max_excess_secrets={int(llm_cfg.get('max_excess_secrets', llm_cfg.get('max_attributes', 5)))}"
    )

    parsed_ok = 0
    parsed_fail = 0
    missing = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, rec in enumerate(tqdm(records, desc="Ingest responses"), start=1):
            cohesion = float(rec["auditability"]["caption_cohesion"])
            stability = float(rec["auditability"]["stability"])
            kth_sim = float(rec["auditability"]["kth_neighbor_similarity"])
            coverage_ok = bool(rec["auditability"].get("coverage_ok", False))
            auditable = bool(rec["auditability"].get("auditable", False))

            result: dict[str, Any] = {
                "image_id": rec["image_id"],
                "target_split": rec["target_split"],
                "predicted_label": rec.get("predicted_label"),
                "neighbor_ids": rec["neighbor_ids"],
                "neighbor_similarities": rec["neighbor_similarities"],
                "candidate_secrets": rec["candidate_secrets"],
                "common_cues": rec["common_cues"],
                "common_phrases": rec["common_phrases"],
                "evidence_snippets": rec["neighbor_texts"][:3],
                "discovery_mode": "offline_llm_integrated_excess",
                "llm_summary": "",
                "shared_attributes": [],
                "predicted_label_covers": "",
                "excess_secrets": [],
                "excess_task_relevance_threshold": float(llm_cfg.get("max_task_relevance", 0.75)),
                "rejected_generic_terms": [],
                "llm_error": "",
                "auditability": {
                    "caption_cohesion": cohesion,
                    "concentration": cohesion,
                    "stability": stability,
                    "kth_neighbor_similarity": kth_sim,
                    "coverage_similarity_threshold": float(rec["auditability"].get("coverage_similarity_threshold", -1.0)),
                    "coverage_ok": coverage_ok,
                    "auditable": auditable,
                },
            }

            m = manifest_by_image.get(str(rec["image_id"]))
            base_name = _recommended_base_filename(idx, rec["image_id"]) if m is None else Path(str(m["prompt_txt"])).name.replace(".prompt.txt", "")
            response_path, response_text = _read_first_existing(_response_candidates(base_name, responses_dir))

            if response_text is None:
                missing += 1
                result["llm_error"] = f"Missing response file for {base_name} under {responses_dir}"
                if disable_fallback:
                    raise FileNotFoundError(result["llm_error"])
            else:
                try:
                    raw = extract_json_object(response_text)
                    llm_result = _sanitize_offline_llm_result(raw, llm_cfg)
                    result["llm_summary"] = llm_result.get("summary", "")
                    result["shared_attributes"] = llm_result.get("shared_attributes", [])
                    result["predicted_label_covers"] = llm_result.get("predicted_label_covers", "")
                    result["excess_secrets"] = llm_result.get("excess_secrets", [])
                    result["excess_task_relevance_threshold"] = float(
                        llm_result.get("excess_task_relevance_threshold", llm_cfg.get("max_task_relevance", 0.75))
                    )
                    result["rejected_generic_terms"] = llm_result.get("rejected_generic_terms", [])
                    result["response_file"] = str(response_path)
                    excess_names = [
                        a.get("name", "")
                        for a in result["excess_secrets"]
                        if isinstance(a, dict) and a.get("name")
                    ]
                    shared_names = [
                        a.get("name", "")
                        for a in result["shared_attributes"]
                        if isinstance(a, dict) and a.get("name")
                    ]
                    names = excess_names or shared_names
                    if names:
                        result["candidate_secrets"] = names
                        result["common_cues"] = names
                    parsed_ok += 1
                except Exception as exc:
                    parsed_fail += 1
                    result["llm_error"] = f"Failed to parse/sanitize response: {repr(exc)}"
                    result["response_file"] = str(response_path)
                    if disable_fallback:
                        raise

            f_out.write(json.dumps(result) + "\n")

    try:
        with open(out_path, "r", encoding="utf-8") as f_chk:
            rows = [json.loads(line) for line in f_chk if line.strip()]
        rows_with_shared = sum(bool(r.get("shared_attributes")) for r in rows if isinstance(r, dict))
        rows_with_excess = sum(bool(r.get("excess_secrets")) for r in rows if isinstance(r, dict))
        total_shared = sum(len(r.get("shared_attributes") or []) for r in rows if isinstance(r, dict))
        total_excess = sum(len(r.get("excess_secrets") or []) for r in rows if isinstance(r, dict))
    except Exception:
        rows_with_shared = rows_with_excess = total_shared = total_excess = -1

    print(f"[OK] Saved Stage 1 discovery results to {out_path}")
    print(f"[INFO] Parsed response files successfully: {parsed_ok}")
    print(f"[INFO] Missing response files: {missing}")
    print(f"[INFO] Rows with shared_attributes: {rows_with_shared}; total shared_attributes: {total_shared}")
    print(f"[INFO] Rows with excess_secrets: {rows_with_excess}; total excess_secrets: {total_excess}")
    print(f"[INFO] Invalid response files (fell back to heuristic fields): {parsed_fail}")


# ── Main ──────────────────────────────────────────────────────────────

def _infer_project_root(args_project_root: str | None) -> Path:
    if args_project_root:
        return Path(args_project_root).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "configs").exists() or (cwd / "audit").exists() or (cwd / "scripts").exists():
        return cwd
    return Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target-split", default=None)
    ap.add_argument(
        "--action",
        choices=["export_prompts", "ingest_responses"],
        required=True,
        help="export_prompts: generate chat-ready prompt files; ingest_responses: merge saved responses into final stage1_jsonl",
    )
    ap.add_argument("--project-root", default=None, help="Project root; defaults to current working directory when it looks like the repo root")
    ap.add_argument("--work-dir", default=None, help="Offline working directory for prompts/cache/responses")
    ap.add_argument("--responses-dir", default=None, help="Override response directory for ingest")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of targets to export")
    ap.add_argument(
        "--disable-fallback",
        action="store_true",
        help="During ingest, fail on missing/invalid responses instead of falling back to heuristic-only output",
    )
    args = ap.parse_args()

    project_root = _infer_project_root(args.project_root)
    cfg = load_config(args.config)
    work_dir = resolve(args.work_dir, project_root) if args.work_dir else _default_work_dir(cfg, project_root)

    if args.action == "export_prompts":
        export_prompts(cfg, project_root, args.target_split, work_dir, limit=args.limit)
    elif args.action == "ingest_responses":
        responses_dir = resolve(args.responses_dir, project_root) if args.responses_dir else None
        ingest_responses(cfg, project_root, work_dir, responses_dir=responses_dir, disable_fallback=args.disable_fallback)
    else:
        raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()
