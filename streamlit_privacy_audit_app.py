from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from html import escape as html_escape
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from PIL import Image

from audit.config import (
    infer_project_root_from_config,
    load_config as load_yaml,
    normalize_manifest_df,
    resolve_image_path_from_record,
    resolve_optional as resolve_path,
    semantic_parts_from_record,
)
from audit.modeling import (
    build_cam_from_activations,
    build_embedder,
    build_preprocess,
    get_model_info,
    resolve_runtime_device,
)
from audit.stage1_semantics import (
    build_stage1_chat_prompt,
    extract_json_object,
    narrow_generic_blacklist,
    sanitize_stage1_result,
    snakeify,
)
from audit.task_data import load_class_name_map, load_task_outputs, safe_pred_label
from audit.text_features import caption_cohesion as compute_caption_cohesion, top_phrases
from audit.utils import normalize_rows, safe_float as _safe_float


def choose_prompt_text_column(manifest: pd.DataFrame, fallback: str) -> str:
    if "semantic_text" in manifest.columns:
        return "semantic_text"
    if "text_base" in manifest.columns:
        return "text_base"
    return fallback


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model_and_preprocess(model_name: str, image_size: int, requested_device: str):
    model_info = get_model_info(model_name)
    device = resolve_runtime_device(requested_device, model_info)
    model = build_embedder(model_name).to(device).eval()
    tf = build_preprocess(image_size)
    return model, tf, device, model_info


# -----------------------------------------------------------------------------
# Artifact loading
# -----------------------------------------------------------------------------

@dataclass
class AppArtifacts:
    cfg: dict[str, Any]
    project_root: Path
    manifest: pd.DataFrame
    generated_targets: pd.DataFrame
    corpus_ids: np.ndarray
    corpus_embeddings: np.ndarray
    full_image_ids: np.ndarray
    full_embeddings: np.ndarray
    task_outputs: pd.DataFrame
    task_by_id: pd.DataFrame
    class_name_map: dict[str, str]
    text_column: str
    id_column: str


@st.cache_data(show_spinner=False)
def load_artifacts(config_path_str: str, project_root_str: str | None = None) -> AppArtifacts:
    config_path = Path(config_path_str).expanduser().resolve()
    cfg = load_yaml(config_path)
    project_root = Path(project_root_str).expanduser().resolve() if project_root_str else infer_project_root_from_config(config_path)

    paths = cfg.get("paths") or {}
    manifest = pd.read_csv(resolve_path(paths["manifest_csv"], project_root))
    manifest = normalize_manifest_df(manifest, cfg, project_root)
    generated_targets = load_generated_targets(cfg, project_root)
    emb = np.load(resolve_path(paths["embeddings_npz"], project_root), allow_pickle=True)
    idx = np.load(resolve_path(paths["index_npz"], project_root), allow_pickle=True)

    class_name_map = load_class_name_map(resolve_path(((cfg.get("task_labels") or {}).get("class_name_json")), project_root))
    full_image_ids = emb["image_ids"].astype(str)
    full_embeddings = normalize_rows(emb["embeddings"].astype(np.float32))
    corpus_ids = idx["corpus_ids"].astype(str)
    corpus_embeddings = normalize_rows(idx["corpus_embeddings"].astype(np.float32))
    task_outputs = load_task_outputs(cfg, project_root, emb, full_image_ids)
    task_outputs["image_id"] = task_outputs["image_id"].astype(str)
    task_by_id = task_outputs.drop_duplicates(subset=["image_id"], keep="first").set_index("image_id") if not task_outputs.empty else pd.DataFrame().set_index(pd.Index([], name="image_id"))

    text_column = str(((cfg.get("operationalization") or {}).get("text_column", "text_full")))
    id_column = str(((cfg.get("operationalization") or {}).get("id_column", "image_id")))
    return AppArtifacts(
        cfg=cfg,
        project_root=project_root,
        manifest=manifest,
        generated_targets=generated_targets,
        corpus_ids=corpus_ids,
        corpus_embeddings=corpus_embeddings,
        full_image_ids=full_image_ids,
        full_embeddings=full_embeddings,
        task_outputs=task_outputs,
        task_by_id=task_by_id,
        class_name_map=class_name_map,
        text_column=text_column,
        id_column=id_column,
    )


def load_generated_targets(cfg: dict[str, Any], project_root: Path) -> pd.DataFrame:
    target_cfg = cfg.get("generated_targets") or {}
    manifest_cfg = target_cfg.get("manifest_csv")
    if not manifest_cfg:
        return pd.DataFrame()

    manifest_path = resolve_path(manifest_cfg, project_root)
    if manifest_path is None or not manifest_path.exists():
        return pd.DataFrame()

    image_root = str(target_cfg.get("image_root") or manifest_path.parent)
    temp_cfg = copy.deepcopy(cfg)
    temp_cfg.setdefault("dataset", {})
    temp_cfg.setdefault("report", {})
    temp_cfg["dataset"]["photos_root"] = image_root
    temp_cfg["report"]["markdown_image_root"] = image_root

    df = pd.read_csv(manifest_path)
    df = normalize_manifest_df(df, temp_cfg, project_root)
    sort_cols = [c for c in ["scene_family", "primary_label", "secondary_label", "ternary_label", "background_label", "seed"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def build_stage1_prompt(
    target_image_id: str,
    predicted_label_desc: str | None,
    captions: list[str],
    heuristic_phrases: list[str],
    retrieved_top_k: int,
) -> str:
    return build_stage1_chat_prompt(
        target_image_id=target_image_id,
        predicted_label_desc=predicted_label_desc,
        captions=captions,
        heuristic_phrases=heuristic_phrases,
        retrieved_top_k=retrieved_top_k,
        include_excess_secrets=False,
    )


def make_target_image_id(filename: str, image_bytes: bytes) -> str:
    stem = snakeify(Path(filename or "selected_image").stem or "selected_image")
    digest = hashlib.sha1(image_bytes).hexdigest()[:10]
    return f"selected_{stem}_{digest}"


def build_support_bundle(artifacts: AppArtifacts, label_scope: str) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = artifacts.manifest.copy()
    if artifacts.text_column not in manifest.columns:
        raise KeyError(f"Configured text column '{artifacts.text_column}' is missing from the manifest.")

    if label_scope == "corpus_index_only":
        support_df = manifest.set_index(artifacts.id_column, drop=False).reindex(artifacts.corpus_ids.astype(str)).reset_index(drop=True)
        support_df = support_df[support_df[artifacts.id_column].notna()].copy()
        support_df["image_id"] = support_df[artifacts.id_column].astype(str)
        corpus_pos = {str(image_id): i for i, image_id in enumerate(artifacts.corpus_ids.astype(str))}
        support_ids = support_df["image_id"].tolist()
        support_embs = np.stack([artifacts.corpus_embeddings[corpus_pos[i]] for i in support_ids]) if support_ids else np.empty((0, artifacts.corpus_embeddings.shape[1]), dtype=np.float32)
        return support_df, support_embs

    support_df = manifest.copy()
    support_df["image_id"] = support_df[artifacts.id_column].astype(str)
    full_pos = {str(image_id): i for i, image_id in enumerate(artifacts.full_image_ids.astype(str))}
    keep_mask = support_df["image_id"].isin(full_pos)
    support_df = support_df.loc[keep_mask].reset_index(drop=True)
    support_ids = support_df["image_id"].tolist()
    support_embs = np.stack([artifacts.full_embeddings[full_pos[i]] for i in support_ids]) if support_ids else np.empty((0, artifacts.full_embeddings.shape[1]), dtype=np.float32)
    return support_df, support_embs


def build_stage1_record(
    target_image_id: str,
    predicted_label_desc: str,
    neighbor_ids: list[str],
    neighbor_similarities: list[float],
    neighbor_captions: list[str],
    heuristic_phrases: list[str],
    parsed_result: dict[str, Any],
) -> dict[str, Any]:
    discovered_names: list[str] = []
    for field in ("shared_attributes", "excess_secrets"):
        for item in parsed_result.get(field) or []:
            name = snakeify(str((item or {}).get("name") or ""))
            if name and name not in discovered_names:
                discovered_names.append(name)

    cohesion = compute_caption_cohesion(neighbor_captions)
    kth_similarity = float(neighbor_similarities[-1]) if neighbor_similarities else float("nan")
    auditable = bool(neighbor_ids)

    return {
        "image_id": str(target_image_id),
        "target_split": "audited",
        "predicted_label": predicted_label_desc,
        "neighbor_ids": [str(x) for x in neighbor_ids],
        "neighbor_similarities": [float(x) for x in neighbor_similarities],
        "candidate_secrets": discovered_names[:10],
        "common_cues": (discovered_names[:10] or heuristic_phrases[:10]),
        "common_phrases": heuristic_phrases[:10],
        "evidence_snippets": neighbor_captions[:3],
        "discovery_mode": "streamlit_selected_target",
        "llm_summary": str(parsed_result.get("summary") or "").strip(),
        "shared_attributes": parsed_result.get("shared_attributes") or [],
        "predicted_label_covers": str(parsed_result.get("predicted_label_covers") or "").strip(),
        "excess_secrets": parsed_result.get("excess_secrets") or [],
        "excess_task_relevance_threshold": _safe_float(parsed_result.get("excess_task_relevance_threshold"), 1.0),
        "rejected_generic_terms": parsed_result.get("rejected_generic_terms") or [],
        "auditability": {
            "caption_cohesion": float(cohesion),
            "concentration": float(cohesion),
            "stability": 1.0 if auditable else 0.0,
            "kth_neighbor_similarity": kth_similarity,
            "coverage_similarity_threshold": kth_similarity,
            "coverage_ok": auditable,
            "auditable": auditable,
        },
    }


def build_temp_manifest(
    artifacts: AppArtifacts,
    support_df: pd.DataFrame,
    target_image_id: str,
    target_image_path: Path,
    neighbor_ids: list[str],
    target_record: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    keep_ids = set(support_df["image_id"].astype(str).tolist()) | {str(x) for x in neighbor_ids}
    manifest_subset = artifacts.manifest[artifacts.manifest["image_id"].astype(str).isin(keep_ids)].copy()

    target_row = {col: None for col in artifacts.manifest.columns}
    if target_record:
        for col in target_row:
            if col in target_record:
                target_row[col] = target_record[col]
    if "image_id" in target_row:
        target_row["image_id"] = str(target_image_id)
    if "entry_key" in target_row:
        target_row["entry_key"] = str(target_image_id)
    if artifacts.text_column in target_row:
        existing_text = str(target_row.get(artifacts.text_column) or "").strip()
        target_row[artifacts.text_column] = existing_text
    semantic_text = str(target_row.get("semantic_text") or "").strip()
    prompt_text = str(target_row.get("prompt") or "").strip()
    if "text_base" in target_row and not str(target_row.get("text_base") or "").strip():
        target_row["text_base"] = semantic_text or prompt_text
    if "text_full" in target_row and not str(target_row.get("text_full") or "").strip():
        full_text = " ".join(part for part in [semantic_text, f"Prompt: {prompt_text}" if prompt_text else ""] if part).strip()
        target_row["text_full"] = full_text
    if "image_path" in target_row:
        target_row["image_path"] = str(target_image_path)
    if "image_relpath" in target_row:
        target_row["image_relpath"] = target_image_path.name
    if "image_exists" in target_row:
        target_row["image_exists"] = 1
    if "source_json" in target_row:
        target_row["source_json"] = "streamlit_selected_target"
    if "source_dir" in target_row:
        target_row["source_dir"] = "dataset_generation"

    target_df = pd.DataFrame([target_row])
    return pd.concat([manifest_subset, target_df], ignore_index=True).drop_duplicates(subset=["image_id"], keep="last")


def build_temp_task_outputs(
    artifacts: AppArtifacts,
    support_ids: list[str],
    target_image_id: str,
    target_pred: int,
    target_logits: np.ndarray,
    baseline_mode: str,
) -> pd.DataFrame | None:
    mode = str(baseline_mode).upper()
    if mode == "NONE":
        return None

    keep_cols = ["image_id"]
    if "pred_label" in artifacts.task_outputs.columns:
        keep_cols.append("pred_label")
    logit_cols = [c for c in artifacts.task_outputs.columns if str(c).startswith("logit_")]
    if mode == "O":
        keep_cols.extend(logit_cols)

    if len(keep_cols) == 1 and mode == "Y":
        keep_cols.append("pred_label")

    if not artifacts.task_by_id.empty:
        support_task = artifacts.task_by_id.reindex(support_ids).reset_index()
    else:
        support_task = pd.DataFrame({"image_id": support_ids})

    for col in keep_cols:
        if col not in support_task.columns:
            support_task[col] = np.nan
    support_task = support_task[keep_cols].copy()

    target_row: dict[str, Any] = {"image_id": str(target_image_id), "pred_label": int(target_pred)}
    if mode == "O" and logit_cols:
        for idx, col in enumerate(logit_cols):
            target_row[col] = float(target_logits[idx]) if idx < len(target_logits) else np.nan

    return pd.concat([support_task, pd.DataFrame([target_row])], ignore_index=True)


def run_project_script(project_root: Path, script_relpath: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    script_path = project_root / script_relpath
    cmd = [sys.executable, str(script_path), *args]
    env = os.environ.copy()
    project_root_str = str(project_root)
    pythonpath = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if project_root_str not in pythonpath:
        pythonpath.insert(0, project_root_str)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def extract_markdown_section(markdown_text: str, section_title: str) -> str:
    lines = (markdown_text or "").splitlines()
    target_heading = f"## {section_title}".strip().lower()
    start_idx: int | None = None

    for idx, line in enumerate(lines):
        if line.strip().lower() == target_heading:
            start_idx = idx
            break

    if start_idx is None:
        return ""

    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("## ") and stripped.lower() != target_heading:
            end_idx = idx
            break

    return "\n".join(lines[start_idx:end_idx]).strip()


def build_temp_splits(support_ids: list[str], target_image_id: str, support_split: str = "streamlit_support") -> pd.DataFrame:
    rows = [{"image_id": str(image_id), "split": support_split} for image_id in support_ids]
    rows.append({"image_id": str(target_image_id), "split": "audited"})
    return pd.DataFrame(rows).drop_duplicates(subset=["image_id"], keep="last")


def summarize_generated_labels(labels_df: pd.DataFrame, target_image_id: str) -> tuple[pd.DataFrame, int]:
    if labels_df.empty:
        return pd.DataFrame(columns=["secret", "positive_rows"]), 0

    if {"target_image_id", "secret", "label"}.issubset(labels_df.columns):
        work = labels_df.copy()
        if "image_id" in work.columns:
            work = work[work["image_id"].astype(str) != str(target_image_id)].copy()
        summary = (
            work.groupby("secret", as_index=False)["label"]
            .sum()
            .rename(columns={"label": "positive_rows"})
            .sort_values(["positive_rows", "secret"], ascending=[False, True])
        ) if not work.empty else pd.DataFrame(columns=["secret", "positive_rows"])
        num_secrets = int(work["secret"].nunique()) if not work.empty else 0
        return summary, num_secrets

    id_cols = {"image_id", "target_image_id"}
    secret_cols = [c for c in labels_df.columns if c not in id_cols]
    if not secret_cols:
        return pd.DataFrame(columns=["secret", "positive_rows"]), 0

    rows = []
    for secret in secret_cols:
        vals = pd.to_numeric(labels_df[secret], errors="coerce").fillna(0.0)
        rows.append({"secret": str(secret), "positive_rows": int(vals.sum())})
    summary = pd.DataFrame(rows).sort_values(["positive_rows", "secret"], ascending=[False, True]).reset_index(drop=True)
    return summary, int(len(secret_cols))


def generate_pipeline_report(
    artifacts: AppArtifacts,
    target_image: Image.Image,
    target_name: str,
    target_bytes: bytes,
    target_record: Mapping[str, Any] | None,
    target_emb: np.ndarray,
    target_logits: np.ndarray,
    target_pred: int,
    predicted_label_desc: str,
    neighbor_ids: list[str],
    neighbor_similarities: list[float],
    neighbor_captions: list[str],
    heuristic_phrases: list[str],
    parsed_result: dict[str, Any],
    secret_source: str,
    baseline_mode: str,
    label_scope: str,
) -> dict[str, Any]:
    support_df, support_embs = build_support_bundle(artifacts, label_scope)
    if support_df.empty or support_embs.size == 0:
        raise ValueError("No support rows are available for the selected label scope.")

    support_df = support_df.copy()
    support_df[artifacts.text_column] = support_df[artifacts.text_column].fillna("").astype(str)
    target_image_id = make_target_image_id(target_name, target_bytes)

    stage1_record = build_stage1_record(
        target_image_id=target_image_id,
        predicted_label_desc=predicted_label_desc,
        neighbor_ids=neighbor_ids,
        neighbor_similarities=neighbor_similarities,
        neighbor_captions=neighbor_captions,
        heuristic_phrases=heuristic_phrases,
        parsed_result=parsed_result,
    )

    with tempfile.TemporaryDirectory(prefix="streamlit_privacy_audit_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        target_image_path = tmp_dir / f"{target_image_id}.png"
        target_image.save(target_image_path)

        temp_manifest = build_temp_manifest(
            artifacts,
            support_df,
            target_image_id,
            target_image_path,
            neighbor_ids,
            target_record=target_record,
        )
        temp_embeddings_ids = np.array([*support_df["image_id"].astype(str).tolist(), target_image_id])
        temp_embeddings = np.vstack([support_embs.astype(np.float32), target_emb.astype(np.float32)[None, :]])

        temp_manifest_path = tmp_dir / "manifest.csv"
        temp_embeddings_path = tmp_dir / "embeddings.npz"
        temp_splits_path = tmp_dir / "splits.csv"
        temp_labels_path = tmp_dir / "labels.csv"
        temp_stage1_path = tmp_dir / "stage1_discovery.jsonl"
        temp_confirmations_path = tmp_dir / "target_confirmations.csv"
        temp_summary_path = tmp_dir / "confirmation_summary.csv"
        temp_report_dir = tmp_dir / "target_reports"
        temp_config_path = tmp_dir / "streamlit_config.yaml"
        support_split_name = "streamlit_support"

        temp_manifest.to_csv(temp_manifest_path, index=False)
        np.savez_compressed(temp_embeddings_path, image_ids=temp_embeddings_ids, embeddings=temp_embeddings)
        build_temp_splits(
            support_ids=support_df["image_id"].astype(str).tolist(),
            target_image_id=target_image_id,
            support_split=support_split_name,
        ).to_csv(temp_splits_path, index=False)
        temp_stage1_path.write_text(json.dumps(stage1_record) + "\n", encoding="utf-8")

        task_outputs_df = build_temp_task_outputs(
            artifacts=artifacts,
            support_ids=support_df["image_id"].astype(str).tolist(),
            target_image_id=target_image_id,
            target_pred=target_pred,
            target_logits=target_logits,
            baseline_mode=baseline_mode,
        )
        temp_task_outputs_path: Path | None = None
        if task_outputs_df is not None:
            temp_task_outputs_path = tmp_dir / "task_outputs.csv"
            task_outputs_df.to_csv(temp_task_outputs_path, index=False)

        temp_cfg = copy.deepcopy(artifacts.cfg)
        temp_cfg.setdefault("paths", {})
        temp_cfg["paths"]["manifest_csv"] = str(temp_manifest_path)
        temp_cfg["paths"]["embeddings_npz"] = str(temp_embeddings_path)
        temp_cfg["paths"]["splits_csv"] = str(temp_splits_path)
        temp_cfg["paths"]["labels_csv"] = str(temp_labels_path)
        temp_cfg["paths"]["stage1_jsonl"] = str(temp_stage1_path)
        temp_cfg["paths"]["target_confirmations_csv"] = str(temp_confirmations_path)
        temp_cfg["paths"]["confirmation_summary_csv"] = str(temp_summary_path)
        temp_cfg["paths"]["target_report_dir"] = str(temp_report_dir)
        if temp_task_outputs_path is not None:
            temp_cfg["paths"]["task_outputs_csv"] = str(temp_task_outputs_path)
        else:
            temp_cfg["paths"].pop("task_outputs_csv", None)

        temp_cfg.setdefault("confirmation", {})
        temp_cfg["confirmation"]["target_split"] = "audited"
        temp_cfg["confirmation"]["baseline_mode"] = str(baseline_mode).upper()
        temp_cfg.setdefault("audit", {})
        temp_cfg["audit"]["target_split"] = "audited"
        temp_cfg.setdefault("operationalization", {})
        temp_cfg["operationalization"]["source"] = "stage1"
        temp_cfg["operationalization"]["stage1_path"] = str(temp_stage1_path)
        temp_cfg["operationalization"]["from_stage1_field"] = str(secret_source)
        temp_cfg["operationalization"]["text_column"] = str(artifacts.text_column)
        temp_cfg["operationalization"]["id_column"] = "image_id"
        temp_cfg["operationalization"]["target_id_column"] = "image_id"
        temp_cfg["operationalization"]["max_task_relevance"] = float(parsed_result.get("excess_task_relevance_threshold", temp_cfg["operationalization"].get("max_task_relevance", 1.0)))
        temp_cfg.setdefault("report", {})
        temp_cfg["report"]["markdown_image_root"] = str(target_image_path.parent)

        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(temp_cfg, f, sort_keys=False)

        build_labels_proc = run_project_script(
            artifacts.project_root,
            "scripts/build_labels.py",
            ["--config", str(temp_config_path), "--label-split", support_split_name],
        )
        if build_labels_proc.returncode != 0:
            raise RuntimeError(
                "build_labels.py failed.\n"
                f"STDOUT:\n{build_labels_proc.stdout}\n\nSTDERR:\n{build_labels_proc.stderr}"
            )
        if not temp_labels_path.exists():
            raise RuntimeError("build_labels.py completed but did not write labels.csv.")
        labels_df = pd.read_csv(temp_labels_path)
        label_support_summary, num_generated_secrets = summarize_generated_labels(labels_df, target_image_id)

        stage2_proc = run_project_script(
            artifacts.project_root,
            "scripts/stage2_confirm_excess_leakage.py",
            ["--config", str(temp_config_path), "--target-split", "audited"],
        )
        if stage2_proc.returncode != 0:
            raise RuntimeError(
                "stage2_confirm_excess_leakage.py failed.\n"
                f"STDOUT:\n{stage2_proc.stdout}\n\nSTDERR:\n{stage2_proc.stderr}"
            )
        if not temp_confirmations_path.exists():
            raise RuntimeError("stage2_confirm_excess_leakage.py completed but did not write target_confirmations.csv.")
        if not temp_summary_path.exists():
            raise RuntimeError("stage2_confirm_excess_leakage.py completed but did not write confirmation_summary.csv.")

        report_proc = run_project_script(
            artifacts.project_root,
            "scripts/report_targets.py",
            [
                "--config", str(temp_config_path),
                "--limit", "1",
                "--include-non-auditable",
                "--output-dir", str(temp_report_dir),
                "--top-posteriors", "10",
            ],
        )
        if report_proc.returncode != 0:
            raise RuntimeError(
                "report_targets.py failed.\n"
                f"STDOUT:\n{report_proc.stdout}\n\nSTDERR:\n{report_proc.stderr}"
            )

        report_files = sorted(p for p in temp_report_dir.glob("*.md") if p.name != "index.md")
        if not report_files:
            raise RuntimeError("report_targets.py completed but did not write a target report.")

        report_markdown = report_files[0].read_text(encoding="utf-8")
        posterior_df = pd.read_csv(temp_confirmations_path) if temp_confirmations_path.exists() else pd.DataFrame()
        metrics_df = pd.read_csv(temp_summary_path) if temp_summary_path.exists() else pd.DataFrame()

    return {
        "target_image_id": target_image_id,
        "report_markdown": report_markdown,
        "report_confirmation_markdown": extract_markdown_section(report_markdown, "Confirmation results"),
        "posterior_df": posterior_df,
        "metrics_df": metrics_df,
        "labels_df": labels_df,
        "label_support_summary": label_support_summary,
        "num_generated_secrets": num_generated_secrets,
        "build_labels_stdout": build_labels_proc.stdout,
        "build_labels_stderr": build_labels_proc.stderr,
        "stage2_stdout": stage2_proc.stdout,
        "stage2_stderr": stage2_proc.stderr,
        "report_stdout": report_proc.stdout,
        "report_stderr": report_proc.stderr,
    }


# -----------------------------------------------------------------------------
# Retrieval + metrics
# -----------------------------------------------------------------------------


def weighted_knn_posterior(sims: np.ndarray, y: np.ndarray, tau: float, top_k: int, alpha: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    sims = np.asarray(sims, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(sims) == 0:
        return 0.5, 0.0, np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)
    if top_k > 0 and len(sims) > top_k:
        order = np.argsort(-sims)[:top_k]
        sims = sims[order]
        y = y[order]
    else:
        order = np.arange(len(sims), dtype=np.int64)
    tau = max(float(tau), 1e-6)
    shifted = (sims - sims.max()) / tau
    raw = np.exp(np.clip(shifted, -80, 80))
    weights = raw / raw.sum() if raw.sum() > 0 else np.ones_like(raw) / max(len(raw), 1)
    n_eff = float((weights.sum() ** 2) / max(np.square(weights).sum(), 1e-12))
    weighted_pos = float(np.dot(weights, y))
    posterior = (alpha + n_eff * weighted_pos) / (2.0 * alpha + n_eff)
    return float(posterior), float(n_eff), weights, order


def prior_with_smoothing(y: np.ndarray, alpha: float) -> float:
    y = np.asarray(y, dtype=np.float64)
    return float((alpha + y.sum()) / (2.0 * alpha + len(y))) if len(y) else 0.5


def clamp(p: float) -> float:
    return min(max(float(p), 1e-12), 1.0 - 1e-12)


def local_kl_binary(post_p: float, prior_p: float) -> float:
    p = clamp(post_p)
    q = clamp(prior_p)
    return float(p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)))


def baseline_from_pred_label(target_label: int | None, support_pred_labels: np.ndarray | None, support_y: np.ndarray, alpha: float) -> tuple[float, int, np.ndarray]:
    if target_label is None or support_pred_labels is None or len(support_pred_labels) != len(support_y):
        return prior_with_smoothing(support_y, alpha), 0, np.zeros(len(support_y), dtype=bool)
    mask = np.asarray(support_pred_labels).astype(int) == int(target_label)
    matched = support_y[mask]
    if len(matched) == 0:
        return prior_with_smoothing(support_y, alpha), 0, mask
    return prior_with_smoothing(matched, alpha), int(len(matched)), mask


def build_joint_features(z_rows: np.ndarray, o_rows: np.ndarray, embed_weight: float, output_weight: float) -> np.ndarray:
    return normalize_rows(np.concatenate([embed_weight * z_rows, output_weight * o_rows], axis=1))


def compute_target_embedding(
    image: Image.Image,
    model_name: str,
    image_size: int,
    requested_device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    model, tf, device, _ = load_model_and_preprocess(model_name, image_size, requested_device)
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb, logits = model(x)
    emb_np = normalize_rows(emb.cpu().numpy().astype(np.float32))[0]
    logits_np = logits.cpu().numpy().astype(np.float32)[0]
    pred = int(np.argmax(logits_np))
    return emb_np, logits_np, pred


def cam_colormap(cam: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    four = 4.0 * x
    r = np.clip(np.minimum(four - 1.5, -four + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(four - 0.5, -four + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(four + 0.5, -four + 2.5), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def compute_grad_cam_visuals(
    image: Image.Image,
    model_name: str,
    image_size: int,
    requested_device: str,
    target_class: int,
    overlay_alpha: float = 0.45,
) -> dict[str, Any]:
    model, tf, device, model_info = load_model_and_preprocess(model_name, image_size, requested_device)
    if not model_info.supports_grad_cam or not model_info.cam_kind:
        raise ValueError(
            f"Class activation maps are not supported for model_name={model_info.canonical_name}."
        )

    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    x.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    _, logits, activations = model.forward_with_activations(x)
    activations.retain_grad()
    score = logits[0, int(target_class)]
    score.backward()

    grads = activations.grad
    cam = build_cam_from_activations(
        activations=activations.detach(),
        gradients=grads.detach(),
        cam_kind=model_info.cam_kind,
    )
    cam_np = cam.detach().cpu().numpy().astype(np.float32)
    if cam_np.size == 0:
        cam_np = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
    else:
        cam_np -= cam_np.min()
        max_val = float(cam_np.max())
        if max_val > 0:
            cam_np /= max_val

    cam_img = Image.fromarray((255.0 * cam_np).astype(np.uint8), mode="L").resize(image.size, resample=Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    heatmap = cam_colormap(cam_resized)
    heatmap_img = Image.fromarray((255.0 * heatmap).astype(np.uint8), mode="RGB")
    base_img = image.convert("RGB").resize(image.size)
    overlay_img = Image.blend(base_img, heatmap_img, alpha=float(overlay_alpha))

    return {
        "target_class": int(target_class),
        "heatmap_image": heatmap_img,
        "overlay_image": overlay_img,
        "cam_strength_mean": float(cam_resized.mean()),
        "cam_strength_max": float(cam_resized.max()),
    }


# -----------------------------------------------------------------------------
# UI rendering helpers
# -----------------------------------------------------------------------------

def render_stage1_result_styles() -> None:
    st.markdown(
        """
        <style>
        .stage1-summary-card {
            border: 1px solid #d6dbe1;
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            background: linear-gradient(135deg, #f8fafc 0%, #edf5ff 100%);
            margin: 0.3rem 0 1rem 0;
        }
        .stage1-summary-kicker {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #52606d;
        }
        .stage1-summary-text {
            margin: 0.45rem 0 0.8rem 0;
            font-size: 1.02rem;
            line-height: 1.6;
            color: #0f172a;
        }
        .stage1-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.3rem 0 0 0;
        }
        .stage1-chip {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid transparent;
        }
        .stage1-chip.count {
            background: #e8eef7;
            border-color: #d6e2f1;
            color: #1f4f82;
        }
        .stage1-chip.coverage {
            background: #eef7ea;
            border-color: #d6ead0;
            color: #245c2b;
        }
        .stage1-chip.generic {
            background: #f3f4f6;
            border-color: #e0e3e8;
            color: #6b7280;
        }
        .stage1-secret-card {
            border: 1px solid #d6dbe1;
            border-left: 0.45rem solid #2f6db5;
            border-radius: 1rem;
            padding: 0.95rem 1rem;
            margin: 0.8rem 0;
            background: #f9fcff;
        }
        .stage1-secret-card.excess {
            border-left-color: #bd7b1f;
            background: #fffaf2;
        }
        .stage1-secret-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.35rem;
        }
        .stage1-secret-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
        }
        .stage1-secret-pill {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 0.2rem 0.5rem;
            border-radius: 999px;
            background: #dce9f8;
            color: #1f4f82;
        }
        .stage1-secret-card.excess .stage1-secret-pill {
            background: #fde7c8;
            color: #8a5300;
        }
        .stage1-secret-description {
            margin: 0.1rem 0 0.75rem 0;
            line-height: 1.55;
            color: #334155;
        }
        .stage1-metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.5rem;
            margin: 0 0 0.75rem 0;
        }
        .stage1-metric {
            border: 1px solid #e1e7ef;
            border-radius: 0.8rem;
            padding: 0.55rem 0.7rem;
            background: rgba(255, 255, 255, 0.82);
        }
        .stage1-metric-label {
            display: block;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            margin-bottom: 0.18rem;
        }
        .stage1-metric-value {
            font-size: 0.98rem;
            font-weight: 700;
            color: #0f172a;
        }
        .stage1-note {
            margin: 0.15rem 0 0.65rem 0;
            color: #5b6470;
        }
        .stage1-tag-group {
            margin-top: 0.55rem;
        }
        .stage1-tag-label {
            display: block;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            margin-bottom: 0.18rem;
        }
        .stage1-tag {
            display: inline-block;
            margin: 0.18rem 0.35rem 0 0;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid transparent;
        }
        .stage1-tag.evidence {
            background: #eaf3ff;
            border-color: #d8e7fb;
            color: #1f4f82;
        }
        .stage1-tag.positive {
            background: #e8f5e9;
            border-color: #cfe9d1;
            color: #1b5e20;
        }
        .stage1-tag.negative {
            background: #fdeeee;
            border-color: #f2d4d4;
            color: #8a2e2e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _stage1_metric_html(label: str, value: Any) -> str:
    parsed = _coerce_float(value)
    if parsed is None:
        return ""
    return (
        "<div class='stage1-metric'>"
        f"<span class='stage1-metric-label'>{html_escape(label)}</span>"
        f"<span class='stage1-metric-value'>{parsed:.2f}</span>"
        "</div>"
    )


def _stage1_tags_html(values: list[Any], css_class: str) -> str:
    tags = [
        f"<span class='stage1-tag {css_class}'>{html_escape(str(value).strip())}</span>"
        for value in values
        if str(value).strip()
    ]
    return "".join(tags)


def render_stage1_summary(parsed_result: dict[str, Any]) -> None:
    render_stage1_result_styles()
    summary = str(parsed_result.get("summary") or "").strip() or "No summary was returned."
    shared_items = parsed_result.get("shared_attributes") or []
    excess_items = parsed_result.get("excess_secrets") or []
    chips = [
        f"<span class='stage1-chip count'>Shared attributes: {len(shared_items)}</span>",
        f"<span class='stage1-chip count'>Excess secrets: {len(excess_items)}</span>",
    ]
    predicted_label_covers = str(parsed_result.get("predicted_label_covers") or "").strip()
    if predicted_label_covers:
        chips.append(
            f"<span class='stage1-chip coverage'>Predicted label covers: {html_escape(predicted_label_covers)}</span>"
        )
    rejected_generic_terms = [
        str(term).strip() for term in (parsed_result.get("rejected_generic_terms") or []) if str(term).strip()
    ]
    chips.extend(
        f"<span class='stage1-chip generic'>Rejected generic term: {html_escape(term)}</span>"
        for term in rejected_generic_terms[:6]
    )
    st.markdown(
        f"""
        <div class="stage1-summary-card">
          <div class="stage1-summary-kicker">Stage 1 Summary</div>
          <div class="stage1-summary-text">{html_escape(summary)}</div>
          <div class="stage1-chip-row">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_retrieval_styles() -> None:
    st.markdown(
        """
        <style>
        .retrieval-summary-card {
            border: 1px solid #d6dbe1;
            border-radius: 1rem;
            padding: 0.95rem 1rem;
            background: linear-gradient(135deg, #f7fbff 0%, #eff6ff 100%);
            margin: 0.3rem 0 1rem 0;
        }
        .retrieval-summary-title {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #52606d;
            margin-bottom: 0.4rem;
        }
        .retrieval-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .retrieval-chip {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid transparent;
        }
        .retrieval-chip.count {
            background: #e6eef9;
            border-color: #d3e0f3;
            color: #1f4f82;
        }
        .retrieval-chip.source {
            background: #eef7ea;
            border-color: #d7e9d0;
            color: #245c2b;
        }
        .retrieval-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.35rem;
        }
        .retrieval-id {
            font-size: 0.98rem;
            font-weight: 700;
            color: #0f172a;
            word-break: break-word;
        }
        .retrieval-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 0.6rem;
        }
        .retrieval-badge {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .retrieval-badge.rank {
            background: #dbeafe;
            border-color: #c8dcfa;
            color: #1d4f91;
        }
        .retrieval-badge.similarity {
            background: #e8f5e9;
            border-color: #d2e8d3;
            color: #1b5e20;
        }
        .retrieval-image-missing {
            border: 1px dashed #c5ccd5;
            border-radius: 0.9rem;
            padding: 1.4rem 0.9rem;
            text-align: center;
            background: #f8fafc;
            color: #6b7280;
            margin: 0.1rem 0 0.85rem 0;
        }
        .retrieval-caption-label {
            display: block;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            margin: 0.15rem 0 0.2rem 0;
        }
        .retrieval-caption {
            border: 1px solid #e1e7ef;
            border-radius: 0.9rem;
            padding: 0.75rem 0.85rem;
            background: #fbfdff;
            color: #334155;
            line-height: 1.55;
        }
        .retrieval-semantics-label {
            display: block;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            margin: 0.8rem 0 0.25rem 0;
        }
        .retrieval-semantics-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem;
            margin: 0 0 0.85rem 0;
        }
        .retrieval-semantics-item {
            border: 1px solid #e2e8f0;
            border-radius: 0.85rem;
            padding: 0.55rem 0.7rem;
            background: #f8fafc;
        }
        .retrieval-semantics-key {
            display: block;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 0.18rem;
        }
        .retrieval-semantics-value {
            color: #0f172a;
            line-height: 1.45;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _generated_semantics_html(row: pd.Series) -> str:
    parts = semantic_parts_from_record(row)
    if not parts:
        return ""
    items = "".join(
        (
            f'<div class="retrieval-semantics-item">'
            f'<span class="retrieval-semantics-key">{html_escape(label)}</span>'
            f'<div class="retrieval-semantics-value">{html_escape(value)}</div>'
            f"</div>"
        )
        for label, value in parts
    )
    return (
        f'<div class="retrieval-semantics-panel">'
        f'<span class="retrieval-semantics-label">Generation Semantics</span>'
        f'<div class="retrieval-semantics-grid">{items}</div>'
        f"</div>"
    )


def render_html_fragment(html: str) -> None:
    if not html:
        return
    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        html_renderer(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def format_generated_target_option(row: pd.Series) -> str:
    image_id = str(row.get("image_id", "")).strip() or "(missing id)"
    family = str(row.get("scene_family_label") or row.get("scene_family") or "").strip()
    primary = str(row.get("primary_label") or row.get("primary_object") or "").strip()
    secondary = str(row.get("secondary_label") or row.get("secondary_object") or "").strip()
    ternary = str(row.get("ternary_label") or row.get("ternary_object") or "").strip()
    background = str(row.get("background_label") or row.get("background_scene") or "").strip()
    return f"{image_id} | {family} | {primary} + {secondary} + {ternary} @ {background}"


def render_retrieved_neighbors(
    neighbor_df: pd.DataFrame,
    cfg: dict[str, Any],
    project_root: Path,
    id_column: str,
    prompt_text_column: str,
) -> None:
    if neighbor_df.empty:
        st.info("No retrieved neighbors available.")
        return

    render_retrieval_styles()
    st.markdown(
        f"""
        <div class="retrieval-summary-card">
          <div class="retrieval-summary-title">Retrieved Neighbor Overview</div>
          <div class="retrieval-chip-row">
            <span class="retrieval-chip count">Neighbors shown: {len(neighbor_df)}</span>
            <span class="retrieval-chip source">Caption source: {html_escape(prompt_text_column)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    grid_cols = st.columns(2)
    for idx, (_, row) in enumerate(neighbor_df.iterrows(), start=1):
        with grid_cols[(idx - 1) % 2]:
            with st.container(border=True):
                image_id = str(row.get(id_column, "")).strip() or "(missing id)"
                similarity = float(row.get("similarity", float("nan")))
                caption = str(row.get(prompt_text_column, "") or "").strip()
                caption_preview = caption if len(caption) <= 420 else caption[:420].rstrip() + "..."

                st.markdown(
                    f"""
                    <div class="retrieval-header">
                      <div class="retrieval-id">{html_escape(image_id)}</div>
                    </div>
                    <div class="retrieval-badge-row">
                      <span class="retrieval-badge rank">#{idx}</span>
                      <span class="retrieval-badge similarity">Similarity {similarity:.4f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                img_path = resolve_image_path_from_record(row, cfg, project_root)
                if img_path is not None and img_path.exists():
                    st.image(str(img_path), width="stretch")
                else:
                    st.markdown(
                        "<div class='retrieval-image-missing'>Image not found</div>",
                        unsafe_allow_html=True,
                    )

                semantics_html = _generated_semantics_html(row)
                if semantics_html:
                    render_html_fragment(semantics_html)

                st.markdown(
                    f"""
                    <span class="retrieval-caption-label">Caption Preview</span>
                    <div class="retrieval-caption">{html_escape(caption_preview or 'No caption available.')}</div>
                    """,
                    unsafe_allow_html=True,
                )


def render_secret_cards(items: list[dict[str, Any]], field: str) -> None:
    if not items:
        st.info(f"No {field} available.")
        return
    kind = "excess" if field == "excess_secrets" else "shared"
    label = "Excess Secret" if kind == "excess" else "Shared Attribute"
    for item in items:
        title = str(item.get("name") or "untitled")
        description = str(item.get("description") or "").strip()
        why_excess = str(item.get("why_excess") or "").strip()
        metrics = "".join([
            _stage1_metric_html("Specificity", item.get("specificity")),
            _stage1_metric_html("Relevance", item.get("relevance")),
            _stage1_metric_html("Privacy relevance", item.get("privacy_relevance")),
            _stage1_metric_html("Task relevance", item.get("task_relevance")),
        ])
        evidence_html = _stage1_tags_html(item.get("evidence") or [], "evidence")
        positive_html = _stage1_tags_html(item.get("positive_patterns") or [], "positive")
        negative_html = _stage1_tags_html(item.get("negative_patterns") or [], "negative")
        blocks: list[str] = []
        if evidence_html:
            blocks.append(
                "<div class='stage1-tag-group'>"
                "<span class='stage1-tag-label'>Evidence</span>"
                f"{evidence_html}"
                "</div>"
            )
        if positive_html:
            blocks.append(
                "<div class='stage1-tag-group'>"
                "<span class='stage1-tag-label'>Positive Patterns</span>"
                f"{positive_html}"
                "</div>"
            )
        if negative_html:
            blocks.append(
                "<div class='stage1-tag-group'>"
                "<span class='stage1-tag-label'>Negative Patterns</span>"
                f"{negative_html}"
                "</div>"
            )
        note_html = (
            f"<div class='stage1-note'><strong>Why excess:</strong> {html_escape(why_excess)}</div>"
            if why_excess else ""
        )
        st.markdown(
            f"""
            <div class="stage1-secret-card {kind}">
              <div class="stage1-secret-head">
                <div class="stage1-secret-title">{html_escape(title)}</div>
                <div class="stage1-secret-pill">{label}</div>
              </div>
              <div class="stage1-secret-description">{html_escape(description) if description else 'No description provided.'}</div>
              <div class="stage1-metric-grid">{metrics}</div>
              {note_html}
              {''.join(blocks)}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _safe_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if _is_missing(value):
        return {}
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if _is_missing(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(v) for v in parsed if str(v).strip()]
    return [str(parsed)] if str(parsed).strip() else []


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _format_prob(value: Any) -> str:
    parsed = _coerce_float(value)
    return f"{100.0 * parsed:.1f}%" if parsed is not None else "NA"


def _format_value(value: Any, ndigits: int = 3, suffix: str = "") -> str:
    parsed = _coerce_float(value)
    return f"{parsed:.{ndigits}f}{suffix}" if parsed is not None else "NA"


def _extract_confirmation_thresholds(row: pd.Series) -> dict[str, Any]:
    thresholds = _safe_json_dict(row.get("thresholds_json"))
    for key in row.index:
        key_str = str(key)
        if not key_str.startswith("threshold_"):
            continue
        value = row.get(key)
        if _is_missing(value):
            continue
        thresholds[key_str.removeprefix("threshold_")] = value
    return thresholds


def _status_rank(status: str) -> int:
    return {"confirmed": 0, "inconclusive": 1, "rejected": 2}.get(str(status or "").strip().lower(), 3)


def _chip_html(label: str, passed: bool | None) -> str:
    if passed is True:
        klass = "pass"
    elif passed is False:
        klass = "fail"
    else:
        klass = "neutral"
    return f'<span class="confirmation-chip {klass}">{html_escape(label)}</span>'


def render_confirmation_results(posterior_df: pd.DataFrame) -> None:
    if posterior_df.empty:
        st.info("No confirmation results available.")
        return

    work = posterior_df.copy()
    status_col = "confirmation_status" if "confirmation_status" in work.columns else "status"
    if status_col not in work.columns:
        st.info("No confirmation status column was present in the generated confirmation rows.")
        return

    work["__status"] = work[status_col].fillna("inconclusive").astype(str).str.strip().str.lower()
    work["__secret"] = (
        work["attribute_name"].fillna("").astype(str)
        if "attribute_name" in work.columns
        else work.get("secret", pd.Series([""] * len(work), index=work.index)).fillna("").astype(str)
    )
    work["__sort_shift"] = pd.to_numeric(work.get("posterior_shift"), errors="coerce").fillna(float("-inf"))
    work = work.sort_values(
        by=["__status", "__sort_shift", "__secret"],
        key=lambda col: col.map(_status_rank) if col.name == "__status" else col,
        ascending=[True, False, True],
    ).reset_index(drop=True)

    thresholds = _extract_confirmation_thresholds(work.iloc[0])
    min_shift = _coerce_float(thresholds.get("min_posterior_shift"), 0.02)
    min_lift = _coerce_float(thresholds.get("min_excess_lift"), 1.10)
    min_excess_kl = _coerce_float(thresholds.get("min_excess_kl"), 0.01)
    min_neighbor_support = int(_coerce_float(thresholds.get("min_neighbor_support"), 3) or 3)

    status_counts = work["__status"].value_counts().to_dict()

    st.markdown(
        """
        <style>
        .confirmation-rule-box {
            border: 1px solid #d6dbe1;
            border-radius: 0.85rem;
            padding: 0.95rem 1rem;
            background: #fafbfc;
            margin: 0.4rem 0 1rem 0;
        }
        .confirmation-rule-box p {
            margin: 0.2rem 0;
        }
        .confirmation-card {
            border: 1px solid #d6dbe1;
            border-left: 0.45rem solid #b7bfc8;
            border-radius: 0.95rem;
            padding: 0.9rem 1rem;
            margin: 0.8rem 0;
            background: #ffffff;
        }
        .confirmation-card.confirmed {
            border-left-color: #2e7d32;
            background: #f4fbf6;
        }
        .confirmation-card.inconclusive {
            border-left-color: #b7791f;
            background: #fffaf2;
        }
        .confirmation-card.rejected {
            border-left-color: #9aa1ab;
            background: #f3f4f6;
            color: #6b7280;
        }
        .confirmation-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
            margin-bottom: 0.35rem;
        }
        .confirmation-secret {
            font-size: 1.02rem;
            font-weight: 700;
        }
        .confirmation-card.confirmed .confirmation-secret {
            color: #1b5e20;
        }
        .confirmation-card.rejected .confirmation-secret {
            color: #6b7280;
        }
        .confirmation-status {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .confirmation-status.confirmed { color: #1b5e20; }
        .confirmation-status.inconclusive { color: #8a5a00; }
        .confirmation-status.rejected { color: #6b7280; }
        .confirmation-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem;
            margin: 0.25rem 0 0.55rem 0;
            font-size: 0.94rem;
        }
        .confirmation-metrics span strong {
            font-weight: 700;
        }
        .confirmation-decision {
            margin: 0.2rem 0 0.55rem 0;
        }
        .confirmation-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.45rem 0 0.55rem 0;
        }
        .confirmation-chip {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid transparent;
        }
        .confirmation-chip.pass {
            background: #e8f5e9;
            border-color: #c8e6c9;
            color: #1b5e20;
        }
        .confirmation-chip.fail {
            background: #eceff1;
            border-color: #d4d8dd;
            color: #6b7280;
        }
        .confirmation-chip.neutral {
            background: #fff3e0;
            border-color: #ffe0b2;
            color: #8a5a00;
        }
        .confirmation-rationale {
            margin: 0.35rem 0 0.25rem 0;
        }
        .confirmation-note {
            margin: 0.2rem 0;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="confirmation-rule-box">
          <p><strong>Decision criteria</strong></p>
          <p><strong>Confirmed</strong>: all evidence checks pass and no hard support failure blocks confirmation.</p>
          <p>Evidence thresholds: <code>posterior_shift ≥ {min_shift:.3f}</code>, <code>excess_lift ≥ {min_lift:.2f}x</code>, <code>excess_KL ≥ {min_excess_kl:.4f}</code>.</p>
          <p>Hard support checks: <code>baseline_available</code>, <code>support_status_ok</code>, <code>neighbor_support_ok</code> with at least <code>{min_neighbor_support}</code> supporting neighbors.</p>
          <p><strong>Rejected</strong>: at least one evidence threshold fails. <strong>Inconclusive</strong>: the evidence thresholds pass, but a hard support check fails.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Confirmed", int(status_counts.get("confirmed", 0)))
    c2.metric("Inconclusive", int(status_counts.get("inconclusive", 0)))
    c3.metric("Rejected", int(status_counts.get("rejected", 0)))

    for _, row in work.iterrows():
        status = str(row.get("__status", "inconclusive"))
        secret = str(row.get("__secret", "")).strip() or str(row.get("secret", "unnamed"))
        rationale = str(row.get("rationale", "")).strip()
        diagnostics = _safe_json_dict(row.get("support_diagnostics_json"))
        evidence_checks = diagnostics.get("evidence_checks") if isinstance(diagnostics.get("evidence_checks"), dict) else {}
        support_checks = diagnostics.get("support_checks") if isinstance(diagnostics.get("support_checks"), dict) else {}
        support_failures = _safe_json_list(row.get("support_failures_json"))
        support_warnings = _safe_json_list(row.get("support_warnings_json"))

        method = str(row.get("method", "")).strip()
        status_label = status.title()
        baseline = _format_prob(row.get("baseline_posterior", row.get("p_baseline_1")))
        conditional = _format_prob(row.get("conditional_posterior", row.get("p_secret_given_baseline_1")))
        shift = _format_value(row.get("posterior_shift", row.get("delta_p")), ndigits=3)
        lift = _format_value(row.get("excess_lift"), ndigits=2, suffix="x")
        excess_kl = _format_value(row.get("excess_kl_nats", row.get("excess_local_kl_nats")), ndigits=4, suffix=" nats")

        if status == "confirmed":
            decision_text = "Confirmed because the incremental evidence clears every Stage 3 threshold and the hard support checks pass."
        elif status == "inconclusive":
            decision_text = "Not confirmed yet: the evidence thresholds pass, but at least one hard support check failed."
        else:
            decision_text = "Rejected because the incremental evidence does not clear at least one Stage 3 threshold."

        chips = [
            _chip_html(f"shift ≥ {min_shift:.3f}", evidence_checks.get("posterior_shift_ok")),
            _chip_html(f"lift ≥ {min_lift:.2f}x", evidence_checks.get("excess_lift_ok")),
            _chip_html(f"excess KL ≥ {min_excess_kl:.4f}", evidence_checks.get("excess_kl_ok")),
            _chip_html("baseline available", support_checks.get("baseline_available")),
            _chip_html("support status ok", support_checks.get("support_status_ok")),
            _chip_html(f"neighbor support ≥ {min_neighbor_support}", support_checks.get("neighbor_support_ok")),
        ]

        notes: list[str] = []
        if support_failures:
            notes.append(f"<div class='confirmation-note'><strong>Hard support failures:</strong> {html_escape(', '.join(support_failures))}</div>")
        if support_warnings:
            notes.append(f"<div class='confirmation-note'><strong>Support warnings:</strong> {html_escape(', '.join(support_warnings))}</div>")

        method_html = f"<span><strong>method</strong> {html_escape(method)}</span>" if method else ""
        rationale_html = f"<div class='confirmation-rationale'>{html_escape(rationale)}</div>" if rationale else ""

        st.markdown(
            f"""
            <div class="confirmation-card {html_escape(status)}">
              <div class="confirmation-head">
                <div class="confirmation-secret">{html_escape(secret)}</div>
                <div class="confirmation-status {html_escape(status)}">{html_escape(status_label)}</div>
              </div>
              <div class="confirmation-metrics">
                <span><strong>q(S|Y)</strong> {html_escape(baseline)}</span>
                <span><strong>q(S|Z,Y)</strong> {html_escape(conditional)}</span>
                <span><strong>shift</strong> {html_escape(shift)}</span>
                <span><strong>lift</strong> {html_escape(lift)}</span>
                <span><strong>excess KL</strong> {html_escape(excess_kl)}</span>
                {method_html}
              </div>
              <div class="confirmation-decision">{html_escape(decision_text)}</div>
              <div class="confirmation-chip-row">{''.join(chips)}</div>
              {rationale_html}
              {''.join(notes)}
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Privacy Leakage Audit", layout="wide")
    st.title("Retrieval-based Privacy Leakage Audit")
    st.caption("Select a generated image, retrieve similar DCI corpus examples, export a Stage 1 prompt, paste the JSON reply, and confirm excess leakage for the discovered semantics.")

    with st.sidebar:
        st.header("Configuration")
        default_config = "configs/mvp.yaml"
        config_path_str = st.text_input("Config path", value=default_config)
        inferred_root = str(infer_project_root_from_config(Path(config_path_str).expanduser().resolve())) if Path(config_path_str).expanduser().exists() else "."
        project_root_str = st.text_input("Project root", value=inferred_root)
        load_button = st.button("Load artifacts", type="primary")

    if load_button or "artifacts" not in st.session_state:
        try:
            st.session_state["artifacts"] = load_artifacts(config_path_str, project_root_str)
        except Exception as e:
            st.error(f"Failed to load artifacts: {e}")
            st.stop()

    artifacts: AppArtifacts = st.session_state["artifacts"]
    cfg = artifacts.cfg
    manifest = artifacts.manifest.copy()
    generated_targets = artifacts.generated_targets.copy()
    id_column = artifacts.id_column
    text_column = artifacts.text_column
    prompt_text_column = choose_prompt_text_column(manifest, text_column)
    emb_cfg = cfg.get("embeddings") or {}
    model_name = str(emb_cfg.get("model_name", "mobilenet_v3_small"))
    requested_device = str(emb_cfg.get("device", "auto"))
    model_info = get_model_info(model_name)
    llm_cfg_for_ui = ((cfg.get("discovery") or {}).get("offline_llm") or (cfg.get("discovery") or {}).get("llm") or {})
    generic_blacklist = narrow_generic_blacklist(llm_cfg_for_ui.get("generic_blacklist") or [])

    with st.sidebar:
        st.divider()
        st.header("Audit Controls")
        top_k_neighbors = st.slider("Top-k retrieved neighbors", 3, 20, int(((cfg.get("retrieval") or {}).get("top_k", 10))))
        stage1_min_spec = st.slider("Stage 1 min specificity", 0.0, 1.0, float((llm_cfg_for_ui.get("min_specificity", 0.20))), 0.05)
        stage1_min_rel = st.slider("Stage 1 min relevance", 0.0, 1.0, float((llm_cfg_for_ui.get("min_relevance", 0.10))), 0.05)
        stage1_min_priv = st.slider("Stage 1 min privacy relevance", 0.0, 1.0, float((llm_cfg_for_ui.get("min_privacy_relevance", 0.00))), 0.05)
        stage1_max_task_rel = st.slider(
            "Stage 1 max task relevance to treat a shared attribute as excess",
            0.0,
            1.0,
            float((((cfg.get("operationalization") or {}).get("max_task_relevance", 0.75)))),
            0.05,
        )

    left, right = st.columns([1, 1.25])
    with left:
        if generated_targets.empty:
            st.error("No generated target images are configured. Check `generated_targets.manifest_csv` in the config.")
            st.stop()
        if "scene_family_label" in generated_targets.columns:
            generated_targets["__scene_family_display"] = generated_targets["scene_family_label"].fillna("").astype(str)
        else:
            generated_targets["__scene_family_display"] = ""
        if generated_targets["__scene_family_display"].eq("").all() and "scene_family" in generated_targets.columns:
            generated_targets["__scene_family_display"] = generated_targets["scene_family"].fillna("").astype(str)

        scene_family_options = ["All"] + sorted(
            value for value in generated_targets["__scene_family_display"].dropna().astype(str).unique().tolist() if value
        )
        selected_scene_family = st.selectbox("Generated scene family", scene_family_options, index=0)
        filtered_targets = generated_targets.copy()
        if selected_scene_family != "All":
            filtered_targets = filtered_targets[
                filtered_targets["__scene_family_display"].astype(str) == str(selected_scene_family)
            ].reset_index(drop=True)
        if filtered_targets.empty:
            st.info("No generated images match the selected scene family.")
            st.stop()
        filtered_targets["__target_option"] = filtered_targets.apply(format_generated_target_option, axis=1)
        target_option = st.selectbox(
            "Generated target image",
            filtered_targets["__target_option"].tolist(),
            index=0,
        )

    selected_target_row = filtered_targets.loc[
        filtered_targets["__target_option"].astype(str) == str(target_option)
    ].iloc[0]
    selected_target_name = str(selected_target_row.get("image_id", "")).strip() or "generated_target.png"
    selected_target_path = resolve_image_path_from_record(selected_target_row, cfg, artifacts.project_root)
    if selected_target_path is None or not selected_target_path.exists():
        st.error(f"Selected generated image could not be resolved: {selected_target_name}")
        st.stop()

    selected_target_bytes = selected_target_path.read_bytes()
    image = Image.open(selected_target_path).convert("RGB")
    with left:
        st.image(image, width="stretch")
        target_semantics_html = _generated_semantics_html(selected_target_row)
        if target_semantics_html:
            render_html_fragment(target_semantics_html)
        target_prompt = str(selected_target_row.get("prompt", "") or "").strip()
        if target_prompt:
            with st.expander("Generation prompt", expanded=False):
                st.code(target_prompt, language="text")

    image_size = int(emb_cfg.get("image_size", 224))
    with st.spinner("Computing embedding and prediction..."):
        target_emb, target_logits, target_pred = compute_target_embedding(
            image,
            model_name=model_name,
            image_size=image_size,
            requested_device=requested_device,
        )

    pred_name = artifacts.class_name_map.get(str(target_pred), "")
    predicted_label_desc = f"{target_pred}: {pred_name}" if pred_name else str(target_pred)
    probs = torch.softmax(torch.tensor(target_logits), dim=0).numpy()
    top5_idx = np.argsort(-probs)[:5]
    top5_df = pd.DataFrame({
        "class_id": top5_idx,
        "class_name": [artifacts.class_name_map.get(str(i), "") for i in top5_idx],
        "probability": probs[top5_idx],
    })

    # retrieval
    corpus_sims = artifacts.corpus_embeddings @ target_emb
    nn_order = np.argsort(-corpus_sims)[:top_k_neighbors]
    neighbor_ids = artifacts.corpus_ids[nn_order].astype(str).tolist()
    neighbor_df = manifest.set_index(id_column, drop=False).reindex(neighbor_ids).reset_index(drop=True)
    neighbor_df["similarity"] = corpus_sims[nn_order]
    neighbor_df[prompt_text_column] = neighbor_df[prompt_text_column].fillna("") if prompt_text_column in neighbor_df.columns else ""
    neighbor_captions = neighbor_df[prompt_text_column].fillna("").astype(str).tolist()
    heuristics = top_phrases(neighbor_captions, top_n=10)
    prompt_text = build_stage1_prompt(
        target_image_id=selected_target_name,
        predicted_label_desc=predicted_label_desc,
        captions=neighbor_captions,
        heuristic_phrases=heuristics,
        retrieved_top_k=len(neighbor_captions),
    )

    with right:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted class", str(target_pred))
        c2.metric("Class name", pred_name or "-")
        c3.metric("Corpus neighbors", len(neighbor_ids))
        c4.metric("Backbone", model_info.canonical_name)
        st.subheader("Top-5 task predictions")
        st.dataframe(top5_df, width="stretch", hide_index=True)

    tab_retrieval, tab_cam, tab_prompt, tab_parse, tab_metrics = st.tabs([
        "Retrieved neighbors",
        "Class activation map",
        "Stage 1 prompt",
        "Parse Stage 1 JSON",
        "Leakage metrics",
    ])

    with tab_retrieval:
        st.subheader("Nearest corpus neighbors")
        render_retrieved_neighbors(
            neighbor_df=neighbor_df,
            cfg=cfg,
            project_root=artifacts.project_root,
            id_column=id_column,
            prompt_text_column=prompt_text_column,
        )

    with tab_cam:
        st.subheader("Class activation map")
        st.caption("Grad-CAM style overlay over the selected generated image for a selected class.")
        cam_options: list[tuple[str, int]] = []
        for _, row in top5_df.iterrows():
            class_id = int(row["class_id"])
            class_name = str(row["class_name"] or "-")
            prob = float(row["probability"])
            cam_options.append((f"{class_id}: {class_name} ({prob:.1%})", class_id))
        if not model_info.supports_grad_cam:
            st.info(
                f"CAM visualization is disabled for `{model_info.canonical_name}`. "
                "Quantized backbones are supported for inference, but not for gradient-based saliency."
            )
        else:
            cam_choice = st.selectbox("Class to visualize", [label for label, _ in cam_options], index=0)
            cam_alpha = st.slider("Overlay strength", 0.10, 0.90, 0.45, 0.05)
            selected_cam_class = next(class_id for label, class_id in cam_options if label == cam_choice)

            with st.spinner("Computing class activation map..."):
                cam_result = compute_grad_cam_visuals(
                    image=image,
                    model_name=model_name,
                    image_size=image_size,
                    requested_device=requested_device,
                    target_class=selected_cam_class,
                    overlay_alpha=cam_alpha,
                )

            cam_col1, cam_col2 = st.columns(2)
            cam_col1.image(
                cam_result["overlay_image"],
                width="stretch",
                caption=f"Overlay for class {cam_result['target_class']}",
            )
            cam_col2.image(
                cam_result["heatmap_image"],
                width="stretch",
                caption="Heatmap only",
            )

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Mean activation", f"{cam_result['cam_strength_mean']:.3f}")
            metric_col2.metric("Peak activation", f"{cam_result['cam_strength_max']:.3f}")

    with tab_prompt:
        st.subheader("Offline Stage 1 prompt")
        st.caption(f"Prompt caption source: `{prompt_text_column}`")
        st.download_button("Download prompt as .txt", data=prompt_text, file_name=f"{Path(selected_target_name).stem}.stage1.prompt.txt", mime="text/plain")
        st.code(prompt_text, language="text")

    # Parse stage1 JSON
    with tab_parse:
        st.subheader("Paste Stage 1 JSON response")
        response_text = st.text_area("Stage 1 response JSON", height=320, placeholder="Paste the JSON-only Stage 1 response here...")
        parsed_result = None
        if response_text.strip():
            try:
                raw = extract_json_object(response_text)
                parsed_result = sanitize_stage1_result(
                    raw,
                    min_specificity=stage1_min_spec,
                    min_relevance=stage1_min_rel,
                    min_privacy_relevance=stage1_min_priv,
                    generic_blacklist=generic_blacklist,
                    max_task_relevance=stage1_max_task_rel,
                )
                st.success("Parsed successfully.")
                st.session_state["parsed_stage1"] = parsed_result
            except Exception as e:
                st.error(f"Could not parse response: {e}")
        elif "parsed_stage1" in st.session_state:
            parsed_result = st.session_state["parsed_stage1"]

        if parsed_result:
            render_stage1_summary(parsed_result)
            shared_items = parsed_result.get("shared_attributes") or []
            excess_items = parsed_result.get("excess_secrets") or []
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### Shared Attributes ({len(shared_items)})")
                st.caption("Concepts the Stage 1 parse treats as plausibly explained by the task label or common class evidence.")
                render_secret_cards(shared_items, "shared_attributes")
            with c2:
                st.markdown(f"### Excess Secrets ({len(excess_items)})")
                st.caption("Concepts the Stage 1 parse treats as potentially leaking information beyond the task label.")
                render_secret_cards(excess_items, "excess_secrets")
            st.download_button(
                "Download sanitized Stage 1 JSON",
                data=json.dumps(parsed_result, indent=2),
                file_name=f"{Path(selected_target_name).stem}.stage1.response.json",
                mime="application/json",
            )

    with tab_metrics:
        st.subheader("Generate confirmation report from project scripts")
        parsed_result = st.session_state.get("parsed_stage1")
        if not parsed_result:
            st.info("Paste a Stage 1 JSON response first.")
        else:
            col1, col2, col3 = st.columns(3)
            secret_source = col1.selectbox("Secret source", ["excess_secrets", "shared_attributes", "both"], index=0, help="`excess_secrets` is derived from shared attributes whose task_relevance is at or below the configured threshold.")
            baseline_mode = col2.selectbox("Baseline mode", ["NONE", "Y", "O"], index=1)
            label_scope = col3.selectbox(
                "Support label scope",
                ["corpus_index_only", "full_manifest"],
                index=1,
                help="`full_manifest` is recommended for confirmation because it gives the task-label baseline enough support. `corpus_index_only` is faster but often too sparse and can force inconclusive results.",
            )

            st.caption(
                "This runs the same script workflow described in the README: "
                "`scripts/build_labels.py` -> `scripts/stage2_confirm_excess_leakage.py` -> `scripts/report_targets.py`."
            )

            run_pipeline = st.button("Generate script-based report", type="primary")
            if run_pipeline:
                with st.spinner("Running build_labels.py, stage2_confirm_excess_leakage.py, and report_targets.py..."):
                    try:
                        st.session_state["pipeline_report_result"] = generate_pipeline_report(
                            artifacts=artifacts,
                            target_image=image,
                            target_name=selected_target_name,
                            target_bytes=selected_target_bytes,
                            target_record=selected_target_row.to_dict(),
                            target_emb=target_emb,
                            target_logits=target_logits,
                            target_pred=target_pred,
                            predicted_label_desc=predicted_label_desc,
                            neighbor_ids=neighbor_ids,
                            neighbor_similarities=[float(v) for v in neighbor_df["similarity"].tolist()],
                            neighbor_captions=neighbor_captions,
                            heuristic_phrases=heuristics,
                            parsed_result=parsed_result,
                            secret_source=secret_source,
                            baseline_mode=baseline_mode,
                            label_scope=label_scope,
                        )
                        st.session_state["pipeline_report_error"] = None
                    except Exception as e:
                        st.session_state["pipeline_report_result"] = None
                        st.session_state["pipeline_report_error"] = str(e)

            pipeline_error = st.session_state.get("pipeline_report_error")
            if pipeline_error:
                st.error(pipeline_error)

            pipeline_result = st.session_state.get("pipeline_report_result")
            if pipeline_result:
                posterior_df = pipeline_result["posterior_df"]
                metrics_df = pipeline_result["metrics_df"]
                report_markdown = pipeline_result["report_markdown"]
                confirmation_markdown = pipeline_result["report_confirmation_markdown"]
                labels_df = pipeline_result["labels_df"]

                st.markdown("### Confirmation Results")
                if not posterior_df.empty:
                    render_confirmation_results(posterior_df)
                elif confirmation_markdown:
                    st.markdown(confirmation_markdown)
                else:
                    st.info("The generated report did not contain a `## Confirmation results` section.")

                if confirmation_markdown:
                    with st.expander("Raw confirmation markdown"):
                        st.markdown(confirmation_markdown)

                st.download_button(
                    "Download generated report (.md)",
                    data=report_markdown,
                    file_name=f"{Path(selected_target_name).stem}.target_report.md",
                    mime="text/markdown",
                )
                if not posterior_df.empty:
                    st.download_button(
                        "Download target confirmations (.csv)",
                        data=posterior_df.to_csv(index=False),
                        file_name=f"{Path(selected_target_name).stem}.target_confirmations.csv",
                        mime="text/csv",
                    )
                if not metrics_df.empty:
                    st.download_button(
                        "Download confirmation summary (.csv)",
                        data=metrics_df.to_csv(index=False),
                        file_name=f"{Path(selected_target_name).stem}.confirmation_summary.csv",
                        mime="text/csv",
                    )

                with st.expander("Script logs"):
                    st.markdown("**build_labels.py stdout**")
                    st.code(pipeline_result["build_labels_stdout"] or "(empty)", language="text")
                    if pipeline_result["build_labels_stderr"]:
                        st.markdown("**build_labels.py stderr**")
                        st.code(pipeline_result["build_labels_stderr"], language="text")
                    st.markdown("**stage2_confirm_excess_leakage.py stdout**")
                    st.code(pipeline_result["stage2_stdout"] or "(empty)", language="text")
                    if pipeline_result["stage2_stderr"]:
                        st.markdown("**stage2_confirm_excess_leakage.py stderr**")
                        st.code(pipeline_result["stage2_stderr"], language="text")
                    st.markdown("**report_targets.py stdout**")
                    st.code(pipeline_result["report_stdout"] or "(empty)", language="text")
                    if pipeline_result["report_stderr"]:
                        st.markdown("**report_targets.py stderr**")
                        st.code(pipeline_result["report_stderr"], language="text")


if __name__ == "__main__":
    main()
