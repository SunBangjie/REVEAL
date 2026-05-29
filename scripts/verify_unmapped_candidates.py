from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.runner import EvaluationRunner
from evaluation.semantic_support import normalize_text


UNMAPPED_LABEL = "Unmapped candidate"
TASK_ALIGNED_LABEL = "Task-aligned recovery"
EXCESS_GROUNDED_LABEL = "Excess grounded recovery"

AUTO_VALID_LABEL = "auto_verified_extra"
AUTO_HALLUCINATION_LABEL = "auto_likely_hallucination"
AUTO_UNCERTAIN_LABEL = "auto_uncertain"
AUTO_MISSING_LABEL = "auto_missing_image"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if np.isnan(number):
        return default
    return number


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    unique_scores, inverse = np.unique(scores, return_inverse=True)
    for group_id in range(len(unique_scores)):
        mask = inverse == group_id
        if int(mask.sum()) <= 1:
            continue
        ranks[mask] = float(ranks[mask].mean())

    pos_rank_sum = float(ranks[labels == 1].sum())
    auc = (pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def _split_patterns(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, list):
            return [_safe_text(item) for item in payload if _safe_text(item)]
    if "|" in text:
        return [item.strip() for item in text.split("|") if item.strip()]
    if ";" in text:
        return [item.strip() for item in text.split(";") if item.strip()]
    return [text]


def _normalize_attribute_name(text: str) -> str:
    return normalize_text(text).replace("  ", " ").strip()


def _pair_key(image_id: Any, attribute_name: Any) -> str:
    return f"{_safe_text(image_id)}::{_normalize_attribute_name(_safe_text(attribute_name))}"


def _setting_signature(row: pd.Series) -> str:
    model_name = _safe_text(row.get("model_name")) or "-"
    k = _safe_text(row.get("K")) or "-"
    llm_name = _safe_text(row.get("llm_name")) or "-"
    return f"{model_name} | K={k} | {llm_name}"


def _resolve_device(device: str) -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return _safe_text(device).lower() or "cpu"
    choice = _safe_text(device).lower() or "auto"
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_prompt_texts(row: pd.Series, max_prompts: int) -> list[str]:
    attribute_name = _normalize_attribute_name(_safe_text(row.get("attribute_name")))
    description = _safe_text(row.get("attribute_description"))
    patterns = [_normalize_attribute_name(item) for item in _split_patterns(row.get("positive_patterns"))]

    candidates: list[str] = []

    def add_prompt(text: str) -> None:
        text = _safe_text(text)
        if not text:
            return
        if text not in candidates:
            candidates.append(text)

    if attribute_name:
        add_prompt(attribute_name)
        add_prompt(f"a photo of {attribute_name}")
        add_prompt(f"an image showing {attribute_name}")

    if description:
        add_prompt(description)

    for pattern in patterns:
        add_prompt(pattern)
        add_prompt(f"a photo with {pattern}")
        if len(candidates) >= max_prompts:
            break

    return candidates[:max_prompts]


class ClipVerifier:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        image_batch_size: int = 16,
        text_batch_size: int = 64,
    ) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ModuleNotFoundError as exc:
            missing = str(exc).split("'")[1] if "'" in str(exc) else "required package"
            raise ModuleNotFoundError(
                "verify_unmapped_candidates.py requires the project vision stack. "
                f"Missing dependency: {missing}. Install the dependencies from pyproject.toml."
            ) from exc

        self.torch = torch
        self.model_name = str(model_name)
        self.device = _resolve_device(device)
        self.image_batch_size = int(image_batch_size)
        self.text_batch_size = int(text_batch_size)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _feature_tensor(self, value: Any) -> Any:
        if isinstance(value, self.torch.Tensor):
            return value

        for attr in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
            candidate = getattr(value, attr, None)
            if isinstance(candidate, self.torch.Tensor):
                if attr == "last_hidden_state" and candidate.ndim >= 3:
                    return candidate[:, 0]
                return candidate

        if isinstance(value, dict):
            for key in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
                candidate = value.get(key)
                if isinstance(candidate, self.torch.Tensor):
                    if key == "last_hidden_state" and candidate.ndim >= 3:
                        return candidate[:, 0]
                    return candidate

        if isinstance(value, (tuple, list)):
            for candidate in value:
                if isinstance(candidate, self.torch.Tensor):
                    return candidate
                resolved = self._feature_tensor(candidate)
                if isinstance(resolved, self.torch.Tensor):
                    return resolved

        raise TypeError(f"Unsupported CLIP feature output type: {type(value)!r}")

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.text_batch_size):
                batch = texts[start : start + self.text_batch_size]
                inputs = self.processor(text=batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                feats = self.model.get_text_features(**inputs)
                feats = self._feature_tensor(feats)
                feats = self.torch.nn.functional.normalize(feats, dim=-1)
                rows.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(rows, axis=0)

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        if not image_paths:
            return np.zeros((0, 0), dtype=np.float32)
        rows: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(image_paths), self.image_batch_size):
                batch_paths = image_paths[start : start + self.image_batch_size]
                images = []
                for path in batch_paths:
                    with Image.open(path) as handle:
                        images.append(handle.convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                feats = self.model.get_image_features(**inputs)
                feats = self._feature_tensor(feats)
                feats = self.torch.nn.functional.normalize(feats, dim=-1)
                rows.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(rows, axis=0)


class DiagonalGaussianCalibrator:
    def __init__(self) -> None:
        self.class_means: dict[int, np.ndarray] = {}
        self.class_vars: dict[int, np.ndarray] = {}
        self.class_priors: dict[int, float] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "DiagonalGaussianCalibrator":
        labels = np.asarray(labels, dtype=np.int32)
        features = np.asarray(features, dtype=np.float64)
        for klass in (0, 1):
            class_features = features[labels == klass]
            if len(class_features) == 0:
                raise ValueError(f"Missing class {klass} in calibration data.")
            self.class_means[klass] = class_features.mean(axis=0)
            self.class_vars[klass] = np.maximum(class_features.var(axis=0), 1e-6)
            self.class_priors[klass] = float(len(class_features) / len(features))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        logps: list[np.ndarray] = []
        for klass in (0, 1):
            mean = self.class_means[klass]
            var = self.class_vars[klass]
            prior = max(self.class_priors[klass], 1e-6)
            diff = features - mean
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var) + (diff * diff) / var, axis=1)
            logps.append(log_likelihood + np.log(prior))
        stacked = np.stack(logps, axis=1)
        stacked = stacked - stacked.max(axis=1, keepdims=True)
        probs = np.exp(stacked)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


def _prepare_target_manifest(runner: EvaluationRunner) -> pd.DataFrame:
    target_df = runner.target_manifest.copy()
    if "image_id" not in target_df.columns:
        raise KeyError("Target manifest must expose image_id.")

    target_df["image_id"] = target_df["image_id"].astype(str)
    if "image_path" in target_df.columns:
        target_df["image_path"] = target_df["image_path"].fillna("").astype(str)
    else:
        target_df["image_path"] = ""
    if "scene_family_label" in target_df.columns:
        target_df["scene_family_label"] = target_df["scene_family_label"].fillna("").astype(str)
    else:
        target_df["scene_family_label"] = ""

    target_df["resolved_image_path"] = target_df["image_path"].apply(lambda value: str(Path(value).expanduser().resolve()) if _safe_text(value) else "")
    target_df["image_exists"] = target_df["resolved_image_path"].apply(lambda value: bool(value) and Path(value).exists())
    target_df = target_df[target_df["image_exists"]].copy()
    if target_df.empty:
        raise FileNotFoundError("No target images could be resolved from the evaluation manifest.")
    return target_df.reset_index(drop=True)


def _prepare_flagged_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "attribute_name" not in df.columns:
        raise KeyError("Flagged CSV must include image_id and attribute_name.")

    df["image_id"] = df["image_id"].fillna("").astype(str)
    df["attribute_name"] = df["attribute_name"].fillna("").astype(str)
    if "grounding_label" in df.columns:
        df["grounding_label"] = df["grounding_label"].fillna("").astype(str)
    else:
        df["grounding_label"] = ""
    if "confirmation_status" in df.columns:
        df["confirmation_status"] = df["confirmation_status"].fillna("").astype(str)
    else:
        df["confirmation_status"] = ""
    if "scene_family_label" in df.columns:
        df["scene_family_label"] = df["scene_family_label"].fillna("").astype(str)
    else:
        df["scene_family_label"] = ""
    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].fillna("").astype(str)
    else:
        df["image_path"] = ""
    if "task_aligned_prediction" not in df.columns:
        matched_slot = df["matched_slot"].fillna("").astype(str) if "matched_slot" in df.columns else pd.Series("", index=df.index, dtype=str)
        task_field = df["task_field"].fillna("").astype(str) if "task_field" in df.columns else pd.Series("", index=df.index, dtype=str)
        semantic_supported = df["semantic_supported"].fillna(False).astype(bool) if "semantic_supported" in df.columns else pd.Series(False, index=df.index, dtype=bool)
        df["task_aligned_prediction"] = semantic_supported & matched_slot.eq(task_field) & task_field.ne("")
    if "annotation_grounded" not in df.columns:
        semantic_supported = df["semantic_supported"].fillna(False).astype(bool) if "semantic_supported" in df.columns else pd.Series(False, index=df.index, dtype=bool)
        df["annotation_grounded"] = semantic_supported & ~df["task_aligned_prediction"].astype(bool)
    df["setting_signature"] = df.apply(_setting_signature, axis=1)
    df["pair_key"] = df.apply(lambda row: _pair_key(row.get("image_id"), row.get("attribute_name")), axis=1)
    return df


def _extract_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "pair_key",
        "image_id",
        "attribute_name",
        "attribute_description",
        "positive_patterns",
        "scene_family_label",
        "image_path",
        "confirmation_status",
        "grounding_label",
        "task_aligned_prediction",
        "annotation_grounded",
    ]
    keep_cols = [col for col in cols if col in df.columns]
    return df[keep_cols].drop_duplicates(subset=["pair_key"], keep="first").reset_index(drop=True)


def _candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "grounding_label" in df.columns and df["grounding_label"].astype(str).str.len().gt(0).any():
        mask = df["grounding_label"].astype(str).eq(UNMAPPED_LABEL)
    else:
        mask = ~df["annotation_grounded"].astype(bool) & ~df["task_aligned_prediction"].astype(bool)
    if "confirmation_status" in df.columns:
        mask = mask & df["confirmation_status"].fillna("").astype(str).eq("confirmed")
    return df[mask].copy()


def _positive_calibration_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["annotation_grounded"].astype(bool) | df["task_aligned_prediction"].astype(bool)
    if "confirmation_status" in df.columns:
        mask = mask & df["confirmation_status"].fillna("").astype(str).eq("confirmed")
    return df[mask].copy()


def _score_row_features(
    prompt_texts: list[str],
    image_id: str,
    scene_family_label: str,
    *,
    image_index: dict[str, int],
    prompt_index: dict[str, int],
    score_matrix: np.ndarray,
    scene_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    if not prompt_texts or image_id not in image_index:
        return {
            "auto_best_prompt": "",
            "auto_similarity": np.nan,
            "auto_global_percentile": np.nan,
            "auto_scene_percentile": np.nan,
            "auto_global_margin": np.nan,
            "auto_scene_margin": np.nan,
            "auto_prompt_margin": np.nan,
        }

    row_idx = image_index[image_id]
    filtered_prompts = [prompt for prompt in prompt_texts if prompt in prompt_index]
    prompt_ids = [prompt_index[prompt] for prompt in filtered_prompts]
    if not prompt_ids:
        return {
            "auto_best_prompt": "",
            "auto_similarity": np.nan,
            "auto_global_percentile": np.nan,
            "auto_scene_percentile": np.nan,
            "auto_global_margin": np.nan,
            "auto_scene_margin": np.nan,
            "auto_prompt_margin": np.nan,
        }

    row_scores = score_matrix[row_idx, prompt_ids]
    best_local_idx = int(np.argmax(row_scores))
    best_prompt_idx = prompt_ids[best_local_idx]
    best_prompt = filtered_prompts[best_local_idx]
    best_score = float(row_scores[best_local_idx])
    second_score = float(np.partition(row_scores, -2)[-2]) if len(row_scores) >= 2 else np.nan

    prompt_scores = score_matrix[:, best_prompt_idx]
    global_percentile = float((prompt_scores <= best_score).mean())
    global_margin = float(best_score - float(np.median(prompt_scores)))

    scene_mask = scene_masks.get(_safe_text(scene_family_label))
    if scene_mask is not None and bool(scene_mask.any()):
        scene_scores = prompt_scores[scene_mask]
        scene_percentile = float((scene_scores <= best_score).mean())
        scene_margin = float(best_score - float(np.median(scene_scores)))
    else:
        scene_percentile = global_percentile
        scene_margin = global_margin

    return {
        "auto_best_prompt": best_prompt,
        "auto_similarity": best_score,
        "auto_global_percentile": global_percentile,
        "auto_scene_percentile": scene_percentile,
        "auto_global_margin": global_margin,
        "auto_scene_margin": scene_margin,
        "auto_prompt_margin": float(best_score - second_score) if not np.isnan(second_score) else np.nan,
    }


def _build_negative_pool(
    row: pd.Series,
    target_df: pd.DataFrame,
    runner: EvaluationRunner,
    *,
    cache: dict[str, list[str]],
) -> list[str]:
    pair_key = str(row["pair_key"])
    if pair_key in cache:
        return cache[pair_key]

    attribute_name = _safe_text(row.get("attribute_name"))
    extra_texts = []
    description = _safe_text(row.get("attribute_description"))
    if description:
        extra_texts.append(description)
    extra_texts.extend(_split_patterns(row.get("positive_patterns")))
    own_image_id = _safe_text(row.get("image_id"))
    own_scene = _safe_text(row.get("scene_family_label"))

    pool_other_scene: list[str] = []
    pool_any_scene: list[str] = []
    for _, target_row in target_df.iterrows():
        image_id = _safe_text(target_row.get("image_id"))
        if not image_id or image_id == own_image_id:
            continue
        decision = runner.support_checker.check(attribute_name, target_row.to_dict(), extra_texts=extra_texts)
        if decision.supported:
            continue
        pool_any_scene.append(image_id)
        if _safe_text(target_row.get("scene_family_label")) != own_scene:
            pool_other_scene.append(image_id)

    cache[pair_key] = pool_other_scene or pool_any_scene
    return cache[pair_key]


def _fit_calibrator(features: np.ndarray, labels: np.ndarray) -> tuple[Any, dict[str, float]]:
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration labels must include at least two classes.")

    model = DiagonalGaussianCalibrator().fit(features, labels)
    probs = model.predict_proba(features)[:, 1]
    metrics = {
        "train_accuracy": float(((probs >= 0.5).astype(np.int32) == labels).mean()),
        "train_auc": _binary_auc(labels, probs),
        "n_examples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
    }
    return model, metrics


def _summarize_verdicts(df: pd.DataFrame) -> dict[str, Any]:
    verdicts = df["auto_verdict"].fillna("").astype(str)
    total = int(len(df))
    valid = int(verdicts.eq(AUTO_VALID_LABEL).sum())
    hallucination = int(verdicts.eq(AUTO_HALLUCINATION_LABEL).sum())
    uncertain = int(verdicts.eq(AUTO_UNCERTAIN_LABEL).sum())
    missing = int(verdicts.eq(AUTO_MISSING_LABEL).sum())
    decided = valid + hallucination
    return {
        "total_unmapped_candidates": total,
        "auto_verified_extra": valid,
        "auto_likely_hallucination": hallucination,
        "auto_uncertain": uncertain,
        "auto_missing_image": missing,
        "auto_valid_rate_all": float(valid / total) if total else 0.0,
        "auto_hallucination_rate_all": float(hallucination / total) if total else 0.0,
        "auto_uncertain_rate_all": float(uncertain / total) if total else 0.0,
        "auto_missing_image_rate_all": float(missing / total) if total else 0.0,
        "auto_valid_rate_decided": float(valid / decided) if decided else 0.0,
        "auto_hallucination_rate_decided": float(hallucination / decided) if decided else 0.0,
        "decided_candidates": int(decided),
    }


def _print_summary(title: str, payload: dict[str, Any]) -> None:
    print(title)
    for key in [
        "total_unmapped_candidates",
        "auto_verified_extra",
        "auto_likely_hallucination",
        "auto_uncertain",
        "auto_missing_image",
        "auto_valid_rate_all",
        "auto_hallucination_rate_all",
        "auto_uncertain_rate_all",
        "auto_valid_rate_decided",
        "auto_hallucination_rate_decided",
        "decided_candidates",
    ]:
        if key in payload:
            value = payload[key]
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def _default_outputs_for_flagged(flagged_path: Path, output_dir: Path | None) -> tuple[Path, Path, Path]:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = flagged_path.stem
        return (
            output_dir / f"{stem}.unmapped_verification.csv",
            output_dir / f"{stem}.unmapped_verification.summary.json",
            output_dir / f"{stem}.unmapped_verification.by_setting.csv",
        )
    return (
        flagged_path.with_name(f"{flagged_path.stem}.unmapped_verification.csv"),
        flagged_path.with_name(f"{flagged_path.stem}.unmapped_verification.summary.json"),
        flagged_path.with_name(f"{flagged_path.stem}.unmapped_verification.by_setting.csv"),
    )


def _default_combined_outputs(base_dir: Path) -> tuple[Path, Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    return (
        base_dir / "unmapped_verification_all.csv",
        base_dir / "unmapped_verification_all.summary.json",
        base_dir / "unmapped_verification_all.by_setting.csv",
    )


def _resolve_flagged_paths(config_path: Path, flagged_values: list[str] | None, mode: str | None) -> list[Path]:
    paths: list[Path] = []
    if flagged_values:
        for value in flagged_values:
            paths.append(Path(value).expanduser().resolve())
    else:
        runner = EvaluationRunner(config_path, dry_run=True)
        modes = ["rq1", "rq2"] if mode in {"both", "", None} else [str(mode)]
        for item in modes:
            paths.append((runner.output_dir / f"flagged_attributes_{item}.csv").resolve())

    ordered: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)

    missing = [str(path) for path in ordered if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Flagged CSV does not exist: {missing}")
    return ordered


def _run_single_verification(
    *,
    config_path: Path,
    flagged_path: Path,
    args: argparse.Namespace,
    output_csv: Path,
    summary_json: Path,
    by_setting_csv: Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rng = random.Random(int(args.seed))

    runner = EvaluationRunner(config_path, dry_run=True)
    target_df = _prepare_target_manifest(runner)
    flagged_df = _prepare_flagged_df(flagged_path)
    flagged_df = flagged_df.merge(
        target_df[["image_id", "resolved_image_path", "scene_family_label"]].rename(columns={"scene_family_label": "target_scene_family_label"}),
        on="image_id",
        how="left",
    )
    flagged_df["scene_family_label"] = flagged_df["scene_family_label"].mask(
        flagged_df["scene_family_label"].fillna("").astype(str).eq(""),
        flagged_df["target_scene_family_label"].fillna("").astype(str),
    )
    flagged_df["resolved_image_path"] = flagged_df["resolved_image_path"].fillna("").astype(str)
    flagged_df.drop(columns=["target_scene_family_label"], inplace=True)

    candidate_rows = _candidate_rows(_extract_pair_rows(flagged_df))
    positive_rows = _positive_calibration_rows(_extract_pair_rows(flagged_df))
    if candidate_rows.empty:
        raise ValueError("No confirmed unmapped candidates were found in the flagged CSV.")
    if int(args.limit) > 0:
        candidate_rows = candidate_rows.head(int(args.limit)).copy()

    prompt_rows = pd.concat([candidate_rows, positive_rows], axis=0, ignore_index=True).drop_duplicates(subset=["pair_key"], keep="first")
    prompt_rows["prompt_texts"] = prompt_rows.apply(lambda row: _build_prompt_texts(row, args.max_prompts_per_row), axis=1)
    prompt_rows = prompt_rows[prompt_rows["prompt_texts"].map(bool)].copy()
    if prompt_rows.empty:
        raise ValueError("No prompt texts could be constructed from the candidate rows.")

    prompt_lookup = dict(zip(prompt_rows["pair_key"], prompt_rows["prompt_texts"]))
    all_prompts = sorted({prompt for prompts in prompt_lookup.values() for prompt in prompts})

    image_ids = target_df["image_id"].astype(str).tolist()
    image_paths = [Path(value) for value in target_df["resolved_image_path"].tolist()]
    image_index = {image_id: idx for idx, image_id in enumerate(image_ids)}
    scene_masks = {
        str(scene): target_df["scene_family_label"].astype(str).eq(str(scene)).to_numpy(dtype=bool)
        for scene in sorted(target_df["scene_family_label"].astype(str).unique().tolist())
    }

    verifier = ClipVerifier(
        args.model_name,
        device=args.device,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
    )
    image_embeddings = verifier.encode_images(image_paths)
    text_embeddings = verifier.encode_texts(all_prompts)
    score_matrix = image_embeddings @ text_embeddings.T
    prompt_index = {prompt: idx for idx, prompt in enumerate(all_prompts)}

    negative_pool_cache: dict[str, list[str]] = {}

    calibration_features: list[list[float]] = []
    calibration_labels: list[int] = []
    for _, row in positive_rows.iterrows():
        prompts = prompt_lookup.get(str(row["pair_key"])) or _build_prompt_texts(row, args.max_prompts_per_row)
        features = _score_row_features(
            prompts,
            _safe_text(row.get("image_id")),
            _safe_text(row.get("scene_family_label")),
            image_index=image_index,
            prompt_index=prompt_index,
            score_matrix=score_matrix,
            scene_masks=scene_masks,
        )
        if np.isnan(features["auto_similarity"]):
            continue
        calibration_features.append([
            float(features["auto_similarity"]),
            float(features["auto_global_percentile"]),
            float(features["auto_scene_percentile"]),
            float(features["auto_global_margin"]),
            float(features["auto_scene_margin"]),
            float(0.0 if np.isnan(features["auto_prompt_margin"]) else features["auto_prompt_margin"]),
        ])
        calibration_labels.append(1)

        negative_pool = _build_negative_pool(row, target_df, runner, cache=negative_pool_cache)
        if not negative_pool:
            continue
        sample_count = min(int(args.negatives_per_positive), len(negative_pool))
        sampled_ids = rng.sample(negative_pool, sample_count)
        for negative_image_id in sampled_ids:
            scene_label = _safe_text(target_df.loc[target_df["image_id"] == negative_image_id, "scene_family_label"].iloc[0])
            neg_features = _score_row_features(
                prompts,
                negative_image_id,
                scene_label,
                image_index=image_index,
                prompt_index=prompt_index,
                score_matrix=score_matrix,
                scene_masks=scene_masks,
            )
            if np.isnan(neg_features["auto_similarity"]):
                continue
            calibration_features.append([
                float(neg_features["auto_similarity"]),
                float(neg_features["auto_global_percentile"]),
                float(neg_features["auto_scene_percentile"]),
                float(neg_features["auto_global_margin"]),
                float(neg_features["auto_scene_margin"]),
                float(0.0 if np.isnan(neg_features["auto_prompt_margin"]) else neg_features["auto_prompt_margin"]),
            ])
            calibration_labels.append(0)

    if not calibration_features or len(set(calibration_labels)) < 2:
        raise ValueError("Could not build a usable calibration set from grounded rows and synthetic negatives.")

    calibrator, calibration_metrics = _fit_calibrator(
        np.asarray(calibration_features, dtype=np.float32),
        np.asarray(calibration_labels, dtype=np.int32),
    )

    candidate_outputs: list[dict[str, Any]] = []
    for _, row in candidate_rows.iterrows():
        prompts = prompt_lookup.get(str(row["pair_key"])) or _build_prompt_texts(row, args.max_prompts_per_row)
        image_id = _safe_text(row.get("image_id"))
        scene_label = _safe_text(row.get("scene_family_label"))
        features = _score_row_features(
            prompts,
            image_id,
            scene_label,
            image_index=image_index,
            prompt_index=prompt_index,
            score_matrix=score_matrix,
            scene_masks=scene_masks,
        )

        resolved_image_path = _safe_text(target_df.loc[target_df["image_id"] == image_id, "resolved_image_path"].iloc[0]) if image_id in set(image_ids) else ""
        out = row.to_dict()
        out["resolved_image_path"] = resolved_image_path
        out.update(features)

        if np.isnan(features["auto_similarity"]):
            out["auto_valid_probability"] = np.nan
            out["auto_verdict"] = AUTO_MISSING_LABEL
            candidate_outputs.append(out)
            continue

        feature_vector = np.asarray(
            [[
                float(features["auto_similarity"]),
                float(features["auto_global_percentile"]),
                float(features["auto_scene_percentile"]),
                float(features["auto_global_margin"]),
                float(features["auto_scene_margin"]),
                float(0.0 if np.isnan(features["auto_prompt_margin"]) else features["auto_prompt_margin"]),
            ]],
            dtype=np.float32,
        )
        valid_probability = float(calibrator.predict_proba(feature_vector)[0, 1])
        out["auto_valid_probability"] = valid_probability
        if valid_probability >= float(args.valid_threshold):
            out["auto_verdict"] = AUTO_VALID_LABEL
        elif valid_probability <= float(args.hallucination_threshold):
            out["auto_verdict"] = AUTO_HALLUCINATION_LABEL
        else:
            out["auto_verdict"] = AUTO_UNCERTAIN_LABEL
        candidate_outputs.append(out)

    verified_df = pd.DataFrame(candidate_outputs)
    if verified_df.empty:
        raise ValueError("No unmapped candidates were scored.")

    verified_df = verified_df.sort_values(
        by=["auto_valid_probability", "auto_similarity", "image_id", "attribute_name"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    overall_summary = _summarize_verdicts(verified_df)
    overall_summary.update(
        {
            "config_path": str(config_path),
            "flagged_csv": str(flagged_path),
            "output_csv": str(output_csv),
            "run_name": flagged_path.stem.removeprefix("flagged_attributes_"),
            "clip_model_name": str(args.model_name),
            "device": verifier.device,
            "valid_threshold": float(args.valid_threshold),
            "hallucination_threshold": float(args.hallucination_threshold),
            "calibration": calibration_metrics,
        }
    )

    group_cols = [col for col in ["model_name", "K", "llm_name", "setting_signature"] if col in verified_df.columns]
    if group_cols:
        setting_rows: list[dict[str, Any]] = []
        for keys, group in verified_df.groupby(group_cols, dropna=False, sort=True):
            key_values = list(keys) if isinstance(keys, tuple) else [keys]
            row = {col: value for col, value in zip(group_cols, key_values)}
            row.update(_summarize_verdicts(group))
            row["mean_auto_valid_probability"] = float(pd.to_numeric(group["auto_valid_probability"], errors="coerce").fillna(0.0).mean())
            setting_rows.append(row)
        by_setting_df = pd.DataFrame(setting_rows).sort_values(group_cols, kind="stable").reset_index(drop=True)
    else:
        by_setting_df = pd.DataFrame()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    verified_df.to_csv(output_csv, index=False)
    if not by_setting_df.empty:
        by_setting_df.to_csv(by_setting_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, indent=2, ensure_ascii=False)

    _print_summary("Unmapped Candidate Verification", overall_summary)
    print(f"row_output: {output_csv}")
    if not by_setting_df.empty:
        print(f"by_setting_output: {by_setting_csv}")
    print(f"summary_output: {summary_json}")
    return verified_df, overall_summary, by_setting_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc image-grounded verification for unmapped candidates in flagged_attributes CSVs. "
            "This does not rerun Stage 1/2 or query the original attack LLM."
        )
    )
    parser.add_argument("--config", required=True, help="Evaluation config, for example configs/eval.yaml")
    parser.add_argument(
        "--flagged-csv",
        action="append",
        help="Path to flagged_attributes_rq1.csv or flagged_attributes_rq2.csv. Repeat to pass more than one file.",
    )
    parser.add_argument(
        "--mode",
        choices=["rq1", "rq2", "both"],
        help="If --flagged-csv is omitted, auto-select output/flagged_attributes_<mode>.csv from the evaluation output directory.",
    )
    parser.add_argument("--output-csv", help="Single-run output CSV, or combined output CSV when multiple inputs are processed")
    parser.add_argument("--summary-json", help="Single-run summary JSON, or combined summary JSON when multiple inputs are processed")
    parser.add_argument("--by-setting-csv", help="Single-run per-setting CSV, or combined per-setting CSV when multiple inputs are processed")
    parser.add_argument("--output-dir", help="Optional directory for per-run outputs when processing one or more flagged CSVs")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="Hugging Face CLIP model name")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-prompts-per-row", type=int, default=6)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--valid-threshold", type=float, default=0.70)
    parser.add_argument("--hallucination-threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on unmapped rows for quick smoke tests")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    flagged_paths = _resolve_flagged_paths(config_path, args.flagged_csv, args.mode)
    multi_run = len(flagged_paths) > 1

    all_verified: list[pd.DataFrame] = []
    all_by_setting: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []

    for flagged_path in flagged_paths:
        if not multi_run:
            default_output_csv, default_summary_json, default_by_setting_csv = _default_outputs_for_flagged(flagged_path, output_dir)
            output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else default_output_csv
            summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else default_summary_json
            by_setting_csv = Path(args.by_setting_csv).expanduser().resolve() if args.by_setting_csv else default_by_setting_csv
        else:
            output_csv, summary_json, by_setting_csv = _default_outputs_for_flagged(flagged_path, output_dir)

        verified_df, overall_summary, by_setting_df = _run_single_verification(
            config_path=config_path,
            flagged_path=flagged_path,
            args=args,
            output_csv=output_csv,
            summary_json=summary_json,
            by_setting_csv=by_setting_csv,
        )
        verified_df = verified_df.copy()
        verified_df["run_name"] = overall_summary.get("run_name", flagged_path.stem)
        verified_df["source_flagged_csv"] = str(flagged_path)
        all_verified.append(verified_df)
        if not by_setting_df.empty:
            setting_copy = by_setting_df.copy()
            setting_copy["run_name"] = overall_summary.get("run_name", flagged_path.stem)
            all_by_setting.append(setting_copy)
        all_summaries.append(overall_summary)

    if multi_run and all_verified:
        combined_verified = pd.concat(all_verified, axis=0, ignore_index=True)
        combined_summary = _summarize_verdicts(combined_verified)
        combined_summary.update(
            {
                "config_path": str(config_path),
                "flagged_csvs": [str(path) for path in flagged_paths],
                "clip_model_name": str(args.model_name),
                "valid_threshold": float(args.valid_threshold),
                "hallucination_threshold": float(args.hallucination_threshold),
                "per_run_summaries": all_summaries,
            }
        )

        combined_base_dir = output_dir or flagged_paths[0].parent
        default_output_csv, default_summary_json, default_by_setting_csv = _default_combined_outputs(combined_base_dir)
        combined_output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else default_output_csv
        combined_summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else default_summary_json
        combined_by_setting_csv = Path(args.by_setting_csv).expanduser().resolve() if args.by_setting_csv else default_by_setting_csv

        combined_output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined_summary_json.parent.mkdir(parents=True, exist_ok=True)
        combined_verified.to_csv(combined_output_csv, index=False)
        with open(combined_summary_json, "w", encoding="utf-8") as handle:
            json.dump(combined_summary, handle, indent=2, ensure_ascii=False)
        if all_by_setting:
            combined_by_setting = pd.concat(all_by_setting, axis=0, ignore_index=True)
            combined_by_setting.to_csv(combined_by_setting_csv, index=False)
        _print_summary("Combined Unmapped Candidate Verification", combined_summary)
        print(f"combined_row_output: {combined_output_csv}")
        if all_by_setting:
            print(f"combined_by_setting_output: {combined_by_setting_csv}")
        print(f"combined_summary_output: {combined_summary_json}")


if __name__ == "__main__":
    main()
