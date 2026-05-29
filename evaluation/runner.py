from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from audit.config import infer_project_root_from_config, load_config, normalize_manifest_df, resolve
from audit.stage1_semantics import snakeify
from evaluation.llm_adapter import LLMSpec, build_llm_client
from evaluation.metrics import (
    AttributeAudit,
    EmbeddingAuditRecord,
    compute_excess_to_task_ratio,
    exceeds_ratio_threshold,
    summarize_audits,
)
from evaluation.semantic_support import (
    SemanticSupportChecker,
    build_support_text_candidates,
)


DEFAULT_RQ1_MODELS = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_base_patch16_224",
]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number):
            return default
        return number
    except Exception:
        return default


def _parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if _safe_text(item)]
    text = _safe_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if _safe_text(item)]
    return [text]


def _available_recovery_fields(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    field_columns = {
        "primary": ("primary_label", "primary_object"),
        "secondary": ("secondary_label", "secondary_object"),
        "ternary": ("ternary_label", "ternary_object"),
        "background": ("background_label", "background_scene"),
    }
    available: list[str] = []
    for field_name, columns in field_columns.items():
        if any(_safe_text(metadata.get(column)) for column in columns):
            available.append(field_name)
    return tuple(available)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        rows = _load_rows(path)
        return pd.DataFrame(rows)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            rows = payload.get("records") if isinstance(payload.get("records"), list) else [payload]
            return pd.DataFrame(rows)
    raise ValueError(f"Unsupported dataset format: {path}")


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any] | None) -> dict[str, Any]:
    if not updates:
        return base
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


@dataclass(frozen=True)
class EvaluationSetting:
    mode: str
    label: str
    model_name: str
    k: int
    llm_name: str
    llm_spec: LLMSpec

    @property
    def slug(self) -> str:
        return snakeify(f"{self.mode}_{self.model_name}_k_{self.k}_{self.llm_name}")


@dataclass(frozen=True)
class EmbeddingBundle:
    image_ids: tuple[str, ...]
    embeddings: np.ndarray
    pred_labels: np.ndarray | None = None
    logits: np.ndarray | None = None

    def has_pred_labels(self) -> bool:
        return self.pred_labels is not None and len(self.pred_labels) == len(self.image_ids)

    def has_logits(self) -> bool:
        return self.logits is not None and self.logits.ndim == 2 and self.logits.shape[0] == len(self.image_ids)


class EvaluationRunner:
    def __init__(
        self,
        config_path: str | Path,
        *,
        dataset_path: str | None = None,
        embeddings_dir: str | None = None,
        output_dir: str | None = None,
        max_samples: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        model_override: str | None = None,
        k_override: int | None = None,
        llm_override: str | None = None,
        tau_excess_kl: float | None = None,
        tau_ratio: float | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.project_root = infer_project_root_from_config(self.config_path)
        self.eval_cfg = load_config(self.config_path)

        base_cfg_rel = _safe_text(self.eval_cfg.get("base_config")) or "configs/mvp.yaml"
        self.base_config_path = resolve(base_cfg_rel, self.project_root)
        self.base_cfg = load_config(self.base_config_path)

        self.seed = int(seed if seed is not None else self.eval_cfg.get("seed", self.base_cfg.get("seed", 123)))
        self.device = _safe_text(device) or _safe_text((self.eval_cfg.get("runtime") or {}).get("device")) or _safe_text((self.base_cfg.get("embeddings") or {}).get("device")) or "auto"
        self.output_dir = resolve(
            output_dir or _safe_text(self.eval_cfg.get("output_dir")) or "output",
            self.project_root,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "eval_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_dir = Path(embeddings_dir).expanduser().resolve() if embeddings_dir else None
        self.max_samples = max_samples
        self.model_override = model_override
        self.k_override = k_override
        self.llm_override = llm_override
        self.dry_run = bool(dry_run or self.eval_cfg.get("dry_run", False))
        self.keep_intermediates = bool(self.eval_cfg.get("keep_intermediates", False))
        self.preserve_stage1_artifacts = bool(self.eval_cfg.get("preserve_stage1_artifacts", True))
        self.label_split = _safe_text(self.eval_cfg.get("label_split")) or "corpus"
        self.stage1_field = _safe_text(self.eval_cfg.get("stage1_field")) or _safe_text(((self.base_cfg.get("operationalization") or {}).get("from_stage1_field"))) or "shared_attributes"

        thresholds_cfg = self.eval_cfg.get("thresholds") or {}
        self.tau_excess_kl = float(tau_excess_kl if tau_excess_kl is not None else thresholds_cfg.get("tau_excess_kl", 0.01))
        self.tau_ratio = float(tau_ratio if tau_ratio is not None else thresholds_cfg.get("tau_ratio", 0.10))

        self.embeddings_cfg = self.eval_cfg.get("embeddings") or {}
        self.baseline_mode = _safe_text(((self.base_cfg.get("confirmation") or {}).get("baseline_mode"))) or "Y"
        self.need_pred_labels = self.baseline_mode.upper() == "Y"
        self.need_logits = self.baseline_mode.upper() == "O"

        semantic_cfg = self.eval_cfg.get("semantic_support") or {}
        synonym_map_path = semantic_cfg.get("synonym_map_path")
        resolved_synonyms = resolve(synonym_map_path, self.project_root) if synonym_map_path else None
        self.support_checker = SemanticSupportChecker.from_path(
            resolved_synonyms,
            minimum_score=float(semantic_cfg.get("minimum_score", 0.75)),
            stopwords=semantic_cfg.get("stopwords"),
        )

        self.llm_specs = self._load_llm_specs()
        self.support_manifest = self._load_support_manifest()
        self.target_manifest = self._load_target_manifest(dataset_path)
        self.target_by_id = self.target_manifest.set_index("image_id", drop=False)
        self._bundle_cache: dict[tuple[str, str], EmbeddingBundle] = {}

    def _load_llm_specs(self) -> dict[str, LLMSpec]:
        specs: dict[str, LLMSpec] = {}
        for item in self.eval_cfg.get("llms") or []:
            spec = LLMSpec.from_mapping(item)
            if spec.name:
                specs[spec.name] = spec
        if not specs:
            raise ValueError("No llms were configured in the evaluation config.")
        return specs

    def _dataset_cfg(self) -> dict[str, Any]:
        return dict(self.eval_cfg.get("dataset") or {})

    def _load_support_manifest(self) -> pd.DataFrame:
        manifest_path = resolve((self.base_cfg.get("paths") or {})["manifest_csv"], self.project_root)
        manifest = pd.read_csv(manifest_path)
        manifest = normalize_manifest_df(manifest, self.base_cfg, self.project_root)
        # Evaluation always uses the full DCI manifest as the auxiliary corpus.
        # Synthetic/generated images are only valid as audited target inputs.
        manifest = manifest.copy()
        manifest["image_id"] = manifest["image_id"].astype(str)
        return manifest.reset_index(drop=True)

    def _load_generated_targets_manifest(self) -> pd.DataFrame:
        target_cfg = self.base_cfg.get("generated_targets") or {}
        manifest_cfg = target_cfg.get("manifest_csv")
        if not manifest_cfg:
            raise ValueError("base_config.generated_targets.manifest_csv is not configured.")

        manifest_path = resolve(manifest_cfg, self.project_root)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Generated target manifest not found: {manifest_path}")

        image_root = _safe_text(target_cfg.get("image_root")) or str(manifest_path.parent)
        temp_cfg = copy.deepcopy(self.base_cfg)
        temp_cfg.setdefault("dataset", {})
        temp_cfg.setdefault("report", {})
        temp_cfg["dataset"]["photos_root"] = image_root
        temp_cfg["report"]["markdown_image_root"] = image_root

        df = pd.read_csv(manifest_path)
        df = normalize_manifest_df(df, temp_cfg, self.project_root)
        sort_cols = [col for col in ("scene_family", "primary_label", "secondary_label", "ternary_label", "background_label", "seed") if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        return df

    def _load_target_manifest(self, dataset_path_override: str | None) -> pd.DataFrame:
        dataset_cfg = self._dataset_cfg()
        dataset_source = _safe_text(dataset_cfg.get("source")) or "generated_targets"
        if dataset_path_override:
            dataset_source = "explicit"

        if dataset_source == "generated_targets":
            df = self._load_generated_targets_manifest()
        else:
            raw_path = dataset_path_override or _safe_text(dataset_cfg.get("path"))
            if not raw_path:
                raise ValueError("No evaluation dataset path was provided.")
            dataset_path = resolve(raw_path, self.project_root)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

            image_root = _safe_text(dataset_cfg.get("image_root")) or str(dataset_path.parent)
            temp_cfg = copy.deepcopy(self.base_cfg)
            temp_cfg.setdefault("dataset", {})
            temp_cfg.setdefault("report", {})
            temp_cfg["dataset"]["photos_root"] = image_root
            temp_cfg["report"]["markdown_image_root"] = image_root

            df = _load_table(dataset_path)
            df = normalize_manifest_df(df, temp_cfg, self.project_root)

        if "image_id" not in df.columns:
            raise KeyError("Target dataset must resolve an image_id column.")
        df["image_id"] = df["image_id"].astype(str)

        task_col = (
            _safe_text(dataset_cfg.get("task_label_column"))
            or ("task_label" if "task_label" in df.columns else "")
            or ("task_attribute" if "task_attribute" in df.columns else "")
            or ("scene_family_label" if "scene_family_label" in df.columns else "")
            or ("primary_label" if "primary_label" in df.columns else "")
        )
        df["task_label"] = df[task_col].fillna("").astype(str) if task_col else ""

        sort_cols = [col for col in ("scene_family", "primary_label", "secondary_label", "ternary_label", "background_label", "seed") if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

        if self.max_samples is not None and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=self.seed).sort_values("image_id", kind="stable").reset_index(drop=True)
        return df

    def _resolve_template_path(self, template: str | None, *, model_name: str) -> Path | None:
        if not template:
            return None
        dataset_stem = snakeify(self.target_manifest["image_id"].iloc[0] if not self.target_manifest.empty else "targets")
        rendered = str(template).format(model=model_name, dataset_stem=dataset_stem)
        return resolve(rendered, self.project_root)

    def _support_cache_path(self, model_name: str) -> Path:
        if self.embeddings_dir is not None:
            return (self.embeddings_dir / model_name / "support_embeddings.npz").resolve()
        templated = self._resolve_template_path(self.embeddings_cfg.get("support_npz_template"), model_name=model_name)
        if templated is not None:
            return templated
        return (self.cache_dir / "support" / f"{model_name}.npz").resolve()

    def _target_cache_path(self, model_name: str) -> Path:
        if self.embeddings_dir is not None:
            return (self.embeddings_dir / model_name / "target_embeddings.npz").resolve()
        templated = self._resolve_template_path(self.embeddings_cfg.get("target_npz_template"), model_name=model_name)
        if templated is not None:
            return templated
        return (self.cache_dir / "targets" / f"{model_name}.npz").resolve()

    def _run_project_script(self, script_relpath: str, args: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, script_relpath, *args]
        proc = subprocess.run(
            cmd,
            cwd=self.project_root,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{script_relpath} failed.\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )
        return proc

    def _load_embedding_bundle(self, npz_path: Path, expected_ids: Iterable[str]) -> EmbeddingBundle:
        npz = np.load(npz_path, allow_pickle=True)
        ids = np.asarray(npz["image_ids"]).astype(str)
        id_to_pos = {image_id: idx for idx, image_id in enumerate(ids)}
        ordered_ids = [str(image_id) for image_id in expected_ids]
        missing = [image_id for image_id in ordered_ids if image_id not in id_to_pos]
        if missing:
            raise ValueError(f"Embeddings at {npz_path} are missing {len(missing)} expected ids; first few: {missing[:5]}")
        positions = [id_to_pos[image_id] for image_id in ordered_ids]

        pred_labels: np.ndarray | None = None
        logits: np.ndarray | None = None
        if "pred_labels" in npz.files:
            pred_labels = np.asarray(npz["pred_labels"])[positions]
        if "logits" in npz.files:
            logits = np.asarray(npz["logits"])[positions]
        return EmbeddingBundle(
            image_ids=tuple(ordered_ids),
            embeddings=np.asarray(npz["embeddings"], dtype=np.float32)[positions],
            pred_labels=pred_labels,
            logits=logits,
        )

    def _bundle_has_required_outputs(self, bundle: EmbeddingBundle) -> bool:
        if self.need_pred_labels and not bundle.has_pred_labels():
            return False
        if self.need_logits and not bundle.has_logits():
            return False
        return True

    def _compute_embeddings_for_manifest(self, manifest_df: pd.DataFrame, model_name: str, out_npz: Path, stem: str) -> None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = out_npz.with_name(f"{out_npz.stem}__{stem}_manifest.csv")
        config_path = out_npz.with_name(f"{out_npz.stem}__{stem}_config.yaml")

        temp_cfg = copy.deepcopy(self.base_cfg)
        temp_cfg.setdefault("paths", {})
        temp_cfg.setdefault("embeddings", {})
        temp_cfg["paths"]["manifest_csv"] = str(manifest_path)
        temp_cfg["paths"]["embeddings_npz"] = str(out_npz)
        temp_cfg["paths"].pop("task_outputs_csv", None)
        temp_cfg["embeddings"]["model_name"] = model_name
        temp_cfg["embeddings"]["device"] = self.device
        temp_cfg["embeddings"]["save_pred_labels"] = bool(self.need_pred_labels or temp_cfg["embeddings"].get("save_pred_labels", False))
        temp_cfg["embeddings"]["save_logits"] = bool(self.need_logits or temp_cfg["embeddings"].get("save_logits", False))

        manifest_df.to_csv(manifest_path, index=False)
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(temp_cfg, handle, sort_keys=False)

        self._run_project_script(
            "scripts/compute_embeddings.py",
            ["--config", str(config_path)],
        )

    def _ensure_support_bundle(self, model_name: str) -> EmbeddingBundle:
        cache_key = ("support", model_name)
        if cache_key in self._bundle_cache:
            return self._bundle_cache[cache_key]

        expected_ids = self.support_manifest["image_id"].astype(str).tolist()
        base_emb_path = resolve((self.base_cfg.get("paths") or {}).get("embeddings_npz"), self.project_root)
        base_model_name = _safe_text(((self.base_cfg.get("embeddings") or {}).get("model_name")))
        reuse_base = bool(self.embeddings_cfg.get("reuse_base_manifest_embeddings_when_available", True))

        if reuse_base and base_emb_path.exists() and base_model_name == model_name:
            bundle = self._load_embedding_bundle(base_emb_path, expected_ids)
            if self._bundle_has_required_outputs(bundle):
                self._bundle_cache[cache_key] = bundle
                return bundle

        out_npz = self._support_cache_path(model_name)
        if out_npz.exists():
            bundle = self._load_embedding_bundle(out_npz, expected_ids)
            if self._bundle_has_required_outputs(bundle):
                self._bundle_cache[cache_key] = bundle
                return bundle

        if not bool(self.embeddings_cfg.get("compute_support_if_missing", True)):
            raise FileNotFoundError(f"Support embeddings are unavailable for model={model_name}: {out_npz}")

        self._compute_embeddings_for_manifest(self.support_manifest, model_name, out_npz, "support")
        bundle = self._load_embedding_bundle(out_npz, expected_ids)
        self._bundle_cache[cache_key] = bundle
        return bundle

    def _ensure_target_bundle(self, model_name: str) -> EmbeddingBundle:
        cache_key = ("target", model_name)
        if cache_key in self._bundle_cache:
            return self._bundle_cache[cache_key]

        expected_ids = self.target_manifest["image_id"].astype(str).tolist()
        out_npz = self._target_cache_path(model_name)
        if out_npz.exists():
            bundle = self._load_embedding_bundle(out_npz, expected_ids)
            if self._bundle_has_required_outputs(bundle):
                self._bundle_cache[cache_key] = bundle
                return bundle

        if not bool(self.embeddings_cfg.get("compute_target_if_missing", True)):
            raise FileNotFoundError(f"Target embeddings are unavailable for model={model_name}: {out_npz}")

        self._compute_embeddings_for_manifest(self.target_manifest, model_name, out_npz, "targets")
        bundle = self._load_embedding_bundle(out_npz, expected_ids)
        self._bundle_cache[cache_key] = bundle
        return bundle

    def _build_task_outputs_df(self, support: EmbeddingBundle, target: EmbeddingBundle) -> pd.DataFrame | None:
        rows: list[pd.DataFrame] = []
        for bundle in (support, target):
            frame = pd.DataFrame({"image_id": list(bundle.image_ids)})
            if bundle.has_pred_labels():
                frame["pred_label"] = bundle.pred_labels
            if bundle.has_logits():
                logits = np.asarray(bundle.logits, dtype=np.float32)
                for idx in range(logits.shape[1]):
                    frame[f"logit_{idx}"] = logits[:, idx]
            rows.append(frame)
        combined = pd.concat(rows, ignore_index=True)
        if len([col for col in combined.columns if col != "image_id"]) == 0:
            return None
        return combined

    def _build_setting_config(
        self,
        setting: EvaluationSetting,
        run_dir: Path,
        *,
        manifest_path: Path,
        embeddings_path: Path,
        splits_path: Path,
        index_path: Path,
        task_outputs_path: Path | None,
    ) -> Path:
        stage1_jsonl = run_dir / "stage1_discovery.jsonl"
        labels_csv = run_dir / "secret_labels.csv"
        confirmations_csv = run_dir / "target_confirmations.csv"
        summary_csv = run_dir / "confirmation_summary.csv"
        config_path = run_dir / "eval_run_config.yaml"

        temp_cfg = copy.deepcopy(self.base_cfg)
        temp_cfg.setdefault("paths", {})
        temp_cfg["paths"]["manifest_csv"] = str(manifest_path)
        temp_cfg["paths"]["embeddings_npz"] = str(embeddings_path)
        temp_cfg["paths"]["splits_csv"] = str(splits_path)
        temp_cfg["paths"]["index_npz"] = str(index_path)
        temp_cfg["paths"]["stage1_jsonl"] = str(stage1_jsonl)
        temp_cfg["paths"]["labels_csv"] = str(labels_csv)
        temp_cfg["paths"]["target_confirmations_csv"] = str(confirmations_csv)
        temp_cfg["paths"]["confirmation_summary_csv"] = str(summary_csv)
        if task_outputs_path is not None:
            temp_cfg["paths"]["task_outputs_csv"] = str(task_outputs_path)
        else:
            temp_cfg["paths"].pop("task_outputs_csv", None)

        temp_cfg.setdefault("embeddings", {})
        temp_cfg["embeddings"]["model_name"] = setting.model_name
        temp_cfg["embeddings"]["device"] = self.device

        temp_cfg.setdefault("audit", {})
        temp_cfg["audit"]["target_split"] = "audited"

        temp_cfg.setdefault("discovery", {})
        temp_cfg["discovery"]["target_split"] = "audited"
        temp_cfg.setdefault("retrieval", {})
        temp_cfg["retrieval"]["top_k"] = int(setting.k)
        temp_cfg["discovery"].setdefault("offline_llm", {})
        temp_cfg["discovery"]["offline_llm"]["only_auditable"] = False

        temp_cfg.setdefault("operationalization", {})
        temp_cfg["operationalization"]["source"] = "stage1"
        temp_cfg["operationalization"]["stage1_path"] = str(stage1_jsonl)
        temp_cfg["operationalization"]["from_stage1_field"] = self.stage1_field

        temp_cfg.setdefault("confirmation", {})
        temp_cfg["confirmation"]["target_split"] = "audited"
        temp_cfg["confirmation"]["min_excess_kl"] = float(self.tau_excess_kl)
        temp_cfg["evaluation_setting"] = {
            "mode": setting.mode,
            "label": setting.label,
            "model_name": setting.model_name,
            "k": int(setting.k),
            "llm_name": setting.llm_name,
        }

        temp_cfg = _deep_update(temp_cfg, self.eval_cfg.get("base_config_overrides"))
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(temp_cfg, handle, sort_keys=False)
        return config_path

    def _mock_stage1_response(self, manifest_row: Mapping[str, Any]) -> str:
        raw_candidates = []
        raw_candidates.extend(_parse_json_list(manifest_row.get("candidate_secrets")))
        raw_candidates.extend(_parse_json_list(manifest_row.get("common_phrases")))

        normalized_pairs: list[tuple[str, str]] = []
        seen_names: set[str] = set()
        for raw in raw_candidates:
            text = _safe_text(raw)
            if not text:
                continue
            match = None
            if "'" in text:
                quoted = [part for part in text.split("'") if _safe_text(part)]
                if len(quoted) >= 2:
                    match = quoted[1]
            phrase = _safe_text(match or text)
            phrase = phrase.replace("presence of", "").replace("content related to", "").replace("context", "").strip()
            name = snakeify(phrase)
            if not name or name in {"image", "photo", "picture"} or name in seen_names:
                continue
            seen_names.add(name)
            normalized_pairs.append((name, phrase))
            if len(normalized_pairs) >= 4:
                break

        if not normalized_pairs:
            normalized_pairs = [("retrieved_context", "retrieved context")]

        shared_attributes = []
        excess_secrets = []
        for idx, (name, phrase) in enumerate(normalized_pairs, start=1):
            item = {
                "name": name,
                "description": f"Dry-run heuristic attribute derived from retrieved captions: {phrase}.",
                "evidence": [phrase],
                "specificity": 0.65,
                "relevance": 0.70,
                "privacy_relevance": 0.40,
                "task_relevance": 0.20,
                "positive_patterns": [phrase],
                "negative_patterns": [],
            }
            shared_attributes.append(item)
            if idx <= 2:
                excess_secrets.append({
                    "name": name,
                    "description": item["description"],
                    "why_excess": "Dry-run fallback response.",
                    "specificity": 0.65,
                    "privacy_relevance": 0.40,
                    "task_relevance": 0.20,
                    "positive_patterns": [phrase],
                    "negative_patterns": [],
                })

        payload = {
            "summary": "Dry-run response generated without remote LLM calls.",
            "shared_attributes": shared_attributes,
            "predicted_label_covers": _safe_text(manifest_row.get("predicted_label")),
            "excess_secrets": excess_secrets,
            "rejected_generic_terms": [],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _materialize_stage1_responses(self, setting: EvaluationSetting, work_dir: Path) -> None:
        manifest_path = work_dir / "prompts" / "prompt_manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Prompt manifest not found: {manifest_path}")

        rows = _load_rows(manifest_path)
        responses_dir = work_dir / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)

        fallback_to_mock = self.dry_run or bool(
            next(
                (
                    item.get("fallback_to_mock_on_error")
                    for item in (self.eval_cfg.get("llms") or [])
                    if _safe_text(item.get("name")) == setting.llm_name
                ),
                False,
            )
        )

        client = None
        if not self.dry_run:
            try:
                client = build_llm_client(setting.llm_spec)
            except Exception:
                if not fallback_to_mock:
                    raise
                client = None

        for row in rows:
            if not bool(row.get("should_export_prompt", False)):
                continue
            prompt_rel = _safe_text(row.get("prompt_txt"))
            response_rel = _safe_text(row.get("recommended_response")) or (
                f"responses/{snakeify(_safe_text(row.get('image_id')))}.response.txt"
            )
            prompt_path = work_dir / prompt_rel
            response_path = work_dir / response_rel
            response_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_text = prompt_path.read_text(encoding="utf-8")
            try:
                response_text = self._mock_stage1_response(row) if (self.dry_run or client is None) else client.generate(prompt_text)
            except Exception as exc:
                if not fallback_to_mock:
                    raise
                response_text = self._mock_stage1_response(row)
                response_text += f"\n\n<!-- fallback_reason: {type(exc).__name__}: {exc} -->\n"
            response_path.write_text(response_text + "\n", encoding="utf-8")

    def _prepare_run_dir(self, setting: EvaluationSetting) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
        if self.keep_intermediates or self.preserve_stage1_artifacts:
            run_dir = Path(tempfile.mkdtemp(prefix=f"{setting.slug}_", dir=self.output_dir))
            return run_dir, None
        tmp = tempfile.TemporaryDirectory(prefix=f"{setting.slug}_", dir=self.output_dir)
        return Path(tmp.name), tmp

    def run_setting(self, setting: EvaluationSetting) -> tuple[dict[str, Any], list[EmbeddingAuditRecord]]:
        run_dir, tmp_ctx = self._prepare_run_dir(setting)
        try:
            support_bundle = self._ensure_support_bundle(setting.model_name)
            target_bundle = self._ensure_target_bundle(setting.model_name)

            combined_manifest = pd.concat([self.support_manifest, self.target_manifest], ignore_index=True)
            if combined_manifest["image_id"].astype(str).duplicated().any():
                dupes = combined_manifest.loc[combined_manifest["image_id"].astype(str).duplicated(), "image_id"].astype(str).tolist()
                raise ValueError(f"Duplicate image_ids in combined manifest: {dupes[:5]}")

            manifest_path = run_dir / "manifest.csv"
            splits_path = run_dir / "splits.csv"
            embeddings_path = run_dir / "embeddings.npz"
            index_path = run_dir / "corpus_index.npz"
            task_outputs_path = run_dir / "task_outputs.csv"

            combined_manifest.to_csv(manifest_path, index=False)

            splits_df = pd.DataFrame(
                {
                    "image_id": [*support_bundle.image_ids, *target_bundle.image_ids],
                    "split": ["corpus"] * len(support_bundle.image_ids) + ["audited"] * len(target_bundle.image_ids),
                }
            )
            splits_df.to_csv(splits_path, index=False)

            np.savez_compressed(
                embeddings_path,
                image_ids=np.array([*support_bundle.image_ids, *target_bundle.image_ids]),
                embeddings=np.vstack([support_bundle.embeddings.astype(np.float32), target_bundle.embeddings.astype(np.float32)]),
            )
            np.savez_compressed(
                index_path,
                corpus_ids=np.array(list(support_bundle.image_ids)),
                corpus_embeddings=support_bundle.embeddings.astype(np.float32),
            )

            task_outputs_df = self._build_task_outputs_df(support_bundle, target_bundle)
            resolved_task_outputs_path = task_outputs_path if task_outputs_df is not None else None
            if task_outputs_df is not None:
                task_outputs_df.to_csv(task_outputs_path, index=False)

            config_path = self._build_setting_config(
                setting,
                run_dir,
                manifest_path=manifest_path,
                embeddings_path=embeddings_path,
                splits_path=splits_path,
                index_path=index_path,
                task_outputs_path=resolved_task_outputs_path,
            )
            stage1_work_dir = run_dir / "stage1_offline"

            self._run_project_script(
                "scripts/stage1_discover.py",
                [
                    "--config", str(config_path),
                    "--action", "export_prompts",
                    "--work-dir", str(stage1_work_dir),
                    "--target-split", "audited",
                ],
            )
            self._materialize_stage1_responses(setting, stage1_work_dir)
            self._run_project_script(
                "scripts/stage1_discover.py",
                [
                    "--config", str(config_path),
                    "--action", "ingest_responses",
                    "--work-dir", str(stage1_work_dir),
                ],
            )
            self._run_project_script(
                "scripts/build_labels.py",
                [
                    "--config", str(config_path),
                    "--label-split", self.label_split,
                ],
            )
            self._run_project_script(
                "scripts/stage2_confirm_excess_leakage.py",
                [
                    "--config", str(config_path),
                    "--target-split", "audited",
                ],
            )

            confirmations_path = run_dir / "target_confirmations.csv"
            confirmation_df = pd.read_csv(confirmations_path) if confirmations_path.exists() else pd.DataFrame()
            if not confirmation_df.empty and "target_image_id" in confirmation_df.columns:
                confirmation_df["target_image_id"] = confirmation_df["target_image_id"].astype(str)
            audits = self._score_audits(setting, confirmation_df)
            summary = summarize_audits(
                audits,
                label_key="Embedding Model" if setting.mode == "rq1" else "Setting",
                label_value=setting.label,
                model_name=setting.model_name,
                k=setting.k,
                llm_name=setting.llm_name,
            )
            return summary, audits
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    def _score_audits(self, setting: EvaluationSetting, confirmation_df: pd.DataFrame) -> list[EmbeddingAuditRecord]:
        audits: list[EmbeddingAuditRecord] = []
        grouped = (
            dict(tuple(confirmation_df.groupby("target_image_id", sort=False)))
            if not confirmation_df.empty and "target_image_id" in confirmation_df.columns
            else {}
        )

        for _, target_row in self.target_manifest.iterrows():
            image_id = str(target_row["image_id"])
            task_label = _safe_text(target_row.get("task_label"))
            available_fields = _available_recovery_fields(target_row.to_dict())
            sub = grouped.get(image_id)
            flagged: list[AttributeAudit] = []
            if sub is not None and not sub.empty:
                for _, row in sub.iterrows():
                    excess_kl = _safe_float(row.get("excess_kl_nats"), default=float("nan"))
                    task_kl = _safe_float(row.get("task_kl_nats"), default=float("nan"))
                    ratio, ratio_is_infinite = compute_excess_to_task_ratio(excess_kl, task_kl)
                    if excess_kl < self.tau_excess_kl:
                        continue
                    if not exceeds_ratio_threshold(ratio, is_infinite=ratio_is_infinite, threshold=self.tau_ratio):
                        continue

                    attribute_name = _safe_text(row.get("attribute_name") or row.get("secret"))
                    description = _safe_text(row.get("attribute_description"))
                    positive_patterns = tuple(_parse_json_list(row.get("attribute_positive_patterns_json")))
                    candidate_texts = build_support_text_candidates(
                        attribute_name,
                        description=description,
                        positive_patterns=positive_patterns,
                    )
                    support = self.support_checker.check(
                        attribute_name,
                        target_row.to_dict(),
                        extra_texts=candidate_texts[1:],
                    )
                    flagged.append(
                        AttributeAudit(
                            attribute_name=attribute_name,
                            excess_kl=float(excess_kl),
                            task_kl=float(task_kl),
                            excess_to_task_kl_ratio=ratio,
                            ratio_is_infinite=ratio_is_infinite,
                            confirmation_status=_safe_text(row.get("confirmation_status") or row.get("status")),
                            supported=bool(support.supported),
                            matched_field=str(support.matched_field),
                            support_score=float(support.support_score),
                            matched_terms=tuple(support.matched_terms),
                            matched_candidate=str(support.matched_candidate),
                            attribute_description=description,
                            positive_patterns=positive_patterns,
                        )
                    )
            audits.append(
                EmbeddingAuditRecord(
                    image_id=image_id,
                    model_name=setting.model_name,
                    k=int(setting.k),
                    llm_name=setting.llm_name,
                    task_label=task_label,
                    available_fields=available_fields,
                    flagged_attributes=tuple(flagged),
                )
            )
        return audits

    def find_existing_run_dir(self, setting: EvaluationSetting) -> Path | None:
        candidates: list[Path] = []
        for path in self.output_dir.glob(f"{setting.slug}*"):
            if not path.is_dir():
                continue
            if not (path / "target_confirmations.csv").exists():
                continue
            if not self._preserved_run_matches_setting(path, setting):
                continue
            candidates.append(path)

        if not candidates:
            return None

        def _mtime(path: Path) -> float:
            confirmations_path = path / "target_confirmations.csv"
            try:
                return confirmations_path.stat().st_mtime
            except Exception:
                return path.stat().st_mtime

        return max(candidates, key=_mtime)

    def _preserved_run_matches_setting(self, run_dir: Path, setting: EvaluationSetting) -> bool:
        name = run_dir.name
        if name != setting.slug and not name.startswith(f"{setting.slug}_"):
            return False

        config_path = run_dir / "eval_run_config.yaml"
        if config_path.exists():
            try:
                run_cfg = load_config(config_path)
            except Exception:
                run_cfg = {}
            meta = run_cfg.get("evaluation_setting") or {}
            if meta:
                meta_mode = _safe_text(meta.get("mode"))
                meta_model = _safe_text(meta.get("model_name"))
                meta_llm = _safe_text(meta.get("llm_name"))
                try:
                    meta_k = int(meta.get("k"))
                except Exception:
                    meta_k = None
                if (
                    meta_mode == setting.mode
                    and meta_model == setting.model_name
                    and meta_llm == setting.llm_name
                    and meta_k == int(setting.k)
                ):
                    return True
                return False

        # Backward-compatible fallback for older preserved runs without explicit metadata.
        # Prefer the longest matching slug so prefix-related names like GPT-5.4 vs GPT-5.4-mini
        # resolve to the correct setting.
        matching_slugs = [setting.slug]
        for llm_name in self.llm_specs:
            other_slug = snakeify(f"{setting.mode}_{setting.model_name}_k_{setting.k}_{llm_name}")
            if other_slug == setting.slug:
                continue
            if name == other_slug or name.startswith(f"{other_slug}_"):
                matching_slugs.append(other_slug)
        return max(matching_slugs, key=len) == setting.slug

    def run_existing_setting(self, setting: EvaluationSetting) -> tuple[dict[str, Any], list[EmbeddingAuditRecord]]:
        run_dir = self.find_existing_run_dir(setting)
        if run_dir is None:
            raise FileNotFoundError(
                f"No preserved run directory was found for setting '{setting.label}' under {self.output_dir}. "
                f"Expected a directory matching '{setting.slug}*' with target_confirmations.csv."
            )

        confirmations_path = run_dir / "target_confirmations.csv"
        confirmation_df = pd.read_csv(confirmations_path) if confirmations_path.exists() else pd.DataFrame()
        if not confirmation_df.empty and "target_image_id" in confirmation_df.columns:
            confirmation_df["target_image_id"] = confirmation_df["target_image_id"].astype(str)

        audits = self._score_audits(setting, confirmation_df)
        summary = summarize_audits(
            audits,
            label_key="Embedding Model" if setting.mode == "rq1" else "Setting",
            label_value=setting.label,
            model_name=setting.model_name,
            k=setting.k,
            llm_name=setting.llm_name,
        )
        return summary, audits

    def settings_for_mode(self, mode: str) -> list[EvaluationSetting]:
        if mode == "rq1":
            return self._rq1_settings()
        if mode == "rq2":
            return self._rq2_settings()
        raise ValueError(f"Unsupported mode: {mode}")

    def _resolve_llm_spec(self, llm_name: str | None) -> tuple[str, LLMSpec]:
        name = llm_name or next(iter(self.llm_specs))
        if name not in self.llm_specs:
            raise KeyError(f"Unknown LLM setting '{name}'. Available: {sorted(self.llm_specs)}")
        return name, self.llm_specs[name]

    def _rq1_models(self) -> list[str]:
        if self.model_override:
            return [self.model_override]
        return list((self.eval_cfg.get("rq1") or {}).get("models") or DEFAULT_RQ1_MODELS)

    def _rq1_settings(self) -> list[EvaluationSetting]:
        llm_name, llm_spec = self._resolve_llm_spec(
            self.llm_override or _safe_text((self.eval_cfg.get("rq1") or {}).get("llm"))
        )
        k = int(self.k_override if self.k_override is not None else (self.eval_cfg.get("rq1") or {}).get("k", 10))
        return [
            EvaluationSetting(
                mode="rq1",
                label=model_name,
                model_name=model_name,
                k=k,
                llm_name=llm_name,
                llm_spec=llm_spec,
            )
            for model_name in self._rq1_models()
        ]

    def _rq2_settings(self) -> list[EvaluationSetting]:
        rq2_cfg = self.eval_cfg.get("rq2") or {}
        model_name = self.model_override or _safe_text(rq2_cfg.get("model")) or _safe_text((self.base_cfg.get("embeddings") or {}).get("model_name")) or "resnet50"

        if self.k_override is not None and self.llm_override:
            llm_name, llm_spec = self._resolve_llm_spec(self.llm_override)
            return [
                EvaluationSetting(
                    mode="rq2",
                    label=f"K={self.k_override} | LLM={llm_name}",
                    model_name=model_name,
                    k=int(self.k_override),
                    llm_name=llm_name,
                    llm_spec=llm_spec,
                )
            ]

        mode = _safe_text(rq2_cfg.get("mode")) or "split"
        k_values = [int(v) for v in (rq2_cfg.get("k_values") or [5, 10, 20])]
        llm_names = list(rq2_cfg.get("llms") or list(self.llm_specs.keys()))

        if self.k_override is not None:
            k_values = [int(self.k_override)]
        if self.llm_override:
            llm_names = [self.llm_override]

        settings: list[EvaluationSetting] = []
        if mode == "full_grid":
            for k in k_values:
                for llm_name in llm_names:
                    resolved_name, spec = self._resolve_llm_spec(llm_name)
                    settings.append(
                        EvaluationSetting(
                            mode="rq2",
                            label=f"K={k} | LLM={resolved_name}",
                            model_name=model_name,
                            k=int(k),
                            llm_name=resolved_name,
                            llm_spec=spec,
                        )
                    )
            return settings

        ref_llm_name, ref_llm_spec = self._resolve_llm_spec(
            self.llm_override or _safe_text(rq2_cfg.get("reference_llm")) or next(iter(self.llm_specs))
        )
        ref_k = int(self.k_override if self.k_override is not None else rq2_cfg.get("reference_k", 10))

        if self.llm_override and self.k_override is None:
            for k in k_values:
                settings.append(
                    EvaluationSetting(
                        mode="rq2",
                        label=f"K={k}",
                        model_name=model_name,
                        k=int(k),
                        llm_name=ref_llm_name,
                        llm_spec=ref_llm_spec,
                    )
                )
            return settings

        if self.k_override is not None and not self.llm_override:
            for llm_name in llm_names:
                resolved_name, spec = self._resolve_llm_spec(llm_name)
                settings.append(
                    EvaluationSetting(
                        mode="rq2",
                        label=f"LLM={resolved_name}",
                        model_name=model_name,
                        k=ref_k,
                        llm_name=resolved_name,
                        llm_spec=spec,
                    )
                )
            return settings

        for k in k_values:
            settings.append(
                EvaluationSetting(
                    mode="rq2",
                    label=f"K={k}",
                    model_name=model_name,
                    k=int(k),
                    llm_name=ref_llm_name,
                    llm_spec=ref_llm_spec,
                )
            )
        for llm_name in llm_names:
            resolved_name, spec = self._resolve_llm_spec(llm_name)
            settings.append(
                EvaluationSetting(
                    mode="rq2",
                    label=f"LLM={resolved_name}",
                    model_name=model_name,
                    k=ref_k,
                    llm_name=resolved_name,
                    llm_spec=spec,
                )
            )
        return settings

    def run_mode(self, mode: str, *, reuse_existing_runs: bool = False) -> tuple[list[dict[str, Any]], list[EmbeddingAuditRecord]]:
        settings = self.settings_for_mode(mode)
        summaries: list[dict[str, Any]] = []
        audits: list[EmbeddingAuditRecord] = []
        for setting in settings:
            if reuse_existing_runs:
                summary, audit_rows = self.run_existing_setting(setting)
            else:
                summary, audit_rows = self.run_setting(setting)
            summaries.append(summary)
            audits.extend(audit_rows)
        return summaries, audits
