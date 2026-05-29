from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.runner import EvaluationRunner


REVIEW_METADATA_COLUMNS = [
    "image_path",
    "image_relpath",
    "output_file",
    "task_label",
    "semantic_text",
    "scene_family",
    "scene_family_label",
    "primary_label",
    "primary_object",
    "primary_attributes",
    "secondary_label",
    "secondary_object",
    "secondary_attributes",
    "ternary_label",
    "ternary_object",
    "ternary_attributes",
    "background_label",
    "background_scene",
    "background_attributes",
]

RQ1_DEFAULT_LLM = "GPT-4o-mini"
RQ2_ABLATION_LLMS = ("GPT-5.4", "GPT-4.1", "GPT-5.4-mini", "o3")
DEFAULT_EVAL_DEVICE = "cuda:1"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reorder_columns(df: pd.DataFrame, leading: list[str]) -> pd.DataFrame:
    cols = [col for col in leading if col in df.columns]
    cols.extend(col for col in df.columns if col not in cols)
    return df[cols]


def _safe_cell(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return value


def _join_values(values: tuple[str, ...]) -> str:
    return " | ".join(str(item) for item in values if str(item).strip())


def _mode_llm_override(mode: str, cli_llm: str | None) -> str | None:
    if cli_llm:
        return cli_llm
    if mode == "rq1":
        return RQ1_DEFAULT_LLM
    return None


def _apply_mode_defaults(runner: EvaluationRunner, mode: str) -> None:
    if mode == "rq1":
        if RQ1_DEFAULT_LLM not in runner.llm_specs:
            raise KeyError(f"Required RQ1 LLM '{RQ1_DEFAULT_LLM}' is not configured.")
        if not runner.llm_override:
            rq1_cfg = dict(runner.eval_cfg.get("rq1") or {})
            rq1_cfg["llm"] = RQ1_DEFAULT_LLM
            runner.eval_cfg["rq1"] = rq1_cfg
        return

    if mode == "rq2":
        rq2_cfg = dict(runner.eval_cfg.get("rq2") or {})
        configured_llms = list(rq2_cfg.get("llms") or RQ2_ABLATION_LLMS)
        missing = [name for name in configured_llms if name not in runner.llm_specs]
        if missing:
            raise KeyError(f"Required RQ2 LLMs are not configured: {missing}")
        if not runner.llm_override:
            rq1_cfg = dict(runner.eval_cfg.get("rq1") or {})
            rq2_cfg["reference_k"] = int(rq1_cfg.get("k", rq2_cfg.get("reference_k", 10)))
            rq2_cfg["reference_llm"] = str(rq1_cfg.get("llm") or RQ1_DEFAULT_LLM)
            rq2_cfg["llms"] = configured_llms
            runner.eval_cfg["rq2"] = rq2_cfg


def _target_lookup(target_manifest: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in target_manifest.iterrows():
        image_id = str(row["image_id"])
        payload = {col: _safe_cell(row.get(col)) for col in REVIEW_METADATA_COLUMNS if col in row.index}
        payload["image_id"] = image_id
        lookup[image_id] = payload
    return lookup


def _review_rows(audits: list[Any], target_manifest: pd.DataFrame) -> list[dict[str, Any]]:
    target_by_id = _target_lookup(target_manifest)
    rows: list[dict[str, Any]] = []
    for audit in audits:
        if not audit.flagged_attributes:
            continue
        target_meta = target_by_id.get(str(audit.image_id), {})
        task_label = target_meta.get("task_label") or audit.task_label
        available_fields = tuple(str(item) for item in getattr(audit, "available_fields", ()) if str(item).strip())
        recovered_fields = tuple(str(item) for item in getattr(audit, "recovered_fields", ()) if str(item).strip())
        leakage_available_fields = tuple(str(item) for item in getattr(audit, "leakage_available_fields", ()) if str(item).strip())
        leakage_recovered_fields = tuple(str(item) for item in getattr(audit, "leakage_recovered_fields", ()) if str(item).strip())
        task_field = str(getattr(audit, "task_field", "") or "")
        for idx, attr in enumerate(audit.flagged_attributes, start=1):
            matched_slot = str(attr.matched_field)
            task_aligned_prediction = bool(attr.supported and task_field and matched_slot == task_field)
            excess_grounded_prediction = bool(attr.supported and matched_slot in leakage_available_fields and not task_aligned_prediction)
            rows.append(
                {
                    "image_id": str(audit.image_id),
                    "image_path": target_meta.get("image_path", ""),
                    "image_relpath": target_meta.get("image_relpath", ""),
                    "output_file": target_meta.get("output_file", ""),
                    "task_label": task_label,
                    "semantic_text": target_meta.get("semantic_text", ""),
                    "scene_family": target_meta.get("scene_family", ""),
                    "scene_family_label": target_meta.get("scene_family_label", ""),
                    "primary_label": target_meta.get("primary_label", ""),
                    "secondary_label": target_meta.get("secondary_label", ""),
                    "ternary_label": target_meta.get("ternary_label", ""),
                    "background_label": target_meta.get("background_label", ""),
                    "model_name": str(audit.model_name),
                    "K": int(audit.k),
                    "llm_name": str(audit.llm_name),
                    "available_fields": _join_values(available_fields),
                    "recovered_fields": _join_values(recovered_fields),
                    "leakage_available_fields": _join_values(leakage_available_fields),
                    "leakage_recovered_fields": _join_values(leakage_recovered_fields),
                    "task_field": task_field,
                    "annotated_available_slot_count": int(len(available_fields)),
                    "annotated_recovered_slot_count": int(getattr(audit, "num_recovered_fields", len(recovered_fields))),
                    "available_slot_count": int(len(leakage_available_fields)),
                    "recovered_slot_count": int(getattr(audit, "num_leakage_recovered_fields", len(leakage_recovered_fields))),
                    "annotated_slot_recall_image": float(getattr(audit, "annotated_slot_recall", 0.0)),
                    "slot_recall_image": float(getattr(audit, "slot_recall", 0.0)),
                    "any_slot_recovered": bool(getattr(audit, "any_slot_recovered", False)),
                    "full_slot_recovered": bool(getattr(audit, "full_slot_recovered", False)),
                    "num_flagged_for_image": int(audit.num_flagged),
                    "num_valid_flagged_for_image": int(getattr(audit, "num_excess_grounded_flagged", 0)),
                    "num_task_aligned_for_image": int(getattr(audit, "num_task_aligned_flagged", 0)),
                    "num_invalid_flagged_for_image": int(audit.num_invalid_flagged),
                    "flagged_attribute_index": int(idx),
                    "attribute_name": str(attr.attribute_name),
                    "attribute_description": str(attr.attribute_description),
                    "positive_patterns": _join_values(attr.positive_patterns),
                    "excess_kl": float(attr.excess_kl),
                    "task_kl": float(attr.task_kl),
                    "excess_to_task_kl_ratio": "inf" if attr.ratio_is_infinite else attr.excess_to_task_kl_ratio,
                    "ratio_is_infinite": bool(attr.ratio_is_infinite),
                    "confirmation_status": str(attr.confirmation_status),
                    "semantic_supported": bool(attr.supported),
                    "annotation_grounded": excess_grounded_prediction,
                    "task_aligned_prediction": task_aligned_prediction,
                    "grounding_label": (
                        "Task-aligned recovery"
                        if task_aligned_prediction
                        else ("Excess grounded recovery" if excess_grounded_prediction else "Unmapped candidate")
                    ),
                    "support_score": float(attr.support_score),
                    "matched_field": str(attr.matched_field),
                    "matched_slot": matched_slot,
                    "matched_terms": _join_values(attr.matched_terms),
                    "matched_candidate": str(getattr(attr, "matched_candidate", "")),
                    "human_review_label": "",
                    "human_reviewer": "",
                    "human_review_notes": "",
                }
            )
    return rows


def _flagged_image_rows(audits: list[Any], target_manifest: pd.DataFrame) -> list[dict[str, Any]]:
    target_by_id = _target_lookup(target_manifest)
    rows: list[dict[str, Any]] = []
    for audit in audits:
        if not audit.flagged_attributes:
            continue
        target_meta = target_by_id.get(str(audit.image_id), {})
        rows.append(
            {
                "image_id": str(audit.image_id),
                "image_path": target_meta.get("image_path", ""),
                "image_relpath": target_meta.get("image_relpath", ""),
                "output_file": target_meta.get("output_file", ""),
                "task_label": target_meta.get("task_label") or audit.task_label,
                "semantic_text": target_meta.get("semantic_text", ""),
                "scene_family": target_meta.get("scene_family", ""),
                "scene_family_label": target_meta.get("scene_family_label", ""),
                "primary_label": target_meta.get("primary_label", ""),
                "secondary_label": target_meta.get("secondary_label", ""),
                "ternary_label": target_meta.get("ternary_label", ""),
                "background_label": target_meta.get("background_label", ""),
                "model_name": str(audit.model_name),
                "K": int(audit.k),
                "llm_name": str(audit.llm_name),
                "available_fields": _join_values(tuple(str(item) for item in getattr(audit, "available_fields", ()) if str(item).strip())),
                "recovered_fields": _join_values(tuple(str(item) for item in getattr(audit, "recovered_fields", ()) if str(item).strip())),
                "leakage_available_fields": _join_values(tuple(str(item) for item in getattr(audit, "leakage_available_fields", ()) if str(item).strip())),
                "leakage_recovered_fields": _join_values(tuple(str(item) for item in getattr(audit, "leakage_recovered_fields", ()) if str(item).strip())),
                "task_field": str(getattr(audit, "task_field", "") or ""),
                "annotated_available_slot_count": int(len(getattr(audit, "available_fields", ()))),
                "annotated_recovered_slot_count": int(getattr(audit, "num_recovered_fields", 0)),
                "available_slot_count": int(len(getattr(audit, "leakage_available_fields", ()))),
                "recovered_slot_count": int(getattr(audit, "num_leakage_recovered_fields", 0)),
                "annotated_slot_recall_image": float(getattr(audit, "annotated_slot_recall", 0.0)),
                "slot_recall_image": float(getattr(audit, "slot_recall", 0.0)),
                "any_slot_recovered": bool(getattr(audit, "any_slot_recovered", False)),
                "full_slot_recovered": bool(getattr(audit, "full_slot_recovered", False)),
                "num_flagged": int(audit.num_flagged),
                "num_valid_flagged": int(getattr(audit, "num_excess_grounded_flagged", 0)),
                "num_task_aligned": int(getattr(audit, "num_task_aligned_flagged", 0)),
                "num_invalid_flagged": int(audit.num_invalid_flagged),
                "flagged_attributes": [item.to_dict() for item in audit.flagged_attributes],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rq1", "rq2", "all"], required=True)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--embeddings_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=DEFAULT_EVAL_DEVICE)
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--llm", default=None)
    ap.add_argument("--tau_excess_kl", type=float, default=None)
    ap.add_argument("--tau_ratio", type=float, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--reuse_existing_runs",
        action="store_true",
        help="Rebuild aggregate evaluation outputs from preserved output/<setting>_* run directories instead of rerunning stage 1/2.",
    )
    args = ap.parse_args()

    modes = ["rq1", "rq2"] if args.mode == "all" else [args.mode]
    generated_at = datetime.now(timezone.utc).isoformat()

    for mode in modes:
        runner = EvaluationRunner(
            args.config,
            dataset_path=args.dataset,
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            device=args.device,
            model_override=args.model,
            k_override=args.k,
            llm_override=_mode_llm_override(mode, args.llm),
            tau_excess_kl=args.tau_excess_kl,
            tau_ratio=args.tau_ratio,
            dry_run=args.dry_run,
        )
        _apply_mode_defaults(runner, mode)
        summaries, audits = runner.run_mode(mode, reuse_existing_runs=bool(args.reuse_existing_runs))
        output_dir = runner.output_dir
        csv_path = output_dir / f"eval_{mode}.csv"
        json_path = output_dir / f"eval_{mode}.json"
        audit_path = output_dir / f"audits_{mode}.jsonl"
        flagged_csv_path = output_dir / f"flagged_attributes_{mode}.csv"
        flagged_images_path = output_dir / f"flagged_images_{mode}.jsonl"

        summary_df = pd.DataFrame(summaries)
        if mode == "rq1":
            summary_df = _reorder_columns(
                summary_df,
                [
                    "Embedding Model",
                    "SlotRecall",
                    "AnyRecoveryRate",
                    "FullRecoveryRate",
                    "AvgRecoveredSlotsPerImage",
                    "GroundedPredictionRate",
                    "OpenWorldCandidateYield",
                    "SVR",
                    "HCE",
                    "ESY",
                ],
            )
        else:
            summary_df = _reorder_columns(
                summary_df,
                [
                    "Setting",
                    "SlotRecall",
                    "AnyRecoveryRate",
                    "FullRecoveryRate",
                    "AvgRecoveredSlotsPerImage",
                    "GroundedPredictionRate",
                    "OpenWorldCandidateYield",
                    "SVR",
                    "HCE",
                    "ESY",
                ],
            )
        summary_df.to_csv(csv_path, index=False)

        review_rows = _review_rows(audits, runner.target_manifest)
        flagged_df = pd.DataFrame(review_rows)
        if not flagged_df.empty:
            flagged_df = _reorder_columns(
                flagged_df,
                [
                    "image_id",
                    "image_path",
                    "task_label",
                    "model_name",
                    "K",
                    "llm_name",
                    "available_slot_count",
                    "recovered_slot_count",
                    "slot_recall_image",
                    "num_flagged_for_image",
                    "flagged_attribute_index",
                    "attribute_name",
                    "annotation_grounded",
                    "grounding_label",
                    "semantic_supported",
                    "support_score",
                    "matched_slot",
                    "matched_field",
                    "confirmation_status",
                    "excess_kl",
                    "task_kl",
                    "excess_to_task_kl_ratio",
                    "human_review_label",
                    "human_reviewer",
                    "human_review_notes",
                ],
            )
        flagged_df.to_csv(flagged_csv_path, index=False)
        _write_jsonl(flagged_images_path, _flagged_image_rows(audits, runner.target_manifest))

        _write_json(
            json_path,
            {
                "mode": mode,
                "generated_at_utc": generated_at,
                "config_path": str(Path(args.config).expanduser().resolve()),
                "base_config_path": str(runner.base_config_path),
                "output_dir": str(output_dir),
                "dry_run": bool(runner.dry_run),
                "reuse_existing_runs": bool(args.reuse_existing_runs),
                "tau_excess_kl": float(runner.tau_excess_kl),
                "tau_ratio": float(runner.tau_ratio),
                "summaries": summaries,
            },
        )
        _write_jsonl(audit_path, [audit.to_dict() for audit in audits])

        print(f"[OK] Wrote {csv_path}")
        print(f"[OK] Wrote {json_path}")
        print(f"[OK] Wrote {audit_path}")
        print(f"[OK] Wrote {flagged_csv_path}")
        print(f"[OK] Wrote {flagged_images_path}")


if __name__ == "__main__":
    main()
