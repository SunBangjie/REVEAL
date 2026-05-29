# REVEAL
This repository provides the codebase for the paper ["What Do Neighbors Know? Open-World Semantic Inference Attack on Intermediate Representations"](https://sunbangjie.github.io/files/REVEAL.pdf) published at NetAISys'26 co-located with MobiSys'26

## Overview

This repo audits semantic information that can be recovered from released intermediate representations for the same audited target set.

Current pipeline:

1. Build a flat manifest from DCI-style annotation JSON files.
2. Compute frozen pretrained image embeddings, and optionally predicted labels or logits.
3. Split the data into:
   - `corpus`: auxiliary retrieval corpus
   - `audited`: fixed target set to audit
4. Build a cosine retrieval index over the `corpus` embeddings.
5. Run Stage 1 discovery on the `audited` split by exporting offline prompts, then ingesting saved LLM responses into `stage1_discovery.jsonl`.
6. Build target-specific weak labels directly from Stage 1 outputs.
7. Confirm, reject, or abstain on excess semantic leakage for each audited target-attribute pair.
8. Generate per-target Markdown reports.

## Installation

The package targets Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you already installed this project before the model-registry change, reinstall dependencies so `timm` is available:

```bash
pip install -e .
```

## Supported backbones

Both `scripts/compute_embeddings.py` and `streamlit_privacy_audit_app.py` read `embeddings.model_name` from `configs/mvp.yaml`.

Supported canonical `model_name` values:

- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `mobilenet_v3_small_quantized`
- `mobilenet_v3_large_quantized`
- `resnet50`
- `efficientnet_b0`
- `vit_base_patch16_224`
- `convnext_tiny`

Backbone sources:

- `mobilenet_v3_*` and `resnet50` come from `torchvision` pretrained ImageNet-1K weights.
- `efficientnet_b0`, `vit_base_patch16_224`, and `convnext_tiny` come from `timm` with pretrained ImageNet-1K weights.

Important operational notes:

- Quantized MobileNet variants run on CPU only in this repo.
- The Streamlit class-activation-map tab is available for non-quantized backbones. It is disabled for quantized MobileNet variants.
- If you change `embeddings.model_name`, you should recompute at least:
  - `python scripts/compute_embeddings.py --config configs/mvp.yaml`
  - `python scripts/build_index.py --config configs/mvp.yaml`
- In practice, after switching backbone, rerunning the full downstream pipeline is recommended so retrieval, discovery, and confirmation evidence all stay consistent with the new embedding space.

## Expected dataset layout

Point the config at the DCI-style dataset roots:

- `dataset.dataset_root`: path to the `densely_captioned_images` folder
- `dataset.complete_root`: path to `densely_captioned_images/complete`
- `dataset.annotations_root`: path to `densely_captioned_images/annotations`
- `dataset.photos_root`: path to `densely_captioned_images/photos`

The manifest builder prefers `complete/` over `annotations/` when `json_root_preference: auto`, and it resolves image files through the JSON `image` field rather than by matching annotation filenames to image filenames.

The manifest code expects DCI-like fields such as:

- `image`
- `short_caption`
- `extra_caption`
- `mask_data`
- `summaries`
- `negatives`

## Active config behavior

- `splits` uses the current two-way scheme: `corpus` and `audited`. If you provide only `corpus`, the remainder is assigned to `audited`.
- `discovery.offline_llm` overrides shared prompt settings from `discovery.llm` during offline export and ingest.
- `operationalization` is the active label-building config. `build_labels.py` derives per-target schemas directly from `paths.stage1_jsonl` when `operationalization.source: stage1`.
- There is no standalone `scripts/stage2_generate_operationalization_schema.py` in the current repo.
- `schema_generation` exists in `configs/mvp.yaml`, but the current scripts build labels from `operationalization`, not from a separate schema-generation executable.
- For `confirmation.baseline_mode: Y` or `O`, provide task outputs through `embeddings.npz` or `paths.task_outputs_csv`. An external CSV should contain `image_id` plus `pred_label`/`predicted_label` and optional `logit_*` columns or a `logits` column.
- `streamlit_privacy_audit_app.py` uses the same configured backbone as the CLI embedding pipeline. The uploaded-image inference path should match the backbone used to generate `embeddings.npz` and `corpus_index.npz`.

## Stage 1 offline discovery

`scripts/stage1_discover.py` is an offline export/ingest workflow.

Export prompts:

```bash
python scripts/stage1_discover.py \
  --config configs/mvp.yaml \
  --action export_prompts \
  --work-dir outputs/stage1_offline
```

This creates a work directory like:

```text
outputs/stage1_offline/
  prompts/
    system_prompt.txt
    prompt_manifest.jsonl
    000001__<imageid>.prompt.txt
    000001__<imageid>.prompt.json
  cache/
    stage1_records_cache.jsonl
    export_summary.json
  responses/
  README.txt
```

For each exported prompt, save the model's JSON reply into the matching file under `responses/`, typically:

```text
outputs/stage1_offline/responses/000001__<imageid>.response.txt
```

Accepted response suffixes are `.response.txt`, `.response.json`, `.response.md`, or bare `.txt`, `.json`, `.md`.

Ingest responses:

```bash
python scripts/stage1_discover.py \
  --config configs/mvp.yaml \
  --action ingest_responses \
  --work-dir outputs/stage1_offline
```

By default, missing or invalid response files fall back to retrieval-only records with auditability metadata preserved. Use `--disable-fallback` if you want ingest to fail instead.

## End-to-end quick start

```bash
pip install -e .

# Prepare the model embeddings
python scripts/build_manifest.py --config configs/mvp.yaml
python scripts/compute_embeddings.py --config configs/mvp.yaml
python scripts/make_splits.py --config configs/mvp.yaml
python scripts/build_index.py --config configs/mvp.yaml

# Running scripts
python scripts/stage1_discover.py --config configs/mvp.yaml --action export_prompts --work-dir outputs/stage1_offline
# Save JSON replies into outputs/stage1_offline/responses/
python scripts/stage1_discover.py --config configs/mvp.yaml --action ingest_responses --work-dir outputs/stage1_offline

python scripts/build_labels.py --config configs/mvp.yaml
python scripts/stage2_confirm_excess_leakage.py --config configs/mvp.yaml
python scripts/report_targets.py --config configs/mvp.yaml

# Alternatively, running the streamlit app
streamlit run streamlit_privacy_audit_app.py
```

To rebuild the aggregate evaluation artifacts from preserved run directories without making fresh LLM calls, use:

```bash
python scripts/run_evaluation.py --mode rq1 --reuse_existing_runs
```

This expects preserved run folders such as `output/rq1_<setting>_*` containing `target_confirmations.csv`.

## Excess leakage confirmation

`scripts/stage2_confirm_excess_leakage.py` treats Stage 3 as a per-embedding confirmation/filtering step.
The null view is that the task label already explains the attribute. The alternative view is that the released representation adds evidence beyond the task label.

For each `(target, attribute)` pair it computes:

- Baseline posterior: `q(S|Y = y_i)`
- Conditional posterior: `q(S|Z = z_i, Y = y_i)`
- Supporting evidence: posterior shift, excess lift, and excess KL
- Decision outcome: `confirmed`, `rejected`, or `inconclusive`

An attribute is confirmed only when:

1. `q(S|Z,Y)` is meaningfully larger than `q(S|Y)`, and
2. support and operationalization reliability checks pass.

If support is weak, the pipeline abstains with `inconclusive`. If the increase beyond the task-label baseline is small, the attribute is rejected as task-explained or unsupported.

Supported baseline modes in `confirmation.baseline_mode`:

- `Y`: baseline uses the predicted class label. This is the primary excess-confirmation path.
- `O`: baseline uses output logits. This is retained as a legacy compatibility/debug option.
- `NONE`: uses the marginal prior as a fallback baseline. This is retained for backward compatibility and debugging, but it does not provide the intended task-relative null.

If `baseline_mode: Y`, `compute_embeddings.py` will auto-enable predicted-label export when needed and `task_outputs_csv` is unavailable. If `baseline_mode: O`, it can likewise auto-enable logits export.

Absence of confirmation does not prove absence of leakage. It only means the current evidence was not strong enough to clear the conservative decision thresholds.

## Reports

- `scripts/report_targets.py` writes one Markdown file per audited target plus `index.md` in `paths.target_report_dir`.
- To render local images in Markdown, set `report.markdown_image_root` to the absolute photos root.

## Streamlit app

The repo also includes an interactive demo:

```bash
streamlit run streamlit_privacy_audit_app.py
```

Current app capabilities include:

- retrieval preview using DCI-style caption text
- offline Stage 1 prompt generation and JSON parsing
- script-backed excess leakage confirmation and target-report rendering
- class activation map visualization for supported non-quantized backbones

Per-target reports can include:

- the target image
- retrieved neighbor images
- auditability statistics
- Stage 1 summaries, shared attributes, and excess secrets
- per-attribute confirmation outcomes and supporting diagnostics

## Outputs

Typical generated files:

- `data/manifests/manifest.csv`
- `data/caches/embeddings.npz`
- `data/splits/splits.csv`
- `data/indices/corpus_index.npz`
- `data/reports/stage1_discovery.jsonl`
- `data/labels/secret_labels.csv`
- `data/reports/confirmation_summary.csv`
- `data/reports/target_confirmations.csv`
- `data/reports/semantic_validity_reviews.csv` or `output/<run_name>/semantic_validity_reviews.csv`
- `data/reports/target_reports/index.md`
- `data/reports/target_reports/<target_image_id>.md`

Note: the current pipeline does not automatically write `data/labels/operationalization.generated.yaml`. If you want to use an external reusable schema file, provide it yourself and point `operationalization.schema_path` to it.