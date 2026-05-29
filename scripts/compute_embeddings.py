from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from audit.confirmation import merged_confirmation_config
from audit.config import load_config, normalize_manifest_df, resolve, resolve_image_path_from_record
from audit.modeling import build_embedder, build_preprocess, get_model_info, resolve_runtime_device
from audit.utils import ensure_parent, set_seed, normalize_rows


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, cfg: dict, project_root: Path):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.cfg = cfg
        self.project_root = project_root
        self.tf = build_preprocess(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = resolve_image_path_from_record(row, self.cfg, self.project_root)
        try:
            if path is None:
                raise FileNotFoundError("No image path could be resolved from the manifest row.")
            img = Image.open(path).convert("RGB")
            ok = 1
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
            ok = 0
        return self.tf(img), row["image_id"], ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 123)))

    manifest_csv = resolve(cfg["paths"]["manifest_csv"], project_root)
    out_npz = resolve(cfg["paths"]["embeddings_npz"], project_root)
    ensure_parent(out_npz)

    emb_cfg = cfg["embeddings"]
    paths_cfg = cfg.get("paths") or {}
    confirmation_cfg = merged_confirmation_config(cfg)
    discovery_cfg = cfg.get("discovery") or {}

    save_logits = bool(emb_cfg.get("save_logits", False))
    save_pred_labels = bool(emb_cfg.get("save_pred_labels", False))

    task_outputs_csv = paths_cfg.get("task_outputs_csv")
    task_outputs_exists = False
    if task_outputs_csv:
        task_outputs_exists = resolve(task_outputs_csv, project_root).exists()

    baseline_mode = str(confirmation_cfg.get("baseline_mode", "NONE") or "NONE").strip().upper()
    wants_stage1_labels = str(discovery_cfg.get("mode", "") or "").strip().lower() == "llm"

    auto_pred_labels = (not save_pred_labels) and (wants_stage1_labels or baseline_mode == "Y") and (not task_outputs_exists)
    auto_logits = (not save_logits) and (baseline_mode == "O") and (not task_outputs_exists)
    if auto_pred_labels:
        save_pred_labels = True
        print("[INFO] Auto-enabling embeddings.save_pred_labels because task_outputs_csv is unavailable and the pipeline needs image-level class labels.")
    if auto_logits:
        save_logits = True
        print("[INFO] Auto-enabling embeddings.save_logits because baseline_mode=O and task_outputs_csv is unavailable.")

    df = pd.read_csv(manifest_csv)
    df = normalize_manifest_df(df, cfg, project_root)
    ds = ManifestDataset(df, image_size=int(emb_cfg["image_size"]), cfg=cfg, project_root=project_root)
    dl = DataLoader(
        ds,
        batch_size=int(emb_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(emb_cfg["num_workers"]),
    )

    model_info = get_model_info(emb_cfg["model_name"])
    device = resolve_runtime_device(emb_cfg.get("device", "auto"), model_info)
    if model_info.requires_cpu and str(emb_cfg.get("device", "auto")).strip().lower() != "cpu":
        print(f"[INFO] model_name={model_info.canonical_name} requires CPU inference; overriding embeddings.device to cpu.")
    model = build_embedder(emb_cfg["model_name"]).to(device).eval()

    all_ids, all_ok, all_embs = [], [], []
    all_logits = [] if save_logits else None
    all_pred_labels = [] if save_pred_labels else None

    with torch.no_grad():
        for x, ids, ok in tqdm(dl, desc="Embedding"):
            x = x.to(device)
            emb, logits = model(x)
            all_embs.append(emb.cpu().numpy().astype(np.float32))
            all_ids.extend(list(ids))
            all_ok.extend(ok.numpy().tolist())

            if save_logits:
                all_logits.append(logits.cpu().numpy().astype(np.float32))
            if save_pred_labels:
                preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
                all_pred_labels.append(preds)

    embs = normalize_rows(np.concatenate(all_embs, axis=0))

    payload = {
        "image_ids": np.array(all_ids),
        "ok": np.array(all_ok),
        "embeddings": embs,
    }
    if save_logits and all_logits is not None:
        payload["logits"] = np.concatenate(all_logits, axis=0)
    if save_pred_labels and all_pred_labels is not None:
        payload["pred_labels"] = np.concatenate(all_pred_labels, axis=0)

    np.savez_compressed(out_npz, **payload)

    extras = []
    if save_logits:
        extras.append(f"logits={payload['logits'].shape}")
    if save_pred_labels:
        extras.append(f"pred_labels={payload['pred_labels'].shape}")
    extras_str = (", " + ", ".join(extras)) if extras else ""
    print(f"Saved embeddings to {out_npz}: embeddings={embs.shape}{extras_str}")


if __name__ == "__main__":
    main()
