from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from audit.config import load_config, resolve
from audit.utils import ensure_parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)

    splits_csv = resolve(cfg["paths"]["splits_csv"], project_root)
    emb_npz = resolve(cfg["paths"]["embeddings_npz"], project_root)
    out_npz = resolve(cfg["paths"]["index_npz"], project_root)
    ensure_parent(out_npz)

    splits = pd.read_csv(splits_csv)
    emb = np.load(emb_npz, allow_pickle=True)
    ids = emb["image_ids"].astype(str)
    embs = emb["embeddings"].astype(np.float32)
    id_to_pos = {k: i for i, k in enumerate(ids)}

    corpus_ids = splits.loc[splits["split"] == "corpus", "image_id"].astype(str).tolist()
    corpus_pos = [id_to_pos[i] for i in corpus_ids if i in id_to_pos]
    corpus_embs = embs[corpus_pos]

    np.savez_compressed(out_npz, corpus_ids=np.array(corpus_ids), corpus_embeddings=corpus_embs)
    print(f"Saved corpus index to {out_npz}: {corpus_embs.shape}")


if __name__ == "__main__":
    main()
