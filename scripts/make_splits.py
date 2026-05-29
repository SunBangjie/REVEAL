from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from audit.config import load_config, normalize_manifest_df, resolve
from audit.utils import ensure_parent, set_seed


LEGACY_SPLITS = ["corpus", "discovery", "rubric", "verification"]
NEW_SPLITS = ["corpus", "audited"]


def _resolve_split_names_and_probs(cfg_splits: dict) -> tuple[list[str], np.ndarray, str]:
    """Resolve split names/probabilities from config.

    Preferred new layout:
      - corpus
      - audited

    Backward-compatible legacy layout:
      - corpus
      - discovery
      - rubric
      - verification
    """
    keys = list(cfg_splits.keys())
    keyset = set(keys)

    has_new = set(NEW_SPLITS).issubset(keyset)
    has_legacy = set(LEGACY_SPLITS).issubset(keyset)

    if has_new and has_legacy:
        raise ValueError(
            "Config mixes new and legacy split schemes. Use either {'corpus', 'audited'} "
            "or {'corpus', 'discovery', 'rubric', 'verification'}, not both."
        )

    if has_new:
        names = NEW_SPLITS
        scheme = "new"
    elif has_legacy:
        names = LEGACY_SPLITS
        scheme = "legacy"
    elif "corpus" in keyset:
        # Convenience mode: if only corpus is provided, assign the remainder to audited.
        corpus_frac = float(cfg_splits["corpus"])
        audited_frac = max(0.0, 1.0 - corpus_frac)
        names = NEW_SPLITS
        cfg_splits = {"corpus": corpus_frac, "audited": audited_frac}
        scheme = "new(auto)"
    else:
        raise ValueError(
            "Invalid cfg['splits']. Expected either keys {'corpus', 'audited'} "
            "or {'corpus', 'discovery', 'rubric', 'verification'}."
        )

    probs = np.array([float(cfg_splits[n]) for n in names], dtype=float)
    if np.any(probs < 0):
        raise ValueError(f"Split fractions must be non-negative. Got: {dict(zip(names, probs.tolist()))}")
    if not np.any(probs > 0):
        raise ValueError("At least one split fraction must be > 0.")

    probs = probs / probs.sum()
    return names, probs, scheme


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 123))
    set_seed(seed)

    manifest_csv = resolve(cfg["paths"]["manifest_csv"], project_root)
    out_csv = resolve(cfg["paths"]["splits_csv"], project_root)
    ensure_parent(out_csv)
    df = pd.read_csv(manifest_csv)
    df = normalize_manifest_df(df, cfg, project_root)

    names, probs, scheme = _resolve_split_names_and_probs(cfg["splits"])

    rng = np.random.default_rng(seed)
    split = rng.choice(names, size=len(df), p=probs)

    out = df[["image_id"]].copy()
    out["split"] = split
    out.to_csv(out_csv, index=False)

    print(f"[INFO] Split scheme: {scheme}")
    print(out["split"].value_counts())
    print(f"Saved splits to {out_csv}")


if __name__ == "__main__":
    main()
