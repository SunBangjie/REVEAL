from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm

from audit.config import load_config, resolve
from audit.dci import parse_dci_record
from audit.utils import ensure_parent, iter_json_files, load_json, set_seed


def choose_dci_json_root(dataset_cfg: dict, project_root: Path) -> Path:
    """Choose which DCI JSON directory to enumerate.

    DCI has both `annotations/` (used by explorer) and `complete/` (used by
    DenseCaptionedImage). For the MVP we prefer `complete/` when available,
    because it contains the richer records, but allow explicit override.
    """
    preference = str(dataset_cfg.get("json_root_preference", "auto")).lower()

    candidates: list[tuple[str, Path]] = []
    dataset_root = dataset_cfg.get("dataset_root")
    if dataset_root:
        root = Path(dataset_root)
        candidates.extend([
            ("complete", (root / "complete")),
            ("annotations", (root / "annotations")),
        ])

    explicit_complete = dataset_cfg.get("complete_root")
    if explicit_complete:
        candidates.append(("complete", Path(explicit_complete)))

    explicit_annotations = dataset_cfg.get("annotations_root")
    if explicit_annotations:
        candidates.append(("annotations", Path(explicit_annotations)))

    if preference in {"complete", "annotations"}:
        ordered = [p for kind, p in candidates if kind == preference] + [p for kind, p in candidates if kind != preference]
    else:
        ordered = [p for kind, p in candidates if kind == "complete"] + [p for kind, p in candidates if kind == "annotations"]

    seen = set()
    unique_ordered: list[Path] = []
    for p in ordered:
        rp = resolve(p, project_root) if not p.is_absolute() else p.resolve()
        key = str(rp)
        if key not in seen:
            seen.add(key)
            unique_ordered.append(rp)

    for p in unique_ordered:
        if p.exists() and p.is_dir():
            return p

    searched = "\n  - " + "\n  - ".join(str(p) for p in unique_ordered) if unique_ordered else " <none>"
    raise FileNotFoundError(f"Could not find a DCI JSON directory. Searched:{searched}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 123)))

    ann_root = choose_dci_json_root(cfg["dataset"], project_root)
    photos_root = resolve(cfg["dataset"]["photos_root"], project_root)
    out_csv = resolve(cfg["paths"]["manifest_csv"], project_root)
    ensure_parent(out_csv)

    print(f"[INFO] Using DCI JSON root: {ann_root}")
    print(f"[INFO] Using DCI photos root: {photos_root}")

    rows = []
    for p in tqdm(sorted(iter_json_files(ann_root)), desc="Scanning annotations"):
        try:
            record = load_json(p)
            if isinstance(record, list):
                for item in record:
                    row = parse_dci_record(
                        item,
                        p,
                        photos_root,
                        cfg["manifest"]["max_mask_captions"],
                        cfg["manifest"]["min_mask_quality"],
                    )
                    if row:
                        rows.append(row)
            elif isinstance(record, dict):
                row = parse_dci_record(
                    record,
                    p,
                    photos_root,
                    cfg["manifest"]["max_mask_captions"],
                    cfg["manifest"]["min_mask_quality"],
                )
                if row:
                    rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}")

    # DCI's stable external key is the annotation filename / entry_key. Multiple
    # rows may point to the same image path, so de-duplicate on entry_key first.
    df = pd.DataFrame(rows).drop_duplicates(subset=["entry_key"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    missing = int((df["image_exists"] == 0).sum()) if len(df) else 0
    print(f"Saved manifest to {out_csv} with {len(df)} rows")
    print(f"[INFO] Missing resolved image paths: {missing}")


if __name__ == "__main__":
    main()
