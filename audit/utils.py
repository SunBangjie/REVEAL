from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Iterable, Any

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def iter_json_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for p in root.rglob("*.json"):
        if p.is_file():
            yield p


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(safe_text(v) for v in value if safe_text(v))
    if isinstance(value, dict):
        return " ".join(safe_text(v) for v in value.values() if safe_text(v))
    return str(value).strip()


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|")


def entropy_binary(p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number):
            return default
        return number
    except Exception:
        return default
