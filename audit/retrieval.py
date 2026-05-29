from __future__ import annotations

import numpy as np

from audit.utils import normalize_rows


class CosineIndex:
    def __init__(self, embeddings: np.ndarray, ids: list[str]):
        self.embeddings = normalize_rows(embeddings.astype(np.float32))
        self.ids = list(ids)
        self.id_to_pos = {k: i for i, k in enumerate(self.ids)}

    def query(self, x: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, list[list[str]]]:
        x = normalize_rows(x.astype(np.float32))
        sims = x @ self.embeddings.T
        idx = np.argsort(-sims, axis=1)[:, :top_k]
        vals = np.take_along_axis(sims, idx, axis=1)
        ids = [[self.ids[j] for j in row] for row in idx]
        return vals, ids
