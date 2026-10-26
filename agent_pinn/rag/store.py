"""Tiny numpy vector store (a few hundred chunks do not need a database)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyVectorStore:
    def __init__(self, ids: list[str], vecs: np.ndarray):
        self.ids = list(ids)
        m = np.asarray(vecs, dtype=np.float32)
        if m.ndim != 2 or m.shape[0] != len(self.ids):
            raise ValueError(f"bad store shape {m.shape} vs {len(self.ids)}")
        # cosine == dot product because vectors are L2-normalised
        self.vecs = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True),
                                   1e-12)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.vecs.shape[1])

    def search(self, qvec: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sc = self.vecs @ q
        k = min(k, sc.size)
        idx = np.argpartition(-sc, k - 1)[:k]
        idx = idx[np.argsort(-sc[idx])]
        return [(int(i), float(sc[i])) for i in idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.array(self.ids), vecs=self.vecs)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyVectorStore":
        z = np.load(path, allow_pickle=False)
        return cls([str(x) for x in z["ids"]], z["vecs"])

    def meta_path(self, path: str | Path) -> Path:
        return Path(path).with_suffix(".meta.json")

    def save_meta(self, path: str | Path, meta: dict) -> None:
        self.meta_path(path).write_text(json.dumps(meta, indent=2),
                                        encoding="utf-8")
