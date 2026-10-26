"""Embedding backends with explicit degradation.

Order: ``bge-m3`` -> ``bge-base-zh-v1.5`` -> offline hashing embedder.

``all-MiniLM-L6-v2`` is deliberately *not* used: it is English-only, and a
published ablation on a Chinese-query/English-corpus PINN assistant showed
HitRate@3 dropping from 48.6% to 20.0% because Chinese pseudo-documents become
a semantic black hole for it.  Use a bilingual model or none at all.
"""
from __future__ import annotations

import hashlib
import queue
import re
import threading

import numpy as np

from .. import config


def _load_with_timeout(fn, timeout_s: float):
    """Run ``fn`` in a daemon thread; return ``None`` on timeout/failure."""
    q: queue.Queue = queue.Queue()

    def target():
        try:
            q.put(("ok", fn()))
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc))

    th = threading.Thread(target=target, daemon=True)
    th.start()
    try:
        status, payload = q.get(timeout=timeout_s)
    except queue.Empty:
        return None
    return payload if status == "ok" else None


def _resolve_local(name: str) -> str:
    """Prefer a vendored copy under ``<root>/models/`` over a hub download.

    The HuggingFace mirror answers ``429`` for large repos often enough that a
    half-written snapshot appears in the cache and every later load fails on a
    missing ``config_sentence_transformers.json``.  Pointing at a local
    directory removes the network from the hot path entirely.
    """
    from ..paths import ROOT
    for cand in (name.replace("/", "--"), name.split("/")[-1], name):
        p = ROOT / "models" / cand
        if (p / "config.json").is_file():
            return str(p)
    return name


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, timeout_s: float = 600.0):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self._model = _load_with_timeout(
            lambda: SentenceTransformer(model_name, device="cpu"), timeout_s)
        if self._model is None:
            raise RuntimeError(f"could not load {model_name}")
        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:  # noqa: BLE001
            self.dim = int(self.encode(["probe"]).shape[1])
        self.degraded = False

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        v = self._model.encode(texts, batch_size=batch_size,
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


class HashingEmbedder:
    """Deterministic offline fallback (char 3-gram hashing, L2-normalised)."""

    name = "hashing-char3g"
    degraded = True

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        t = re.sub(r"\s+", " ", (text or "").lower())
        toks = re.findall(r"[a-z0-9]+", t)
        cjk = re.findall(r"[\u4e00-\u9fff]+", t)
        for run in cjk:
            toks.extend(run[i:i + 3] for i in range(max(1, len(run) - 2)))
        return toks

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._tokens(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-12)


def get_embedder(model: str | None = None, timeout_s: float = 600.0):
    cfg = config.embed_settings()
    primary = model or cfg["model"]
    chain = [primary, cfg["fallback"], "BAAI/bge-small-zh-v1.5"]
    for name in chain:
        try:
            emb = SentenceTransformerEmbedder(_resolve_local(name),
                                              timeout_s=timeout_s)
            emb.requested = name
            return emb
        except Exception:  # noqa: BLE001
            continue
    return HashingEmbedder()
