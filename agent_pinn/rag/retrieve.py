"""Hybrid retrieval: BM25 + dense -> RRF fusion -> cross-encoder rerank."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .. import config, paths
from .bm25 import BM25
from .embed import get_embedder
from .store import NumpyVectorStore

_CACHE: dict = {"kb": None}


class KnowledgeBase:
    def __init__(self, kb_dir: Path):
        self.dir = kb_dir
        chunks_path = kb_dir / "chunks.jsonl"
        index_path = kb_dir / "index.npz"
        if not chunks_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"knowledge base not built under {kb_dir} "
                f"(run `python -m agent_pinn.rag.ingest`)")
        self.chunks = [json.loads(line) for line in
                       chunks_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.store = NumpyVectorStore.load(index_path)
        meta_path = index_path.with_suffix(".meta.json")
        self.meta = (json.loads(meta_path.read_text(encoding="utf-8"))
                     if meta_path.is_file() else {})
        self.bm25 = BM25([c["text"] for c in self.chunks])
        self._embedder = None
        self._reranker = None
        self._rerank_failed = False

    # -- lazy heavy resources -----------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            model = None if self.meta.get("degraded") else self.meta.get("model")
            self._embedder = get_embedder(model)
            if self._embedder.dim != self.store.dim:
                raise RuntimeError(
                    f"embedding dim mismatch: index {self.store.dim} vs "
                    f"model {self._embedder.dim} ({self._embedder.name}); "
                    "re-run ingest with the same model")
        return self._embedder

    def reranker(self):
        if self._reranker is not None or self._rerank_failed:
            return self._reranker
        if not config.embed_settings()["rerank"]:
            self._rerank_failed = True
            return None
        from .embed import _resolve_local
        override = os.environ.get("AGENT_PINN_RERANK_MODEL")
        names = [override] if override else ["BAAI/bge-reranker-v2-m3",
                                             "BAAI/bge-reranker-base"]
        for name in names:
            path = _resolve_local(name)
            # A bare hub id here would stall on a 2 GB download mid-search;
            # offline-first, so only vendored copies are used.
            if path == name and "/" in name:
                continue
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(path, device="cpu")
                self.rerank_model = path
                break
            except Exception:  # noqa: BLE001
                self._reranker = None
        if self._reranker is None:
            self._rerank_failed = True
        return self._reranker

    # -- retrieval -----------------------------------------------------
    def search(self, query: str, k: int = 3, dense_k: int = 20,
               sparse_k: int = 20, use_rerank: bool = True) -> dict:
        dense = self.store.search(self.embedder.encode([query])[0], dense_k)
        sparse = self.bm25.search(query, sparse_k)

        # reciprocal rank fusion
        rrf: dict[int, float] = {}
        for rank, (i, _s) in enumerate(dense):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
        for rank, (i, _s) in enumerate(sparse):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
        fused = sorted(rrf.items(), key=lambda kv: -kv[1])[:max(k, 20)]

        rerank_used = False
        if use_rerank and len(fused) > 1:
            ce = self.reranker()
            if ce is not None:
                pairs = [(query, self.chunks[i]["text"]) for i, _ in fused]
                try:
                    scores = ce.predict(pairs, show_progress_bar=False)
                    order = np.argsort(-np.asarray(scores))
                    fused = [(fused[j][0], float(scores[j])) for j in order]
                    rerank_used = True
                except Exception:  # noqa: BLE001
                    pass

        results = []
        for rank, (i, score) in enumerate(fused[:k], start=1):
            c = self.chunks[i]
            results.append({
                "rank": rank, "score": float(score),
                "chunk_id": c["chunk_id"], "title": c["title"],
                "doc_id": c["doc_id"], "source_path": c["source_path"],
                "page": c["page"], "section": c["section"],
                "lang": c["lang"], "text": c["text"][:1200],
            })
        return {"query": query, "results": results,
                "rerank_used": rerank_used,
                "degraded": bool(self.meta.get("degraded")),
                "n_chunks": len(self.chunks)}


def get_kb(kb_dir: str | Path | None = None, refresh: bool = False) -> KnowledgeBase:
    d = Path(kb_dir) if kb_dir else paths.KB
    if refresh or _CACHE["kb"] is None or _CACHE["kb"].dir != d:
        _CACHE["kb"] = KnowledgeBase(d)
    return _CACHE["kb"]


def search(query: str, k: int = 3, **kw) -> dict:
    return get_kb().search(query, k=k, **kw)


if __name__ == "__main__":  # pragma: no cover - manual probe
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    print(json.dumps(search(a.query, a.k), indent=2, ensure_ascii=False)[:4000])
