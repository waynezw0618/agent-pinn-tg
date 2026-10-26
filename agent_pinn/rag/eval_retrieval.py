"""Retrieval metrics: Recall@k, HitRate@k, MRR, NDCG@10."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .. import paths
from .retrieve import get_kb


def _ndcg(rels: list[int], k: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(queries_path: str | Path, ks: tuple[int, ...] = (1, 3, 5),
             kb_dir: str | Path | None = None,
             limit: int | None = None) -> dict:
    rows = [json.loads(line) for line in
            Path(queries_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if limit:
        rows = rows[:limit]
    kb = get_kb(kb_dir)
    kmax = max(max(ks), 10)
    agg = {f"recall@{k}": [] for k in ks}
    agg.update({f"hit@{k}": [] for k in ks})
    rr, ndcg = [], []
    used_rerank = []
    for row in rows:
        gold = set(row["gold_chunk_ids"])
        res = kb.search(row["query"], k=kmax)
        used_rerank.append(1.0 if res.get("rerank_used") else 0.0)
        ids = [r["chunk_id"] for r in res["results"]]
        rels = [1 if i in gold else 0 for i in ids]
        for k in ks:
            top = ids[:k]
            agg[f"recall@{k}"].append(len(gold & set(top)) / max(1, len(gold)))
            agg[f"hit@{k}"].append(1.0 if gold & set(top) else 0.0)
        rr.append(1.0 / (1 + rels.index(1)) if 1 in rels else 0.0)
        ndcg.append(_ndcg(rels, 10))
    out = {k: (sum(v) / len(v) if v else 0.0) for k, v in agg.items()}
    out["mrr"] = sum(rr) / max(1, len(rr))
    out["ndcg@10"] = sum(ndcg) / max(1, len(ndcg))
    out["n_queries"] = len(rows)
    out["rerank_used"] = (sum(used_rerank) / len(used_rerank)) if rows else 0.0
    out["rerank_model"] = getattr(kb, "rerank_model", None)
    out["kind"] = rows[0].get("kind") if rows else None
    out["kb"] = {"chunks": len(kb.chunks), "model": kb.meta.get("model"),
                 "degraded": kb.meta.get("degraded")}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(paths.KB_QUERIES))
    ap.add_argument("--k", nargs="*", type=int, default=[1, 3, 5])
    ap.add_argument("--kb", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N queries (for quick runs)")
    a = ap.parse_args(argv)
    out = evaluate(a.queries, tuple(a.k), a.kb, limit=a.limit)
    # Tag by what actually happened, not by what was requested: a requested
    # reranker that fails to load must not overwrite the fusion baseline.
    tag = "rerank" if out.get("rerank_used") else "fusion"
    dest = paths.AGENT / f"rag_eval_{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(json.dumps(out, indent=2))
    th = {"hit@3": 0.75, "mrr": 0.55, "ndcg@10": 0.60}
    ok = all(out.get(k, 0) >= v for k, v in th.items())
    print("thresholds", th, "->", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
