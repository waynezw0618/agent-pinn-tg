"""Build the literature index: chunk -> embed -> (optional) DB mirror.

Usage
-----
python -m agent_pinn.rag.ingest --sources extracted outputs/report \
       --pdfs "Turbulent Secondary Flows.pdf" --out agent_pinn/kb

The chunker is deliberately simple (800 chars, 120 overlap, paragraph-aware)
and keeps page/section metadata so a proposal can cite an inspectable span.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from .. import db, paths, repo
from .bm25 import tokenize
from .embed import get_embedder
from .store import NumpyVectorStore

HEADING = re.compile(r"^(?:#{1,4}\s+\S|\d+(?:\.\d+)*\.?\s+\S|"
                     r"\b[A-Z][A-Z0-9 \-:,]{6,60}$)")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", text)


def _lang(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return "zh" if cjk > len(text) * 0.05 else "en"


def chunk_text(text: str, size: int = 800, overlap: int = 120,
               section: str | None = None) -> list[dict]:
    """Paragraph-aware fixed-size chunking with overlap."""
    text = _clean(text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[dict] = []
    buf, cur_sec = "", section
    start = 0

    def flush(end: int):
        nonlocal buf, start
        if len(buf.strip()) >= 60:
            out.append({"text": buf.strip(), "section": cur_sec,
                        "start": start})
        buf = ""
        start = end

    for p in paras:
        head = p.split("\n")[0].strip()
        if len(head) < 80 and HEADING.match(head):
            cur_sec = head
        if len(p) > size:
            if buf:
                flush(start + len(buf))
            for i in range(0, len(p), max(1, size - overlap)):
                piece = p[i:i + size]
                if len(piece.strip()) >= 60:
                    out.append({"text": piece.strip(), "section": cur_sec,
                                "start": start + i})
            start += len(p)
            continue
        if len(buf) + len(p) + 2 > size and buf:
            flush(start + len(buf))
            # overlap: keep the tail of the previous chunk
            buf = buf[-overlap:] if overlap else ""
            start = start + len(buf)
        buf = (buf + "\n\n" + p).strip() if buf else p
    if buf:
        flush(start + len(buf))
    return out


def _shingles(text: str, n: int = 5, cap: int = 6000) -> set[str]:
    words = re.findall(r"[\w\u4e00-\u9fff]+", text[:cap].lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n))}


def dedup_docs(docs: list[dict], threshold: float = 0.5) -> tuple[list[dict], list[str]]:
    """Drop documents that are near-duplicates of an earlier, cleaner one.

    The corpus contains both ``extracted/*.txt`` and the source PDFs; keeping
    both would put two near-identical chunks in competition for every query.
    Sources are processed before PDFs, so the text extraction wins.

    Uses *containment* (|intersection| / |smaller set|) rather than Jaccard so
    that a short PDF page which is fully contained in a long extraction is
    still recognised as a duplicate.
    """
    kept: list[dict] = []
    dropped: list[str] = []
    sets: list[set[str]] = []
    for d in docs:
        s = _shingles(d["text"])
        dup = False
        for prev in sets:
            if not s or not prev:
                continue
            inter = len(s & prev)
            if inter / max(1, min(len(s), len(prev))) > threshold:
                dup = True
                break
        if dup:
            dropped.append(d["doc_id"])
            continue
        kept.append(d)
        sets.append(s)
    return kept, dropped


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _name_duplicate(stem: str, existing: list[str]) -> bool:
    """True when a PDF's stem matches an already-ingested text extraction."""
    n = _norm_name(stem)
    for e in existing:
        m = _norm_name(e)
        if len(m) < 16 or len(n) < 16:
            continue
        if n.startswith(m) or m.startswith(n):
            return True
    return False


def collect_docs(sources: list[str], pdfs: list[str]) -> list[dict]:
    docs: list[dict] = []
    for src in sources:
        p = Path(src)
        if not p.is_absolute():
            p = paths.ROOT / p
        if p.is_dir():
            files = sorted([*p.glob("*.txt"), *p.glob("*.md")])
        else:
            files = [p]
        for f in files:
            if f.name.startswith("_") or f.name.startswith("."):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            first = next((ln.strip() for ln in text.splitlines()
                          if ln.strip()), f.stem)
            docs.append({"doc_id": f.stem, "title": first[:120],
                         "text": text, "page": None,
                         "source_path": paths.rel(f)})
    for pdf in pdfs:
        p = Path(pdf)
        if not p.is_absolute():
            cand = paths.ROOT / pdf
            p = cand if cand.is_file() else p
        if not p.is_file():
            continue
        if _name_duplicate(p.stem, [d["doc_id"] for d in docs]):
            continue          # already present as a cleaner text extraction
        try:
            import fitz
        except ImportError:
            continue
        with fitz.open(str(p)) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if len(text.strip()) < 200:
                    continue
                docs.append({"doc_id": f"{p.stem}_p{i + 1}",
                             "title": p.stem,
                             "text": text, "page": i + 1,
                             "source_path": paths.rel(p)})
    return docs


def build(sources: list[str], pdfs: list[str], out_dir: str | Path,
          model: str | None = None, size: int = 800, overlap: int = 120,
          query_model_timeout: float = 600.0,
          mirror_db: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = collect_docs(sources, pdfs)
    docs, dropped = dedup_docs(docs)
    chunks: list[dict] = []
    for d in docs:
        for c in chunk_text(d["text"], size=size, overlap=overlap):
            cid = hashlib.sha1(
                f"{d['doc_id']}|{d['page']}|{c['start']}".encode()
            ).hexdigest()[:16]
            chunks.append({
                "chunk_id": cid, "doc_id": d["doc_id"], "title": d["title"],
                "source_path": d["source_path"], "page": d["page"],
                "section": c["section"], "text": c["text"],
                "lang": _lang(c["text"]),
            })

    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    embedder = get_embedder(model, timeout_s=query_model_timeout)
    vecs = embedder.encode([c["text"] for c in chunks])
    store = NumpyVectorStore([c["chunk_id"] for c in chunks], vecs)
    index_path = out_dir / "index.npz"
    store.save(index_path)
    store.save_meta(index_path, {"model": embedder.name,
                                 "dim": int(vecs.shape[1]),
                                 "degraded": bool(embedder.degraded),
                                 "n_chunks": len(chunks),
                                 "built_at": db.now()})

    _write_eval_queries(chunks, out_dir / "eval_queries.jsonl")

    mirrored = 0
    if mirror_db:
        try:
            conn = db.connect(paths.AGENT_DB)
            mirrored = repo.add_literature(conn, chunks)
            conn.close()
        except Exception:  # noqa: BLE001 - mirroring is a convenience
            mirrored = 0

    return {"docs": len(docs), "chunks": len(chunks),
            "dropped_duplicate_docs": dropped,
            "index": str(index_path), "embed_model": embedder.name,
            "degraded": bool(embedder.degraded), "dim": int(vecs.shape[1]),
            "mirrored_to_db": mirrored}


def _write_eval_queries(chunks: list[dict], path: Path, n: int = 40) -> None:
    """Self-retrieval sanity set: query = a trailing sentence of the chunk.

    This measures whether the index can find a chunk from a sentence that is
    *not* at its beginning.  It is a sanity check, not a semantic QA benchmark.
    """
    rng = random.Random(0)
    pool = [c for c in chunks if len(c["text"]) > 300]
    rng.shuffle(pool)
    rows = []
    for c in pool[:n]:
        sents = [s.strip() for s in re.split(r"(?<=[.?!。？])\s+", c["text"])
                 if len(s.strip()) > 60]
        q = sents[-1][:400] if sents else c["text"][-300:]
        rows.append({"query": q, "gold_chunk_ids": [c["chunk_id"]],
                     "kind": "self_retrieval"})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=["extracted",
                                                     "outputs/report"])
    ap.add_argument("--pdfs", nargs="*", default=[])
    ap.add_argument("--out", default=str(paths.KB))
    ap.add_argument("--model", default=None)
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args(argv)
    res = build(args.sources, args.pdfs, args.out, model=args.model,
                size=args.size, overlap=args.overlap,
                query_model_timeout=args.timeout, mirror_db=not args.no_db)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
