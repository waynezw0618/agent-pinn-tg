"""Pure-Python BM25 with a bilingual tokenizer.

Chinese has no whitespace, so a char-bigram stream is emitted alongside the
latin word tokens.  This keeps the sparse retriever usable on a mixed
Chinese/English corpus without jieba.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_LATIN = re.compile(r"[a-z0-9][a-z0-9_\-'.]*")
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_STOP = set("""a an the of to in for on and or is are was were be been being
with without by as at from that this these those it its we our they their
which who whom whose than then so such not no can may might will would could
should do does did done have has had using used use into within between over
under about more most less least very much many few also both each other
""".split())


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    out: list[str] = []
    for w in _LATIN.findall(text):
        if w not in _STOP and len(w) > 1:
            out.append(w)
    for run in _CJK.findall(text):
        if len(run) == 1:
            out.append(run)
            continue
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
        out.extend(run)          # single chars help short queries
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.avg = (sum(len(t) for t in self.tokens) / max(1, self.n))
        self.tf = [Counter(t) for t in self.tokens]
        df: Counter = Counter()
        for t in self.tokens:
            df.update(set(t))
        self.idf = {w: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5))
                    for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * self.n
        if not q:
            return out
        for i in range(self.n):
            tf, dl = self.tf[i], len(self.tokens[i])
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                idf = self.idf.get(w, 0.0)
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / (self.avg or 1.0))
                s += idf * f * (self.k1 + 1.0) / (denom or 1.0)
            out[i] = s
        return out

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        sc = self.scores(query)
        idx = sorted(range(self.n), key=lambda i: -sc[i])[:k]
        return [(i, sc[i]) for i in idx if sc[i] > 0]
