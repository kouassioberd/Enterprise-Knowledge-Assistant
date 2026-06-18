from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from rag.chunking import Chunk
from rag.text import tokenize
from rag.vector_store import ROLE_RANK


@dataclass
class SparseHit:
    chunk: Chunk
    score: float
    rank: int


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.text + " " + c.title) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        df: defaultdict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] += 1
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, user_role: str, top_k: int = 12) -> list[SparseHit]:
        q_tokens = tokenize(query)
        allowed = ROLE_RANK[user_role]
        hits: list[SparseHit] = []
        for idx, chunk in enumerate(self.chunks):
            if ROLE_RANK[chunk.min_role] > allowed:
                continue
            counts = Counter(self.doc_tokens[idx])
            score = 0.0
            for token in q_tokens:
                tf = counts[token]
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                score += self.idf.get(token, 0.0) * ((tf * (self.k1 + 1)) / denom if denom else 0.0)
            hits.append(SparseHit(chunk, score, 0))
        hits.sort(key=lambda h: h.score, reverse=True)
        return [SparseHit(h.chunk, h.score, i + 1) for i, h in enumerate(hits[:top_k])]

