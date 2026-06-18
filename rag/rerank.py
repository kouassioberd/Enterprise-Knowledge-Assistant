from __future__ import annotations

from dataclasses import dataclass

from rag.chunking import Chunk
from rag.text import cosine, hashed_embedding


@dataclass
class RankedChunk:
    chunk: Chunk
    dense_score: float
    sparse_score: float
    fused_score: float
    rerank_score: float


def mmr_rerank(candidates: list[RankedChunk], query: str, top_k: int = 5, lambda_: float = 0.72) -> list[RankedChunk]:
    q = hashed_embedding(query)
    selected: list[RankedChunk] = []
    remaining = candidates[:]
    embeddings = {c.chunk.chunk_id: hashed_embedding(c.chunk.text) for c in remaining}
    while remaining and len(selected) < top_k:
        best = None
        best_score = -10.0
        for candidate in remaining:
            relevance = cosine(q, embeddings[candidate.chunk.chunk_id])
            diversity_penalty = max(
                [cosine(embeddings[candidate.chunk.chunk_id], embeddings[s.chunk.chunk_id]) for s in selected],
                default=0.0,
            )
            score = lambda_ * relevance - (1 - lambda_) * diversity_penalty + candidate.fused_score
            if score > best_score:
                best = candidate
                best_score = score
        assert best is not None
        remaining.remove(best)
        selected.append(RankedChunk(best.chunk, best.dense_score, best.sparse_score, best.fused_score, best_score))
    return selected

