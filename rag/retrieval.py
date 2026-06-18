from __future__ import annotations

import json
from pathlib import Path

from rag.rerank import RankedChunk, mmr_rerank
from rag.sparse import BM25Index
from rag.vector_store import JsonVectorStore


class HybridRetriever:
    def __init__(self, root: Path):
        self.root = root
        self.store = JsonVectorStore(root / "data" / "vector_store.json")
        self.sparse = BM25Index(self.store.chunks())
        self.log_path = root / "logs" / "retrieval.jsonl"
        self.log_path.parent.mkdir(exist_ok=True)

    def retrieve(self, query: str, user_role: str, top_k: int = 5) -> list[RankedChunk]:
        dense = self.store.search(query, user_role, top_k=12)
        sparse = self.sparse.search(query, user_role, top_k=12)
        by_id: dict[str, RankedChunk] = {}

        for hit in dense:
            by_id[hit.chunk.chunk_id] = RankedChunk(hit.chunk, hit.score, 0.0, 1 / (60 + hit.rank), 0.0)
        for hit in sparse:
            current = by_id.get(hit.chunk.chunk_id)
            if current:
                current.sparse_score = hit.score
                current.fused_score += 1 / (60 + hit.rank)
            else:
                by_id[hit.chunk.chunk_id] = RankedChunk(hit.chunk, 0.0, hit.score, 1 / (60 + hit.rank), 0.0)

        fused = sorted(by_id.values(), key=lambda x: x.fused_score, reverse=True)
        reranked = mmr_rerank(fused[:10], query, top_k=top_k)
        self._log(query, reranked)
        return reranked

    def _log(self, query: str, chunks: list[RankedChunk]) -> None:
        event = {
            "query": query,
            "results": [
                {
                    "chunk_id": c.chunk.chunk_id,
                    "dense": round(c.dense_score, 4),
                    "sparse": round(c.sparse_score, 4),
                    "fused": round(c.fused_score, 4),
                    "rerank": round(c.rerank_score, 4),
                }
                for c in chunks
            ],
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

