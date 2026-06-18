from __future__ import annotations

from pathlib import Path

from agents.messages import RetrievalRequest, RetrievalResult
from rag.retrieval import HybridRetriever


class RetrieverAgent:
    name = "retriever"

    def __init__(self, root: Path):
        self.retriever = HybridRetriever(root)

    def handle(self, request: RetrievalRequest) -> RetrievalResult:
        chunks = self.retriever.retrieve(request.query, request.user_role, request.top_k)
        return RetrievalResult(
            chunks=[
                {
                    "chunk_id": c.chunk.chunk_id,
                    "title": c.chunk.title,
                    "source": c.chunk.source,
                    "min_role": c.chunk.min_role,
                    "text": c.chunk.text,
                    "dense_score": c.dense_score,
                    "sparse_score": c.sparse_score,
                    "fused_score": c.fused_score,
                    "rerank_score": c.rerank_score,
                }
                for c in chunks
            ]
        )

