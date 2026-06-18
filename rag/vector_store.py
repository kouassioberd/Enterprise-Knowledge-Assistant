from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.chunking import Chunk
from rag.text import cosine, hashed_embedding


ROLE_RANK = {"intern": 0, "employee": 1, "manager": 2, "security": 3}


@dataclass
class DenseHit:
    chunk: Chunk
    score: float
    rank: int


class JsonVectorStore:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict] = []
        if path.exists():
            self.records = json.loads(path.read_text(encoding="utf-8"))

    def persist(self, chunks: list[Chunk]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = [
            {"chunk": chunk.to_dict(), "embedding": hashed_embedding(chunk.text)}
            for chunk in chunks
        ]
        self.path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def chunks(self) -> list[Chunk]:
        return [Chunk(**record["chunk"]) for record in self.records]

    def search(self, query: str, user_role: str, top_k: int = 12) -> list[DenseHit]:
        q = hashed_embedding(query)
        allowed = ROLE_RANK[user_role]
        hits: list[DenseHit] = []
        for record in self.records:
            chunk = Chunk(**record["chunk"])
            if ROLE_RANK[chunk.min_role] > allowed:
                continue
            hits.append(DenseHit(chunk=chunk, score=cosine(q, record["embedding"]), rank=0))
        hits.sort(key=lambda h: h.score, reverse=True)
        return [DenseHit(h.chunk, h.score, i + 1) for i, h in enumerate(hits[:top_k])]

