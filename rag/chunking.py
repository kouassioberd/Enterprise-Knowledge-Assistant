from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    min_role: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_document(doc: dict, chunk_size: int = 90, overlap: int = 20) -> Iterable[Chunk]:
    words = doc["text"].split()
    step = max(1, chunk_size - overlap)
    for index, start in enumerate(range(0, len(words), step)):
        part = words[start : start + chunk_size]
        if not part:
            continue
        yield Chunk(
            chunk_id=f"{doc['doc_id']}#{index}",
            doc_id=doc["doc_id"],
            title=doc["title"],
            source=doc["source"],
            min_role=doc["min_role"],
            text=" ".join(part),
        )
        if start + chunk_size >= len(words):
            break

