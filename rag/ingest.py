from __future__ import annotations

import json
from pathlib import Path

from rag.chunking import Chunk, chunk_document
from rag.vector_store import JsonVectorStore


def ingest(corpus_path: Path, store_path: Path) -> int:
    docs = json.loads(corpus_path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    JsonVectorStore(store_path).persist(chunks)
    return len(chunks)

