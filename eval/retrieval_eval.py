from __future__ import annotations

import json
from pathlib import Path

from rag.ingest import ingest
from rag.retrieval import HybridRetriever


def run_retrieval_eval(root: Path) -> dict:
    if not (root / "data" / "vector_store.json").exists():
        ingest(root / "data" / "corpus.json", root / "data" / "vector_store.json")
    labels = json.loads((root / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    retriever = HybridRetriever(root)
    rows = []
    recall_total = 0.0
    mrr_total = 0.0
    for item in labels:
        hits = retriever.retrieve(item["question"], user_role="manager", top_k=5)
        ids = [h.chunk.chunk_id for h in hits]
        expected = set(item["expected"])
        found = [i for i, chunk_id in enumerate(ids, start=1) if chunk_id in expected]
        recall = 1.0 if found else 0.0
        mrr = 1.0 / found[0] if found else 0.0
        recall_total += recall
        mrr_total += mrr
        rows.append({"question": item["question"], "expected": item["expected"], "retrieved": ids, "recall@5": recall, "mrr": mrr})
    n = len(labels)
    return {"recall@5": round(recall_total / n, 3), "mrr": round(mrr_total / n, 3), "rows": rows}

