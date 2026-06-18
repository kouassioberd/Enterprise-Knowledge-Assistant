from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.orchestrator import Orchestrator
from eval.red_team_eval import run_red_team
from eval.retrieval_eval import run_retrieval_eval
from llm.openrouter_client import load_dotenv
from rag.ingest import ingest


ROOT = Path(__file__).parent


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Enterprise Knowledge Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest")

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--role", default="employee", choices=["intern", "employee", "manager", "security"])
    ask.add_argument("--correlation-id", default=None)

    sub.add_parser("eval")
    sub.add_parser("redteam")

    args = parser.parse_args()

    if args.command == "ingest":
        count = ingest(ROOT / "data" / "corpus.json", ROOT / "data" / "vector_store.json")
        print(f"Ingested {count} chunks into data/vector_store.json")
        return

    if args.command == "ask":
        orchestrator = Orchestrator(ROOT)
        result = orchestrator.answer(args.question, args.role, args.correlation_id)
        print(json.dumps(result, indent=2))
        return

    if args.command == "eval":
        print(json.dumps(run_retrieval_eval(ROOT), indent=2))
        return

    if args.command == "redteam":
        print(json.dumps(run_red_team(ROOT), indent=2))


if __name__ == "__main__":
    main()
