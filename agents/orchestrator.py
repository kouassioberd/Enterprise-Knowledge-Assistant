from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agents.messages import (
    Envelope,
    RetrievalRequest,
    SafetyReviewRequest,
    SynthesisRequest,
    new_correlation_id,
)
from agents.retriever_agent import RetrieverAgent
from agents.safety_reviewer import SafetyReviewerAgent
from agents.synthesizer_agent import SynthesizerAgent
from safety.guardrails import Guardrails


class Orchestrator:
    name = "orchestrator"

    def __init__(self, root: Path, max_rounds: int = 2):
        self.root = root
        self.max_rounds = max_rounds
        self.guardrails = Guardrails(root)
        self.retriever = RetrieverAgent(root)
        self.synthesizer = SynthesizerAgent()
        self.reviewer = SafetyReviewerAgent(root)
        self.trace_path = root / "logs" / "traces.jsonl"
        self.trace_path.parent.mkdir(exist_ok=True)

    def answer(self, question: str, user_role: str = "employee", correlation_id: str | None = None) -> dict:
        cid = correlation_id or new_correlation_id()
        self._trace(Envelope("user", self.name, "UserRequest", cid, {"question": question, "user_role": user_role}))
        input_decision = self.guardrails.check_input(question, user_role)
        if input_decision.decision == "reject":
            return {"decision": "reject", "answer": input_decision.reason, "trace_id": cid, "input_guardrail": asdict(input_decision)}
        safe_question = input_decision.text

        retrieval_request = RetrievalRequest(safe_question, top_k=5, user_role=user_role)
        self._trace(Envelope(self.name, "retriever", "RetrievalRequest", cid, asdict(retrieval_request)))
        retrieval_result = self.retriever.handle(retrieval_request)
        self._trace(Envelope("retriever", self.name, "RetrievalResult", cid, {"chunks": [c["chunk_id"] for c in retrieval_result.chunks]}))

        critique = None
        final_answer = None
        final_decision = "reject"
        for _round in range(self.max_rounds):
            synthesis_request = SynthesisRequest(safe_question, retrieval_result.chunks, critique)
            self._trace(Envelope(self.name, "synthesizer", "SynthesisRequest", cid, {"chunk_count": len(retrieval_result.chunks), "critique": critique}))
            synthesis_result = self.synthesizer.handle(synthesis_request)
            self._trace(Envelope("synthesizer", self.name, "SynthesisResult", cid, asdict(synthesis_result)))

            review_request = SafetyReviewRequest(safe_question, synthesis_result.answer, retrieval_result.chunks)
            self._trace(Envelope(self.name, "safety_reviewer", "SafetyReviewRequest", cid, {"draft_len": len(synthesis_result.answer)}))
            verdict = self.reviewer.handle(review_request)
            self._trace(Envelope("safety_reviewer", self.name, "SafetyVerdict", cid, asdict(verdict)))

            if verdict.decision in {"approve", "redact"}:
                final_decision = verdict.decision
                final_answer = verdict.answer
                break
            critique = "; ".join(verdict.issues)

        if final_answer is None:
            final_answer = "I cannot provide a safe, grounded answer from the retrieved corpus."
        return {"decision": final_decision, "answer": final_answer, "trace_id": cid, "input_guardrail": asdict(input_decision)}

    def _trace(self, envelope: Envelope) -> None:
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(envelope.to_dict()) + "\n")
