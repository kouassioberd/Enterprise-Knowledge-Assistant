from __future__ import annotations

from pathlib import Path

from agents.messages import SafetyReviewRequest, SafetyVerdict
from safety.guardrails import Guardrails


class SafetyReviewerAgent:
    name = "safety_reviewer"

    def __init__(self, root: Path):
        self.guardrails = Guardrails(root)

    def handle(self, request: SafetyReviewRequest) -> SafetyVerdict:
        cited = {chunk["chunk_id"]: chunk["text"] for chunk in request.chunks}
        decision = self.guardrails.check_output(request.answer, cited)
        if decision.decision == "approve":
            return SafetyVerdict("approve", decision.text, [])
        if decision.decision == "redact":
            return SafetyVerdict("redact", decision.text, [decision.reason])
        if decision.decision == "regenerate":
            return SafetyVerdict("regenerate", decision.text, [decision.reason])
        return SafetyVerdict("reject", "I cannot provide that response safely.", [decision.reason])

