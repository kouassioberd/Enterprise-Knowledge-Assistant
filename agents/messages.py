from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
from uuid import uuid4


MessageType = Literal["UserRequest", "RetrievalRequest", "RetrievalResult", "SynthesisRequest", "SynthesisResult", "SafetyReviewRequest", "SafetyVerdict"]


@dataclass
class Envelope:
    sender: str
    recipient: str
    message_type: MessageType
    correlation_id: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.sender or not self.recipient or not self.message_type:
            raise ValueError("Envelope sender, recipient, and message_type are required.")
        if self.payload is None:
            raise ValueError("Envelope payload must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalRequest:
    query: str
    top_k: int
    user_role: str


@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]]


@dataclass
class SynthesisRequest:
    question: str
    chunks: list[dict[str, Any]]
    critique: str | None = None


@dataclass
class SynthesisResult:
    answer: str
    citations: list[str]


@dataclass
class SafetyReviewRequest:
    question: str
    answer: str
    chunks: list[dict[str, Any]]


@dataclass
class SafetyVerdict:
    decision: Literal["approve", "redact", "regenerate", "reject"]
    answer: str
    issues: list[str]


def new_correlation_id() -> str:
    return str(uuid4())

