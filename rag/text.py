from __future__ import annotations

import math
import re
from hashlib import blake2b
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "should", "the", "to", "what",
    "when", "with", "do", "does", "me", "my", "can",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def hashed_embedding(text: str, dims: int = 256) -> list[float]:
    counts = Counter(tokenize(text))
    vec = [0.0] * dims
    for token, count in counts.items():
        digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
        vec[int.from_bytes(digest, "big") % dims] += 1.0 + math.log(count)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
