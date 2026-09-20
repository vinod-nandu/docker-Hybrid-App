"""Deterministic output-side guardrails: grounding check + disallowed-content filter."""
import re
from backend.config import settings

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "it", "this", "that", "with", "as", "by", "be", "at", "from",
}

_BLOCKED_OUTPUT_RE = re.compile(
    r"\b(bomb making|synthesize nerve agent|child sexual)\b",
    re.IGNORECASE,
)

_REFUSAL_MARKER = "not found in the provided document(s)"


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class OutputGuardResult:
    def __init__(self, allowed: bool, reason: str = "", overlap: float = 0.0):
        self.allowed = allowed
        self.reason = reason
        self.overlap = overlap


def check_grounding(answer: str, context_chunks: list[str]) -> OutputGuardResult:
    """Lexical-overlap grounding check: flags answers with little support in retrieved context."""
    if _REFUSAL_MARKER in answer.lower():
        # Model explicitly said it couldn't answer from context -- that's fine, not a violation.
        return OutputGuardResult(True, overlap=1.0)

    if _BLOCKED_OUTPUT_RE.search(answer):
        return OutputGuardResult(False, "Answer blocked: disallowed content detected.")

    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return OutputGuardResult(False, "Answer blocked: empty or non-substantive response.")

    context_tokens: set[str] = set()
    for c in context_chunks:
        context_tokens |= _tokenize(c)

    if not context_tokens:
        return OutputGuardResult(False, "Answer blocked: no retrieved context to ground the answer.")

    overlap_count = len(answer_tokens & context_tokens)
    overlap_ratio = overlap_count / max(len(answer_tokens), 1)

    if overlap_ratio < settings.MIN_GROUNDING_OVERLAP:
        return OutputGuardResult(
            False,
            f"Answer blocked: insufficient grounding in retrieved sources (overlap={overlap_ratio:.2f}).",
            overlap=overlap_ratio,
        )

    return OutputGuardResult(True, overlap=overlap_ratio)
