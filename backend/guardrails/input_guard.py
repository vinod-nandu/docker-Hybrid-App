"""Deterministic input-side guardrails: length, PII, prompt-injection, blocklist."""
import re
from backend.config import settings

# Patterns for common PII (kept intentionally simple/deterministic)
_PII_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}

# Deterministic prompt-injection / jailbreak phrase blocklist (extend as needed)
_INJECTION_PATTERNS = [
    r"ignore (all|any|previous|prior) instructions",
    r"disregard (all|any|previous|prior) (instructions|rules)",
    r"you are now (in )?developer mode",
    r"reveal (your|the) system prompt",
    r"act as (an? )?unfiltered",
    r"jailbreak",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Basic disallowed-content keyword blocklist (illustrative, not exhaustive)
_BLOCKED_TOPICS_RE = re.compile(
    r"\b(bomb making|synthesize nerve agent|build a weapon|child sexual)\b",
    re.IGNORECASE,
)


class GuardrailResult:
    def __init__(self, allowed: bool, reason: str = "", redacted_text: str | None = None):
        self.allowed = allowed
        self.reason = reason
        self.redacted_text = redacted_text


def check_input(question: str) -> GuardrailResult:
    if not question or not question.strip():
        return GuardrailResult(False, "Empty query is not allowed.")

    if len(question) > settings.MAX_QUERY_CHARS:
        return GuardrailResult(False, f"Query exceeds max length of {settings.MAX_QUERY_CHARS} characters.")

    if _INJECTION_RE.search(question):
        return GuardrailResult(False, "Query blocked: potential prompt-injection pattern detected.")

    if _BLOCKED_TOPICS_RE.search(question):
        return GuardrailResult(False, "Query blocked: disallowed topic.")

    # Redact PII rather than hard-block, so benign questions still work
    redacted = question
    for label, pattern in _PII_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)

    return GuardrailResult(True, redacted_text=redacted)
