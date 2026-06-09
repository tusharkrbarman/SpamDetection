"""Shared data model for spam classification."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationResult:
    """Result of spam classification."""

    is_spam: bool
    confidence: float
    reason: str
    evidence_lines: list[str]
    full_transcript: str
