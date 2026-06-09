"""Spam classifier using OpenAI."""

import logging
import os
from pathlib import Path

from src.classification_result import ClassificationResult
from src.config import AppConfig
from src.openai_classifier import OpenAIClassifier

logger = logging.getLogger("spam-classifier")
logger.setLevel(logging.INFO)

# Re-export for backwards compatibility
__all__ = ["classify_transcript", "ClassificationResult"]


def _get_classifier() -> OpenAIClassifier:
    """Create the OpenAI classifier from environment configuration.

    Raises:
        ValueError: If OpenAI credentials are not configured
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key" or api_key.startswith("sk-or-"):
        raise ValueError("OpenAI API credentials are not configured")

    model = os.environ.get("SPAM_CLASSIFICATION_MODEL")
    classifier = OpenAIClassifier(api_key=api_key, model=model)

    logger.info("Using OpenAI classifier with model: %s", classifier.model)
    return classifier


async def classify_transcript(
    transcript: str,
    config_path: Path | None = None,
) -> ClassificationResult:
    """Classify a transcript as spam or legitimate.

    Args:
        transcript: The call transcript to classify
        config_path: Optional path to configuration file

    Returns:
        ClassificationResult with classification details
    """
    # Load config for validation
    app_config = AppConfig.load(config_path)

    # Validate transcript length
    if len(transcript) > app_config.spam_detection.max_transcript_length:
        logger.warning(
            "Transcript exceeds maximum length (%d > %d)",
            len(transcript),
            app_config.spam_detection.max_transcript_length,
        )
        return ClassificationResult(
            is_spam=False,
            confidence=0.0,
            reason="Transcript too long to analyze",
            evidence_lines=[],
            full_transcript=transcript,
        )

    # Check for empty transcript
    if not transcript.strip():
        return ClassificationResult(
            is_spam=False,
            confidence=0.0,
            reason="No transcript to analyze",
            evidence_lines=[],
            full_transcript=transcript,
        )

    classifier = None
    try:
        classifier = _get_classifier()
        result = await classifier.classify(transcript)
        return result

    except Exception as e:
        logger.error("Classification failed: %s", e, exc_info=True)
        return ClassificationResult(
            is_spam=False,
            confidence=0.0,
            reason=f"Classification error: {e}",
            evidence_lines=[],
            full_transcript=transcript,
        )
    finally:
        if classifier:
            await classifier.close()
