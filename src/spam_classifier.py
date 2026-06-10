"""Spam classifier using the configured text-to-text provider."""

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

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "agent_config.toml"


def _classification_error_reason(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    error_text = str(error).lower()
    if status_code == 429 or "resource_exhausted" in error_text or "quota" in error_text:
        return "Classification error: LLM quota exceeded; try again later"
    return "Classification error: provider request failed"


def _get_classifier(app_config: AppConfig):
    """Create the classifier from environment configuration.

    Raises:
        ValueError: If provider credentials are not configured
    """
    provider = os.environ.get(
        "SPAM_CLASSIFIER_PROVIDER",
        os.environ.get(
            "LLM_PROVIDER",
            app_config.spam_detection.classification_provider,
        ),
    ).lower()
    model = os.environ.get("SPAM_CLASSIFICATION_MODEL") or app_config.spam_detection.classification_model

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key":
            raise ValueError("Gemini API credentials are not configured")

        from src.gemini_llm import GeminiClassifier

        classifier = GeminiClassifier(api_key=api_key, model=model)
        logger.info("Using Gemini classifier with model: %s", classifier.model)
        return classifier

    if provider != "openai":
        raise ValueError(f"Unsupported spam classifier provider: {provider}")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key" or api_key.startswith("sk-or-"):
        raise ValueError("OpenAI API credentials are not configured")

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
    app_config = AppConfig.load(config_path or DEFAULT_CONFIG_PATH)

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
        classifier = _get_classifier(app_config)
        result = await classifier.classify(transcript)
        return result

    except Exception as e:
        logger.error("Classification failed: %s", e, exc_info=True)
        return ClassificationResult(
            is_spam=False,
            confidence=0.0,
            reason=_classification_error_reason(e),
            evidence_lines=[],
            full_transcript=transcript,
        )
    finally:
        if classifier:
            await classifier.close()
