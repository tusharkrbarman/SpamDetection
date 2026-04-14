"""Spam classifier using provider abstraction layer."""

import logging
from pathlib import Path

from src.config import AppConfig, Provider
from src.providers import (
    LLMProvider,
    ClassificationResult,
    OpenRouterProvider,
    KiloProvider,
    OllamaProvider,
    OpenAIProvider,
)

logger = logging.getLogger("spam-classifier")
logger.setLevel(logging.INFO)

# Provider mapping
PROVIDER_MAP = {
    Provider.OPENROUTER: OpenRouterProvider,
    Provider.KILO: KiloProvider,
    Provider.OLLAMA: OllamaProvider,
    Provider.OPENAI: OpenAIProvider,
}


def _get_provider(config_path: Path | None = None) -> LLMProvider:
    """Get the appropriate LLM provider based on available credentials.

    Args:
        config_path: Optional path to configuration file

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If no provider credentials are configured
    """
    app_config = AppConfig.load(config_path)
    provider_config = app_config.spam_detection.get_available_provider()

    if not provider_config:
        raise ValueError("No LLM API credentials configured")

    # Determine provider type from API key prefix
    if provider_config.api_key.startswith("sk-or-"):
        provider_type = Provider.OPENROUTER
    elif provider_config.api_key.startswith("eyJ"):
        provider_type = Provider.KILO
    elif provider_config.base_url and "ollama" in provider_config.base_url.lower():
        provider_type = Provider.OLLAMA
    else:
        provider_type = Provider.OPENAI

    provider_class = PROVIDER_MAP[provider_type]
    provider = provider_class(
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        model=provider_config.model,
    )

    logger.info(f"Using {provider.get_provider_name()} with model: {provider.get_model_name()}")
    return provider


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

    provider = None
    try:
        provider = _get_provider(config_path)
        result = await provider.classify(transcript)
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
        if provider:
            await provider.close()
