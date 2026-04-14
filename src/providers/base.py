"""Base provider interface for LLM classification providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassificationResult:
    """Result of spam classification."""
    is_spam: bool
    confidence: float
    reason: str
    evidence_lines: list[str]
    full_transcript: str


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, base_url: str | None = None):
        """Initialize the provider.

        Args:
            api_key: API key for authentication
            base_url: Optional base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def classify(self, transcript: str) -> ClassificationResult:
        """Classify a transcript as spam or legitimate.

        Args:
            transcript: The call transcript to classify

        Returns:
            ClassificationResult with classification details
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name used by this provider.

        Returns:
            Model name string
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name.

        Returns:
            Provider name string
        """
        pass
