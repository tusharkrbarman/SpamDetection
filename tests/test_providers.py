"""Tests for provider implementations."""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from src.providers import (
    OpenRouterProvider,
    KiloProvider,
    OllamaProvider,
    OpenAIProvider,
    ClassificationResult,
)


class TestOpenRouterProvider:
    """Test cases for OpenRouter provider."""

    @pytest.mark.asyncio
    async def test_classify_spam(self):
        """Test spam classification with OpenRouter."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            base_url="https://openrouter.ai/api/v1",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_spam": True,
            "confidence": 0.95,
            "reason": "Unsolicited financial product offer",
            "evidence_lines": ["Caller: Hello, I'm calling from Bank of America..."]
        })

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await provider.classify("Caller: Hello, I'm calling from Bank of America...")

        assert result.is_spam == True
        assert result.confidence == 0.95
        assert result.reason == "Unsolicited financial product offer"
        assert len(result.evidence_lines) == 1

        await provider.close()

    @pytest.mark.asyncio
    async def test_classify_legitimate(self):
        """Test legitimate classification with OpenRouter."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            base_url="https://openrouter.ai/api/v1",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_spam": False,
            "confidence": 0.90,
            "reason": "Legitimate community member",
            "evidence_lines": []
        })

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await provider.classify("Caller: Hi, this is John from Nextdoor.")

        assert result.is_spam == False
        assert result.confidence == 0.90
        assert result.reason == "Legitimate community member"
        assert result.evidence_lines == []

        await provider.close()

    @pytest.mark.asyncio
    async def test_empty_transcript(self):
        """Test classification with empty transcript."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            base_url="https://openrouter.ai/api/v1",
        )

        result = await provider.classify("")

        assert result.is_spam == False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"

        await provider.close()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in OpenRouter provider."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            base_url="https://openrouter.ai/api/v1",
        )

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")

            result = await provider.classify("Some transcript")

        assert result.is_spam == False
        assert result.confidence == 0.0
        assert "OpenRouter classification error" in result.reason

        await provider.close()

    def test_get_provider_name(self):
        """Test provider name."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert provider.get_provider_name() == "openrouter"

    def test_get_model_name(self):
        """Test model name."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            base_url="https://openrouter.ai/api/v1",
            model="custom-model",
        )
        assert provider.get_model_name() == "custom-model"


class TestKiloProvider:
    """Test cases for Kilo provider."""

    @pytest.mark.asyncio
    async def test_classify_spam(self):
        """Test spam classification with Kilo."""
        provider = KiloProvider(
            api_key="eyJtest-key",
            base_url="https://kilo.ai/api/v1",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_spam": True,
            "confidence": 0.85,
            "reason": "Sales call detected",
            "evidence_lines": ["Caller: We have a special offer..."]
        })

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await provider.classify("Caller: We have a special offer...")

        assert result.is_spam == True
        assert result.confidence == 0.85

        await provider.close()

    def test_get_provider_name(self):
        """Test provider name."""
        provider = KiloProvider(
            api_key="eyJtest-key",
            base_url="https://kilo.ai/api/v1",
        )
        assert provider.get_provider_name() == "kilo"


class TestOllamaProvider:
    """Test cases for Ollama provider."""

    @pytest.mark.asyncio
    async def test_classify_spam(self):
        """Test spam classification with Ollama."""
        provider = OllamaProvider(
            api_key="test-key",
            base_url="http://localhost:11434",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_spam": True,
            "confidence": 0.80,
            "reason": "Promotional content",
            "evidence_lines": ["Caller: Limited time offer..."]
        })

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await provider.classify("Caller: Limited time offer...")

        assert result.is_spam == True
        assert result.confidence == 0.80

        await provider.close()

    def test_get_provider_name(self):
        """Test provider name."""
        provider = OllamaProvider(
            api_key="test-key",
            base_url="http://localhost:11434",
        )
        assert provider.get_provider_name() == "ollama"


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_classify_spam(self):
        """Test spam classification with OpenAI."""
        provider = OpenAIProvider(
            api_key="sk-test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_spam": True,
            "confidence": 0.92,
            "reason": "Scam attempt detected",
            "evidence_lines": ["Caller: Your account will be closed..."]
        })

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await provider.classify("Caller: Your account will be closed...")

        assert result.is_spam == True
        assert result.confidence == 0.92

        await provider.close()

    def test_get_provider_name(self):
        """Test provider name."""
        provider = OpenAIProvider(
            api_key="sk-test-key",
        )
        assert provider.get_provider_name() == "openai"


class TestClassificationResult:
    """Test cases for ClassificationResult dataclass."""

    def test_classification_result_creation(self):
        """Test creating a ClassificationResult."""
        result = ClassificationResult(
            is_spam=True,
            confidence=0.95,
            reason="Test reason",
            evidence_lines=["line1", "line2"],
            full_transcript="Test transcript",
        )

        assert result.is_spam == True
        assert result.confidence == 0.95
        assert result.reason == "Test reason"
        assert result.evidence_lines == ["line1", "line2"]
        assert result.full_transcript == "Test transcript"

    def test_classification_result_immutability(self):
        """Test that ClassificationResult is immutable."""
        result = ClassificationResult(
            is_spam=True,
            confidence=0.95,
            reason="Test reason",
            evidence_lines=[],
            full_transcript="Test transcript",
        )

        # Should raise AttributeError when trying to modify
        with pytest.raises(AttributeError):
            result.is_spam = False
