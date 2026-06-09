"""Tests for the OpenAI classification provider."""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.providers.openai import OpenAIProvider
from src.providers.base import ClassificationResult


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_classify_spam(self):
        """Test spam classification with OpenAI."""
        provider = OpenAIProvider(api_key="sk-test-key", model="gpt-4o-mini")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "is_spam": True,
                "confidence": 0.95,
                "reason": "Unsolicited loan offer",
                "evidence_lines": ["Caller: We can approve your loan today"],
            }
        )

        with patch.object(
            provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await provider.classify("Caller: We can approve your loan today")

        assert isinstance(result, ClassificationResult)
        assert result.is_spam is True
        assert result.confidence == 0.95
        assert result.reason == "Unsolicited loan offer"
        assert result.evidence_lines == ["Caller: We can approve your loan today"]

    @pytest.mark.asyncio
    async def test_classify_empty_transcript(self):
        """Test empty transcript handling."""
        provider = OpenAIProvider(api_key="sk-test-key")
        result = await provider.classify("")

        assert result.is_spam is False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"
        assert result.evidence_lines == []

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test OpenAI provider error handling."""
        provider = OpenAIProvider(api_key="sk-test-key")

        with patch.object(
            provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = Exception("API Error")
            result = await provider.classify("Caller: test")

        assert result.is_spam is False
        assert result.confidence == 0.0
        assert "OpenAI classification error" in result.reason

    def test_provider_metadata(self):
        """Test provider metadata."""
        provider = OpenAIProvider(api_key="sk-test-key", model="gpt-4o-mini")

        assert provider.get_provider_name() == "openai"
        assert provider.get_model_name() == "gpt-4o-mini"
