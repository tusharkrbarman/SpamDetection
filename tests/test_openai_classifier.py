"""Tests for the OpenAI transcript classifier."""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification_result import ClassificationResult
from src.openai_classifier import OpenAIClassifier


class TestOpenAIClassifier:
    """Test cases for OpenAI classifier."""

    @pytest.mark.asyncio
    async def test_classify_spam(self):
        classifier = OpenAIClassifier(api_key="sk-test-key", model="gpt-4o-mini")
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
            classifier.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await classifier.classify("Caller: We can approve your loan today")

        assert isinstance(result, ClassificationResult)
        assert result.is_spam is True
        assert result.confidence == 0.95
        assert result.reason == "Unsolicited loan offer"
        assert result.evidence_lines == ["Caller: We can approve your loan today"]

    @pytest.mark.asyncio
    async def test_string_false_is_not_truthy(self):
        classifier = OpenAIClassifier(api_key="sk-test-key")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "is_spam": "false",
                "confidence": 0.1,
                "reason": "Not enough evidence",
                "evidence_lines": [],
            }
        )

        with patch.object(
            classifier.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await classifier.classify("Caller: hello")

        assert result.is_spam is False

    @pytest.mark.asyncio
    async def test_classify_empty_transcript(self):
        classifier = OpenAIClassifier(api_key="sk-test-key")
        result = await classifier.classify("")

        assert result.is_spam is False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"
        assert result.evidence_lines == []

    @pytest.mark.asyncio
    async def test_error_handling(self):
        classifier = OpenAIClassifier(api_key="sk-test-key")

        with patch.object(
            classifier.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = Exception("API Error")
            result = await classifier.classify("Caller: test")

        assert result.is_spam is False
        assert result.confidence == 0.0
        assert "OpenAI classification error" in result.reason
