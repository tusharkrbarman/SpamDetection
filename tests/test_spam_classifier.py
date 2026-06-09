"""Tests for spam classifier."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification_result import ClassificationResult
from src.spam_classifier import classify_transcript


class TestSpamClassifier:
    """Test cases for spam_classifier.py."""

    @pytest.mark.asyncio
    async def test_empty_transcript(self):
        result = await classify_transcript("")

        assert isinstance(result, ClassificationResult)
        assert result.is_spam is False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"
        assert result.evidence_lines == []
        assert result.full_transcript == ""

    @pytest.mark.asyncio
    async def test_whitespace_transcript(self):
        result = await classify_transcript("   \n\t  ")

        assert isinstance(result, ClassificationResult)
        assert result.is_spam is False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"
        assert result.evidence_lines == []
        assert result.full_transcript == "   \n\t  "

    @pytest.mark.asyncio
    async def test_spam_classification_with_openai_classifier(self):
        transcript = "Caller: Hello, I'm calling from Bank of America about your credit card offer..."
        mock_result = ClassificationResult(
            is_spam=True,
            confidence=0.95,
            reason="Unsolicited financial product offer",
            evidence_lines=["Caller: Hello, I'm calling from Bank of America about your credit card offer..."],
            full_transcript=transcript,
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}), patch(
            "src.spam_classifier.OpenAIClassifier"
        ) as classifier_class:
            classifier = MagicMock()
            classifier.model = "gpt-4o-mini"
            classifier.classify = AsyncMock(return_value=mock_result)
            classifier.close = AsyncMock()
            classifier_class.return_value = classifier

            result = await classify_transcript(transcript)

        assert result.is_spam is True
        assert result.confidence == 0.95
        assert result.reason == "Unsolicited financial product offer"
        assert len(result.evidence_lines) == 1
        assert "Bank of America" in result.evidence_lines[0]
        assert result.full_transcript == transcript

    @pytest.mark.asyncio
    async def test_classification_error_handling(self):
        transcript = "Some test transcript"

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}), patch(
            "src.spam_classifier.OpenAIClassifier"
        ) as classifier_class:
            classifier = MagicMock()
            classifier.model = "gpt-4o-mini"
            classifier.classify = AsyncMock(side_effect=Exception("API Error"))
            classifier.close = AsyncMock()
            classifier_class.return_value = classifier

            result = await classify_transcript(transcript)

        assert result.is_spam is False
        assert result.confidence == 0.0
        assert "Classification error:" in result.reason
        assert result.evidence_lines == []
        assert result.full_transcript == transcript
