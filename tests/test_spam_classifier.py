"""Tests for spam classifier."""

import os
import sys
import types
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

        with patch.dict(
            os.environ,
            {
                "SPAM_CLASSIFIER_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test-key",
            },
        ), patch(
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
    async def test_spam_classification_with_gemini_classifier(self):
        transcript = "Caller: You have won a prize and must act now."
        mock_result = ClassificationResult(
            is_spam=True,
            confidence=0.9,
            reason="Prize scam language",
            evidence_lines=["Caller: You have won a prize and must act now."],
            full_transcript=transcript,
        )

        fake_gemini_module = types.SimpleNamespace(GeminiClassifier=MagicMock())

        with patch.dict(
            os.environ,
            {
                "SPAM_CLASSIFIER_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-key",
                "SPAM_CLASSIFICATION_MODEL": "gemini-2.5-flash-lite",
            },
        ), patch.dict(sys.modules, {"src.gemini_llm": fake_gemini_module}):
            classifier = MagicMock()
            classifier.model = "gemini-2.5-flash-lite"
            classifier.classify = AsyncMock(return_value=mock_result)
            classifier.close = AsyncMock()
            fake_gemini_module.GeminiClassifier.return_value = classifier

            result = await classify_transcript(transcript)

        assert result.is_spam is True
        assert result.confidence == 0.9
        assert result.reason == "Prize scam language"
        assert result.evidence_lines == ["Caller: You have won a prize and must act now."]
        assert result.full_transcript == transcript

    @pytest.mark.asyncio
    async def test_classification_error_handling(self):
        transcript = "Some test transcript"

        with patch.dict(
            os.environ,
            {
                "SPAM_CLASSIFIER_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test-key",
            },
        ), patch(
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

    @pytest.mark.asyncio
    async def test_quota_error_handling_is_concise(self):
        transcript = "Some test transcript"

        class QuotaError(Exception):
            status_code = 429

        with patch.dict(
            os.environ,
            {
                "SPAM_CLASSIFIER_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test-key",
            },
        ), patch(
            "src.spam_classifier.OpenAIClassifier"
        ) as classifier_class:
            classifier = MagicMock()
            classifier.model = "gpt-4o-mini"
            classifier.classify = AsyncMock(
                side_effect=QuotaError("very long quota response " * 200)
            )
            classifier.close = AsyncMock()
            classifier_class.return_value = classifier

            result = await classify_transcript(transcript)

        assert result.is_spam is False
        assert result.confidence == 0.0
        assert result.reason == "Classification error: LLM quota exceeded; try again later"
        assert len(result.reason) < 100
        assert result.full_transcript == transcript
