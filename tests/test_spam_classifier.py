"""Tests for spam classifier with provider abstraction."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json
import os
import sys

# Add the project root to sys.path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spam_classifier import classify_transcript
from src.providers.base import ClassificationResult
from src.config import Provider


class TestSpamClassifier:
    """Test cases for spam_classifier.py with provider abstraction."""

    @pytest.mark.asyncio
    async def test_empty_transcript(self):
        """Test classification with empty transcript."""
        result = await classify_transcript("")

        assert isinstance(result, ClassificationResult)
        assert result.is_spam == False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"
        assert result.evidence_lines == []
        assert result.full_transcript == ""

    @pytest.mark.asyncio
    async def test_whitespace_transcript(self):
        """Test classification with whitespace-only transcript."""
        result = await classify_transcript("   \n\t  ")

        assert isinstance(result, ClassificationResult)
        assert result.is_spam == False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"
        assert result.evidence_lines == []
        assert result.full_transcript == "   \n\t  "

    @pytest.mark.asyncio
    async def test_spam_classification_with_mock_provider(self):
        """Test spam classification with mocked provider."""
        transcript = "Caller: Hello, I'm calling from Bank of America about your credit card offer..."

        # Mock the provider result
        mock_result = ClassificationResult(
            is_spam=True,
            confidence=0.95,
            reason="Unsolicited financial product offer",
            evidence_lines=["Caller: Hello, I'm calling from Bank of America about your credit card offer..."],
            full_transcript=transcript
        )

        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-or-test-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
        }), patch('src.spam_classifier.PROVIDER_MAP') as mock_map:
            mock_provider_class = MagicMock()
            mock_provider = MagicMock()
            mock_provider.classify = AsyncMock(return_value=mock_result)
            mock_provider.get_provider_name = MagicMock(return_value="openrouter")
            mock_provider.get_model_name = MagicMock(return_value="google/gemma-4-31b-it:free")
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider
            mock_map.__getitem__.return_value = mock_provider_class

            result = await classify_transcript(transcript)

        assert isinstance(result, ClassificationResult)
        assert result.is_spam == True
        assert result.confidence == 0.95
        assert result.reason == "Unsolicited financial product offer"
        assert len(result.evidence_lines) == 1
        assert "Bank of America" in result.evidence_lines[0]
        assert result.full_transcript == transcript

    @pytest.mark.asyncio
    async def test_legitimate_classification_with_mock_provider(self):
        """Test legitimate classification with mocked provider."""
        transcript = "Caller: Hi, this is John from Nextdoor. I found your lost dog and wanted to return it."

        # Mock the provider result
        mock_result = ClassificationResult(
            is_spam=False,
            confidence=0.90,
            reason="Legitimate community member returning lost pet",
            evidence_lines=[],
            full_transcript=transcript
        )

        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-or-test-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
        }), patch('src.spam_classifier.PROVIDER_MAP') as mock_map:
            mock_provider_class = MagicMock()
            mock_provider = MagicMock()
            mock_provider.classify = AsyncMock(return_value=mock_result)
            mock_provider.get_provider_name = MagicMock(return_value="openrouter")
            mock_provider.get_model_name = MagicMock(return_value="google/gemma-4-31b-it:free")
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider
            mock_map.__getitem__.return_value = mock_provider_class

            result = await classify_transcript(transcript)

        assert isinstance(result, ClassificationResult)
        assert result.is_spam == False
        assert result.confidence == 0.90
        assert result.reason == "Legitimate community member returning lost pet"
        assert result.evidence_lines == []
        assert result.full_transcript == transcript

    @pytest.mark.asyncio
    async def test_classification_error_handling(self):
        """Test that classification handles provider errors gracefully."""
        transcript = "Some test transcript"

        # Mock provider to raise an exception
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-or-test-key",
            "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
        }), patch('src.spam_classifier.PROVIDER_MAP') as mock_map:
            mock_provider_class = MagicMock()
            mock_provider = MagicMock()
            mock_provider.classify = AsyncMock(side_effect=Exception("API Error"))
            mock_provider.get_provider_name = MagicMock(return_value="openrouter")
            mock_provider.get_model_name = MagicMock(return_value="google/gemma-4-31b-it:free")
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider
            mock_map.__getitem__.return_value = mock_provider_class

            result = await classify_transcript(transcript)

        # Should return a safe fallback result
        assert isinstance(result, ClassificationResult)
        assert result.is_spam == False
        assert result.confidence == 0.0
        assert "Classification error:" in result.reason
        assert result.evidence_lines == []
        assert result.full_transcript == transcript


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
