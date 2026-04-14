"""Integration tests for the full spam detection pipeline."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json

from src.spam_classifier import classify_transcript
from src.config import AppConfig, Provider
from src.providers.base import ClassificationResult


class TestFullPipeline:
    """Integration tests for the complete spam detection pipeline."""

    @pytest.mark.asyncio
    async def test_spam_call_pipeline(self):
        """Test end-to-end spam detection for a spam call."""
        spam_transcript = """Caller: Hello, I'm calling from Bank of America about your credit card offer.
Agent: I see, can you tell me more?
Caller: We have a special limited-time offer with 0% APR for 12 months.
Agent: That sounds interesting.
Caller: You need to act now before this offer expires."""

        # Mock the provider response - should return ClassificationResult
        mock_result = ClassificationResult(
            is_spam=True,
            confidence=0.95,
            reason="Unsolicited financial product offer with urgency",
            evidence_lines=[
                "Caller: Hello, I'm calling from Bank of America about your credit card offer.",
                "Caller: We have a special limited-time offer with 0% APR for 12 months.",
                "Caller: You need to act now before this offer expires."
            ],
            full_transcript=spam_transcript
        )

        with patch('src.spam_classifier.PROVIDER_MAP') as mock_map:
            mock_provider_class = MagicMock()
            mock_provider = MagicMock()
            mock_provider.classify = AsyncMock(return_value=mock_result)
            mock_provider.get_provider_name = MagicMock(return_value="openrouter")
            mock_provider.get_model_name = MagicMock(return_value="google/gemma-4-31b-it:free")
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider
            mock_map.__getitem__.return_value = mock_provider_class

            result = await classify_transcript(spam_transcript)

        assert result.is_spam == True
        assert result.confidence == 0.95
        assert "financial product" in result.reason.lower()
        assert len(result.evidence_lines) == 3

    @pytest.mark.asyncio
    async def test_legitimate_call_pipeline(self):
        """Test end-to-end spam detection for a legitimate call."""
        legitimate_transcript = """Caller: Hi, this is John from Nextdoor. I found your lost dog and wanted to return it.
Agent: Oh my goodness! That's wonderful news.
Caller: Yes, I saw him wandering near the park and checked his collar.
Agent: Thank you so much! Where can I pick him up?
Caller: I'm at 123 Main Street, come by anytime."""

        # Mock the provider response - should return ClassificationResult
        mock_result = ClassificationResult(
            is_spam=False,
            confidence=0.92,
            reason="Legitimate community member returning lost pet",
            evidence_lines=[],
            full_transcript=legitimate_transcript
        )

        with patch('src.spam_classifier.PROVIDER_MAP') as mock_map:
            mock_provider_class = MagicMock()
            mock_provider = MagicMock()
            mock_provider.classify = AsyncMock(return_value=mock_result)
            mock_provider.get_provider_name = MagicMock(return_value="openrouter")
            mock_provider.get_model_name = MagicMock(return_value="google/gemma-4-31b-it:free")
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider
            mock_map.__getitem__.return_value = mock_provider_class

            result = await classify_transcript(legitimate_transcript)

        assert result.is_spam == False
        assert result.confidence == 0.92
        assert "legitimate" in result.reason.lower()
        assert result.evidence_lines == []

    @pytest.mark.asyncio
    async def test_empty_transcript_pipeline(self):
        """Test pipeline with empty transcript."""
        result = await classify_transcript("")

        assert result.is_spam == False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"

    @pytest.mark.asyncio
    async def test_whitespace_transcript_pipeline(self):
        """Test pipeline with whitespace-only transcript."""
        result = await classify_transcript("   \n\t  ")

        assert result.is_spam == False
        assert result.confidence == 0.0
        assert result.reason == "No transcript to analyze"

    @pytest.mark.asyncio
    async def test_long_transcript_validation(self):
        """Test pipeline with transcript exceeding maximum length."""
        # Create a very long transcript
        long_transcript = "Caller: " + "This is a very long call. " * 10000

        result = await classify_transcript(long_transcript)

        assert result.is_spam == False
        assert result.confidence == 0.0
        assert "too long" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_provider_error_handling(self):
        """Test pipeline handles provider errors gracefully."""
        transcript = "Some test transcript"

        with patch('src.spam_classifier.PROVIDER_MAP') as mock_map:
            mock_provider_class = MagicMock()
            mock_provider = MagicMock()
            mock_provider.classify = AsyncMock(side_effect=Exception("Provider error"))
            mock_provider.get_provider_name = MagicMock(return_value="openrouter")
            mock_provider.get_model_name = MagicMock(return_value="google/gemma-4-31b-it:free")
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider
            mock_map.__getitem__.return_value = mock_provider_class

            result = await classify_transcript(transcript)

        assert result.is_spam == False
        assert result.confidence == 0.0
        assert "Classification error" in result.reason

    @pytest.mark.asyncio
    async def test_provider_cleanup_on_error(self):
        """Test that provider is cleaned up even on error."""
        transcript = "Some test transcript"

        with patch('src.spam_classifier.PROVIDER_MAP') as mock_map:
            mock_provider_class = MagicMock()
            mock_provider = MagicMock()
            mock_provider.classify = AsyncMock(side_effect=Exception("Provider error"))
            mock_provider.get_provider_name = MagicMock(return_value="openrouter")
            mock_provider.get_model_name = MagicMock(return_value="google/gemma-4-31b-it:free")
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider
            mock_map.__getitem__.return_value = mock_provider_class

            await classify_transcript(transcript)

        # Verify close was called
        mock_provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_provider_cleanup_on_success(self):
        """Test that provider is cleaned up on success."""
        transcript = "Some test transcript"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "is_spam": False,
            "confidence": 0.90,
            "reason": "Test",
            "evidence_lines": []
        })

        with patch('src.spam_classifier.PROVIDER_MAP') as mock_map:
            mock_provider_class = MagicMock()
            mock_provider = MagicMock()
            mock_provider.classify = AsyncMock(return_value=mock_response.choices[0].message.content)
            mock_provider.get_provider_name = MagicMock(return_value="openrouter")
            mock_provider.get_model_name = MagicMock(return_value="google/gemma-4-31b-it:free")
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider
            mock_map.__getitem__.return_value = mock_provider_class

            await classify_transcript(transcript)

        # Verify close was called
        mock_provider.close.assert_called_once()


class TestConfigIntegration:
    """Integration tests for configuration management."""

    def test_default_config(self):
        """Test default configuration loading."""
        config = AppConfig.load()

        assert config.spam_detection.call_duration_seconds == 12.0
        assert len(config.spam_detection.provider_priority) == 4
        assert Provider.OPENROUTER in config.spam_detection.provider_priority

    def test_provider_priority_order(self):
        """Test provider priority is in correct order."""
        config = AppConfig.load()

        expected_order = [Provider.OPENROUTER, Provider.KILO, Provider.OLLAMA, Provider.OPENAI]
        assert config.spam_detection.provider_priority == expected_order

    def test_telegram_config_disabled(self):
        """Test Telegram config is disabled without credentials."""
        with patch.dict('os.environ', {}, clear=True):
            config = AppConfig.load()
            assert config.telegram.enabled == False

    def test_telegram_config_enabled(self):
        """Test Telegram config is enabled with credentials."""
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'test-token',
            'TELEGRAM_CHAT_ID': '123456789'
        }):
            config = AppConfig.load()
            assert config.telegram.enabled == True
            assert config.telegram.bot_token == 'test-token'
            assert config.telegram.chat_id == '123456789'
