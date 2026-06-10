"""Integration tests for the full spam detection pipeline."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.spam_classifier import classify_transcript
from src.config import AppConfig
from src.classification_result import ClassificationResult


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

        with _mock_openai_classifier(mock_result):
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

        mock_result = ClassificationResult(
            is_spam=False,
            confidence=0.92,
            reason="Legitimate community member returning lost pet",
            evidence_lines=[],
            full_transcript=legitimate_transcript
        )

        with _mock_openai_classifier(mock_result):
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
    async def test_openai_error_handling(self):
        """Test pipeline handles OpenAI errors gracefully."""
        transcript = "Some test transcript"

        with _mock_openai_classifier(Exception("OpenAI error")):
            result = await classify_transcript(transcript)

        assert result.is_spam == False
        assert result.confidence == 0.0
        assert "Classification error" in result.reason

    @pytest.mark.asyncio
    async def test_openai_classifier_cleanup_on_error(self):
        """Test that OpenAI classifier is cleaned up even on error."""
        transcript = "Some test transcript"

        with _mock_openai_classifier(Exception("OpenAI error")) as classifier:
            await classify_transcript(transcript)

        classifier.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_classifier_cleanup_on_success(self):
        """Test that OpenAI classifier is cleaned up on success."""
        transcript = "Some test transcript"
        mock_result = ClassificationResult(False, 0.9, "Test", [], transcript)

        with _mock_openai_classifier(mock_result) as classifier:
            await classify_transcript(transcript)

        classifier.close.assert_called_once()


class TestConfigIntegration:
    """Integration tests for configuration management."""

    def test_default_config(self):
        """Test default configuration loading."""
        config = AppConfig.load()

        assert config.spam_detection.call_duration_seconds == 38.0
        assert config.spam_detection.max_transcript_length == 100_000

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


class _mock_openai_classifier:
    def __init__(self, classify_result):
        self.classify_result = classify_result
        self.classifier = MagicMock()
        self.classifier.model = "gpt-4o-mini"
        if isinstance(classify_result, Exception):
            self.classifier.classify = AsyncMock(side_effect=classify_result)
        else:
            self.classifier.classify = AsyncMock(return_value=classify_result)
        self.classifier.close = AsyncMock()

    def __enter__(self):
        self.env_patch = patch.dict(
            "os.environ",
            {
                "SPAM_CLASSIFIER_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test-key",
            },
        )
        self.class_patch = patch("src.spam_classifier.OpenAIClassifier", return_value=self.classifier)
        self.env_patch.__enter__()
        self.class_patch.__enter__()
        return self.classifier

    def __exit__(self, exc_type, exc, traceback):
        self.class_patch.__exit__(exc_type, exc, traceback)
        self.env_patch.__exit__(exc_type, exc, traceback)
