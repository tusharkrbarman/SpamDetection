import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add the project root to sys.path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.telegram_notifier import send_spam_alert, _build_message, _build_message_html, _html_escape
from src.spam_classifier import ClassificationResult


class TestTelegramNotifier:
    """Test cases for telegram_notifier.py"""
    
    def test_html_escape(self):
        """Test HTML escaping function"""
        assert _html_escape("Hello & World") == "Hello &amp; World"
        assert _html_escape("<script>") == "&lt;script&gt;"
        assert _html_escape(">") == "&gt;"
        assert _html_escape("Plain text") == "Plain text"
        assert _html_escape("Mix & < >") == "Mix &amp; &lt; &gt;"
    
    def _create_test_result(self, is_spam=True, confidence=0.85):
        """Helper to create a test ClassificationResult"""
        return ClassificationResult(
            is_spam=is_spam,
            confidence=confidence,
            reason="Test reason",
            evidence_lines=["Evidence line 1", "Evidence line 2"],
            full_transcript="Test transcript\nLine 2\nLine 3"
        )
    
    def test_build_message_spam(self):
        """Test message building for spam call"""
        result = self._create_test_result(is_spam=True, confidence=0.9)
        message = _build_message(result)
        
        assert "🚨 SPAM CALL DETECTED" in message
        assert "Confidence:" in message
        assert "90%" in message  # 0.9 * 100
        assert "Test reason" in message
        assert "Evidence from transcript:" in message
        assert "_Evidence line 1_" in message
        assert "_Evidence line 2_" in message
        assert "Test transcript" in message
        assert "Line 2" in message
        assert "Line 3" in message
    
    def test_build_message_legitimate(self):
        """Test message building for legitimate call"""
        result = self._create_test_result(is_spam=False, confidence=0.75)
        message = _build_message(result)
        
        assert "✅ LEGITIMATE CALL" in message
        assert "Confidence:" in message
        assert "75%" in message  # 0.75 * 100
        assert "Test reason" in message
        # Since our test result has evidence_lines, the evidence section should be present
        assert "Evidence from transcript:" in message
        assert "_Evidence line 1_" in message
        assert "_Evidence line 2_" in message
    
    def test_build_message_no_evidence(self):
        """Test message building when there's no evidence"""
        result = ClassificationResult(
            is_spam=False,
            confidence=0.6,
            reason="No spam indicators",
            evidence_lines=[],  # Empty evidence
            full_transcript="Short transcript"
        )
        message = _build_message(result)
        
        # When evidence_lines is empty, "Evidence from transcript:" section should NOT be present
        assert "Evidence from transcript:" not in message
        # Full transcript section should always be present (it's added regardless of evidence)
        assert "Full transcript (16 chars):" in message
        assert "Short transcript" in message
        assert "Short transcript" in message
    
    def test_build_message_html(self):
        """Test HTML message building"""
        result = self._create_test_result(is_spam=True, confidence=0.8)
        html_message = _build_message_html(result)
        
        assert "🚨 <b>SPAM CALL DETECTED</b>" in html_message
        assert "Confidence:" in html_message
        assert "<code>80%</code>" in html_message
        assert "Test reason" in html_message
        assert "<b>Evidence from transcript:</b>" in html_message
        assert "<i>• Evidence line 1</i>" in html_message
        assert "<i>• Evidence line 2</i>" in html_message
        assert "<code>Test transcript" in html_message
    
    @pytest.mark.asyncio
    async def test_send_spam_alert_missing_credentials(self):
        """Test sending alert when Telegram credentials are missing"""
        result = self._create_test_result()
        
        # Test with missing bot token
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "12345"}):
            result = await send_spam_alert(result)
            assert result == False
        
        # Test with missing chat ID
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "token", "TELEGRAM_CHAT_ID": ""}):
            result = await send_spam_alert(result)
            assert result == False
        
        # Test with both missing
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
            result = await send_spam_alert(result)
            assert result == False
    
    @pytest.mark.asyncio
    async def test_send_spam_alert_success(self):
        """Test successful sending of spam alert"""
        result = self._create_test_result(is_spam=True, confidence=0.9)
        
        # Mock httpx.AsyncClient
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        with patch('src.telegram_notifier.httpx.AsyncClient', return_value=mock_client):
            with patch.dict(os.environ, {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_CHAT_ID": "123456789"
            }):
                sent = await send_spam_alert(result)
                assert sent == True
    
    @pytest.mark.asyncio
    async def test_send_spam_alert_http_error(self):
        """Test handling of HTTP errors when sending alert"""
        result = self._create_test_result()
        
        # Mock httpx to raise an exception
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(
            side_effect=Exception("Network error")
        )
        
        with patch('src.telegram_notifier.httpx.AsyncClient', return_value=mock_client):
            with patch.dict(os.environ, {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_CHAT_ID": "123456789"
            }):
                sent = await send_spam_alert(result)
                assert sent == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])