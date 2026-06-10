import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os
import logging

import httpx

# Add the project root to sys.path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.telegram_notifier import (
    send_spam_alert,
    _build_message,
    _build_message_html,
    _build_reply_markup,
    _html_escape,
)
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

    def test_build_message_html_truncates_and_escapes(self):
        """Test long messages are bounded and HTML-sensitive fields are escaped."""
        result = ClassificationResult(
            is_spam=False,
            confidence=0.0,
            reason="<quota> " + ("provider error " * 200),
            evidence_lines=["<evidence> " + ("x" * 500)],
            full_transcript="<transcript> " + ("hello " * 1000),
        )

        message = _build_message_html(result)

        assert len(message) <= 3900
        assert "&lt;quota&gt;" in message
        assert "<quota>" not in message
        assert "&lt;transcript&gt;" in message
    
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
        assert "<b>TRAI complaint draft:</b>" in html_message
        assert "To: <code>1909</code>" in html_message

    def test_build_reply_markup_without_report_url(self):
        """Test that no Telegram button is sent without a report URL."""
        result = self._create_test_result(is_spam=True, confidence=0.8)

        with patch.dict(os.environ, {"TRAI_REPORT_CONFIRM_URL": ""}):
            assert _build_reply_markup(result) is None

    def test_build_reply_markup_with_report_url(self):
        """Test Telegram report button payload."""
        result = self._create_test_result(is_spam=True, confidence=0.8)

        with patch.dict(os.environ, {"TRAI_REPORT_CONFIRM_URL": "https://example.com/report"}):
            markup = _build_reply_markup(result, caller_number="+91 98765 43210")

        assert markup is not None
        button = markup["inline_keyboard"][0][0]
        assert button["text"] == "Report to TRAI"
        assert button["url"].startswith("https://example.com/report?to=1909&body=")
        assert "%2B919876543210" in button["url"]
    
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
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        with patch('src.telegram_notifier.httpx.AsyncClient', return_value=mock_client):
            with patch.dict(os.environ, {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_CHAT_ID": "123456789",
                "TRAI_REPORT_CONFIRM_URL": "https://example.com/report",
            }):
                sent = await send_spam_alert(result)
                assert sent == True
                payload = mock_client.__aenter__.return_value.post.call_args.kwargs["json"]
                assert "reply_markup" in payload
    
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

    @pytest.mark.asyncio
    async def test_send_spam_alert_redacts_token_on_http_status_error(self, caplog):
        """Test that Telegram HTTP errors do not leak the bot token."""
        result = self._create_test_result()
        token = "test-secret-token"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        request = httpx.Request("POST", url)
        response = httpx.Response(401, request=request)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Client error '401 Unauthorized'",
            request=request,
            response=response,
        )

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        with patch('src.telegram_notifier.httpx.AsyncClient', return_value=mock_client):
            with patch.dict(os.environ, {
                "TELEGRAM_BOT_TOKEN": token,
                "TELEGRAM_CHAT_ID": "123456789"
            }):
                with caplog.at_level(logging.ERROR, logger="telegram-notifier"):
                    sent = await send_spam_alert(result)

        assert sent == False
        assert token not in caplog.text
        assert "api.telegram.org" not in caplog.text
        assert "HTTP 401" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
