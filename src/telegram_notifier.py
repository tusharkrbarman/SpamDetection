import logging
import os
from datetime import datetime, timezone

import httpx

from src.providers.base import ClassificationResult

logger = logging.getLogger("telegram-notifier")
logger.setLevel(logging.INFO)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def _build_message(result: ClassificationResult) -> str:
    if result.is_spam:
        status = "🚨 SPAM CALL DETECTED"
        confidence_bar = "█" * int(result.confidence * 10) + "░" * (10 - int(result.confidence * 10))
    else:
        status = "✅ LEGITIMATE CALL"
        confidence_bar = "░" * int((1 - result.confidence) * 10) + "█" * int(result.confidence * 10)

    lines = [
        f"*{status}*",
        "",
        f"Confidence: `{confidence_bar}` {result.confidence:.0%}",
        f"Reason: {result.reason}",
        "",
    ]

    if result.evidence_lines:
        lines.append("*Evidence from transcript:*")
        for line in result.evidence_lines:
            lines.append(f"_{line}_")
        lines.append("")

    lines.append(f"Full transcript ({len(result.full_transcript)} chars):")
    lines.append(f"```\n{result.full_transcript}\n```")
    lines.append("")
    lines.append(f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_")

    return "\n".join(lines)


async def send_spam_alert(result: ClassificationResult) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.warning("Telegram credentials not configured, skipping notification")
        return False

    message = _build_message(result)
    html_message = _build_message_html(result)
    url = TELEGRAM_API_URL.format(token=token)

    payload = {
        "chat_id": chat_id,
        "text": html_message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        logger.info("Telegram alert sent successfully")
        return True

    except Exception as e:
        logger.error("Failed to send Telegram alert: %s", e, exc_info=True)
        return False


def _build_message_html(result: ClassificationResult) -> str:
    if result.is_spam:
        status = "🚨 <b>SPAM CALL DETECTED</b>"
    else:
        status = "✅ <b>LEGITIMATE CALL</b>"

    confidence_pct = f"{result.confidence:.0%}"

    lines = [
        status,
        "",
        f"Confidence: <code>{confidence_pct}</code>",
        f"Reason: {result.reason}",
        "",
    ]

    if result.evidence_lines:
        lines.append("<b>Evidence from transcript:</b>")
        for line in result.evidence_lines:
            escaped = _html_escape(line)
            lines.append(f"<i>• {escaped}</i>")
        lines.append("")

    lines.append("<b>Full transcript:</b>")
    escaped_transcript = _html_escape(result.full_transcript)
    lines.append(f"<code>{escaped_transcript}</code>")
    lines.append("")
    lines.append(f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>")

    return "\n".join(lines)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
