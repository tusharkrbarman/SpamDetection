import logging
import os
from datetime import datetime, timezone

import httpx

from src.classification_result import ClassificationResult
from src.trai_report import (
    TRAI_SMS_SHORT_CODE,
    build_report_confirmation_url,
    build_trai_complaint_text,
)

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


async def send_spam_alert(
    result: ClassificationResult,
    *,
    caller_number: str | None = None,
    received_at: datetime | None = None,
) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.warning("Telegram credentials not configured, skipping notification")
        return False

    html_message = _build_message_html(result, caller_number=caller_number, received_at=received_at)
    url = TELEGRAM_API_URL.format(token=token)

    payload = {
        "chat_id": chat_id,
        "text": html_message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    reply_markup = _build_reply_markup(result, caller_number=caller_number, received_at=received_at)
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        logger.info("Telegram alert sent successfully")
        return True

    except Exception as e:
        logger.error("Failed to send Telegram alert: %s", e, exc_info=True)
        return False


def _build_message_html(
    result: ClassificationResult,
    *,
    caller_number: str | None = None,
    received_at: datetime | None = None,
) -> str:
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

    if result.is_spam:
        complaint_text = build_trai_complaint_text(
            result,
            sender=caller_number,
            received_at=received_at,
        )
        report_url = _build_report_url(complaint_text)
        lines.extend(
            [
                "<b>TRAI complaint draft:</b>",
                f"To: <code>{TRAI_SMS_SHORT_CODE}</code>",
                f"<code>{_html_escape(complaint_text)}</code>",
                "",
                (
                    "Tap the report button to confirm, then review and send the SMS yourself."
                    if report_url
                    else "Copy this draft into an SMS to 1909, review it, and send it yourself."
                ),
                "",
            ]
        )

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


def _build_reply_markup(
    result: ClassificationResult,
    *,
    caller_number: str | None = None,
    received_at: datetime | None = None,
) -> dict | None:
    if not result.is_spam:
        return None

    complaint_text = build_trai_complaint_text(
        result,
        sender=caller_number,
        received_at=received_at,
    )
    report_url = _build_report_url(complaint_text)
    if not report_url:
        return None

    return {
        "inline_keyboard": [
            [
                {
                    "text": "Report to TRAI",
                    "url": report_url,
                }
            ]
        ]
    }


def _build_report_url(complaint_text: str) -> str | None:
    base_url = os.environ.get("TRAI_REPORT_CONFIRM_URL", "").strip()
    if not base_url:
        return None
    if not base_url.startswith(("https://", "http://")):
        logger.warning("TRAI_REPORT_CONFIRM_URL must be http(s), got: %s", base_url)
        return None
    return build_report_confirmation_url(base_url, complaint_text)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
