"""Utilities for preparing TRAI UCC complaint drafts."""

from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from src.classification_result import ClassificationResult

TRAI_SMS_SHORT_CODE = "1909"
UNKNOWN_SENDER = "UNKNOWN"
INDIA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


def normalize_sender(sender: str | None) -> str:
    """Return a TRAI-friendly sender phone number/header."""
    if not sender:
        return UNKNOWN_SENDER

    value = sender.strip()
    if not value:
        return UNKNOWN_SENDER

    if value.startswith("+"):
        digits = "".join(ch for ch in value[1:] if ch.isdigit())
        return f"+{digits}" if digits else UNKNOWN_SENDER

    alnum = "".join(ch for ch in value if ch.isalnum())
    return alnum or UNKNOWN_SENDER


def build_trai_complaint_text(
    result: ClassificationResult,
    *,
    sender: str | None = None,
    received_at: datetime | None = None,
) -> str:
    """Build a user-editable SMS complaint draft for TRAI/TSP reporting."""
    timestamp = received_at or datetime.now(INDIA_TIMEZONE)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=INDIA_TIMEZONE)
    complaint_date = timestamp.astimezone(INDIA_TIMEZONE).strftime("%d/%m/%y")
    normalized_sender = normalize_sender(sender)
    description = _build_description(result)
    return f"{description}, {normalized_sender}, {complaint_date}"


def build_sms_uri(
    message: str,
    *,
    destination: str = TRAI_SMS_SHORT_CODE,
) -> str:
    """Build a mobile SMS composer deep link."""
    return f"sms:{quote(destination, safe='')}?body={quote(message, safe='')}"


def build_report_confirmation_url(
    base_url: str,
    message: str,
    *,
    destination: str = TRAI_SMS_SHORT_CODE,
) -> str:
    """Build an HTTPS confirmation URL that can later redirect to an SMS draft."""
    separator = "&" if "?" in base_url else "?"
    return (
        f"{base_url.rstrip('/')}{separator}"
        f"to={quote(destination, safe='')}&body={quote(message, safe='')}"
    )


def _build_description(result: ClassificationResult) -> str:
    reason = " ".join(result.reason.split())
    if not reason:
        return "Unsolicited commercial communication"

    if len(reason) <= 120:
        return reason

    return f"{reason[:117].rstrip()}..."
