"""Utilities for preparing TRAI UCC complaint drafts."""

from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from src.classification_result import ClassificationResult

TRAI_SMS_SHORT_CODE = "1909"
UNKNOWN_SENDER = "UNKNOWN"
INDIA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
MAX_COMPLAINT_DESCRIPTION_CHARS = 80


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
    text = " ".join(
        part
        for part in [
            result.reason,
            " ".join(result.evidence_lines),
            result.full_transcript,
        ]
        if part
    ).lower()

    compact_description = _keyword_description(text)
    if compact_description:
        return compact_description

    reason = " ".join(result.reason.split())
    if not reason:
        return "Unsolicited commercial communication"

    if len(reason) <= MAX_COMPLAINT_DESCRIPTION_CHARS:
        return reason

    return f"{reason[: MAX_COMPLAINT_DESCRIPTION_CHARS - 3].rstrip()}..."


def _keyword_description(text: str) -> str | None:
    if "otp" in text and "credit card" in text:
        return "OTP requested for credit card"
    if "otp" in text:
        return "OTP requested"
    if "cvv" in text or "pin" in text:
        return "Card/banking secret requested"
    if "credit card" in text:
        return "Unsolicited credit card call"
    if "loan" in text:
        return "Unsolicited loan call"
    if "insurance" in text or "policy" in text:
        return "Unsolicited insurance call"
    if "kyc" in text:
        return "KYC verification call"
    if "upi" in text or "payment" in text:
        return "Payment/UPI request call"
    if "real estate" in text or "property" in text:
        return "Unsolicited real estate call"
    return None
