from datetime import datetime, timezone

from src.classification_result import ClassificationResult
from src.trai_report import (
    TRAI_SMS_SHORT_CODE,
    build_report_confirmation_url,
    build_sms_uri,
    build_trai_complaint_text,
    normalize_sender,
)


def _result(reason: str = "Unsolicited credit card offer") -> ClassificationResult:
    return ClassificationResult(
        is_spam=True,
        confidence=0.92,
        reason=reason,
        evidence_lines=["Caller: We have a credit card offer"],
        full_transcript="Caller: We have a credit card offer",
    )


def test_normalize_sender():
    assert normalize_sender("+91 98765-43210") == "+919876543210"
    assert normalize_sender("AX-BANK") == "AXBANK"
    assert normalize_sender("") == "UNKNOWN"
    assert normalize_sender(None) == "UNKNOWN"


def test_build_trai_complaint_text():
    received_at = datetime(2026, 6, 9, tzinfo=timezone.utc)
    text = build_trai_complaint_text(
        _result(),
        sender="+91 98765-43210",
        received_at=received_at,
    )

    assert text == "Unsolicited credit card offer, +919876543210, 09/06/26"


def test_build_trai_complaint_text_truncates_long_reason():
    text = build_trai_complaint_text(_result("x" * 200), received_at=datetime(2026, 6, 9))

    description = text.split(",", maxsplit=1)[0]
    assert len(description) == 120
    assert description.endswith("...")


def test_build_sms_uri():
    uri = build_sms_uri("Loan offer, +919876543210, 09/06/26")

    assert uri.startswith(f"sms:{TRAI_SMS_SHORT_CODE}?body=")
    assert "Loan%20offer" in uri


def test_build_report_confirmation_url():
    url = build_report_confirmation_url(
        "https://example.com/report",
        "Loan offer, +919876543210, 09/06/26",
    )

    assert url.startswith("https://example.com/report?to=1909&body=")
    assert "Loan%20offer" in url
