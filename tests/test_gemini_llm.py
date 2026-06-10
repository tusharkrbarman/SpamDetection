"""Tests for Gemini development adapter helpers."""

import sys

import httpx

for module_name in list(sys.modules):
    if module_name == "livekit.agents" or module_name.startswith("livekit.agents."):
        module = sys.modules[module_name]
        if getattr(module, "__file__", None) is None:
            sys.modules.pop(module_name, None)

from src.gemini_llm import _summarize_error_response


def test_summarize_quota_error_response_is_concise():
    response = httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "You exceeded your current quota.\n" + ("extra " * 200),
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"retryDelay": "39s"}],
            }
        },
    )

    assert _summarize_error_response(response) == "quota exceeded; retry after 39s"
