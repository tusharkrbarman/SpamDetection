"""Gemini API adapters for development use."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from livekit.agents import llm
from livekit.agents._exceptions import APIConnectionError, APIStatusError
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN

from src.classification_result import ClassificationResult
from src.openai_classifier import CLASSIFICATION_SYSTEM_PROMPT, _parse_bool, _parse_evidence_lines

logger = logging.getLogger("gemini")
logger.setLevel(logging.INFO)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_FALLBACK_RESPONSE = "Okay, go on."
MAX_ERROR_SUMMARY_CHARS = 240


class GeminiLLM(llm.LLM):
    """Small LiveKit LLM adapter that calls Gemini's generateContent API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        max_output_tokens: int = 96,
        temperature: float = 0.4,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "google-gemini"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
        extra_kwargs=NOT_GIVEN,
    ) -> llm.LLMStream:
        if tools:
            raise ValueError("Gemini development adapter does not support tool calls")
        return GeminiLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=[],
            conn_options=conn_options,
        )

    async def generate_text(
        self,
        *,
        contents: list[dict[str, Any]],
        system_messages: list[str] | None = None,
        response_mime_type: str | None = None,
    ) -> tuple[str, llm.CompletionUsage | None]:
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self._max_output_tokens,
                "temperature": self._temperature,
            },
        }
        if response_mime_type:
            body["generationConfig"]["responseMimeType"] = response_mime_type
        if system_messages:
            body["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_messages)}],
            }

        url = f"{GEMINI_API_BASE}/models/{self._model}:generateContent"
        try:
            response = await self._client.post(url, params={"key": self._api_key}, json=body)
        except httpx.TimeoutException as e:
            raise APIConnectionError("Gemini API request timed out") from e
        except httpx.HTTPError as e:
            raise APIConnectionError(f"Gemini API connection error: {e}") from e

        if response.status_code != 200:
            error_summary = _summarize_error_response(response)
            raise APIStatusError(
                message=f"Gemini API error ({response.status_code}): {error_summary}",
                status_code=response.status_code,
                body=error_summary,
            )

        payload = response.json()
        text = _extract_text(payload)
        usage = _extract_usage(payload)
        return text, usage

    async def aclose(self) -> None:
        await self._client.aclose()


class GeminiLLMStream(llm.LLMStream):
    def __init__(
        self,
        gemini_llm: GeminiLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            gemini_llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._gemini_llm = gemini_llm

    async def _run(self) -> None:
        contents, google_data = self._chat_ctx.to_provider_format("google")
        try:
            text, usage = await self._gemini_llm.generate_text(
                contents=contents,
                system_messages=google_data.system_messages,
            )
        except APIStatusError as e:
            if not _is_quota_error(e):
                raise
            logger.warning("Gemini quota exceeded during live response; using fallback reply")
            text = GEMINI_FALLBACK_RESPONSE
            usage = None

        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="gemini",
                delta=llm.ChoiceDelta(role="assistant", content=text),
                usage=usage,
            )
        )


class GeminiClassifier:
    """Classifies transcripts with Gemini's free-tier-friendly text API."""

    def __init__(self, api_key: str, model: str | None = None):
        self.model = model or "gemini-3.1-flash-lite"
        self._llm = GeminiLLM(
            api_key=api_key,
            model=self.model,
            max_output_tokens=512,
            temperature=0.1,
        )

    async def classify(self, transcript: str) -> ClassificationResult:
        contents = [
            {
                "role": "user",
                "parts": [{"text": f"Classify this call transcript:\n\n{transcript}"}],
            }
        ]
        text, _ = await self._llm.generate_text(
            contents=contents,
            system_messages=[CLASSIFICATION_SYSTEM_PROMPT],
            response_mime_type="application/json",
        )
        logger.info("Gemini classification raw response: %s", text)
        result = json.loads(_strip_json_fence(text))
        return ClassificationResult(
            is_spam=_parse_bool(result.get("is_spam", False)),
            confidence=float(result.get("confidence", 0.0)),
            reason=str(result.get("reason", "")),
            evidence_lines=_parse_evidence_lines(result.get("evidence_lines", [])),
            full_transcript=transcript,
        )

    async def close(self) -> None:
        await self._llm.aclose()


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(str(part.get("text", "")) for part in parts).strip()


def _extract_usage(payload: dict[str, Any]) -> llm.CompletionUsage | None:
    metadata = payload.get("usageMetadata") or {}
    if not metadata:
        return None
    prompt_tokens = int(metadata.get("promptTokenCount", 0))
    completion_tokens = int(metadata.get("candidatesTokenCount", 0))
    total_tokens = int(metadata.get("totalTokenCount", prompt_tokens + completion_tokens))
    return llm.CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        return stripped.removeprefix("```json").removesuffix("```").strip()
    if stripped.startswith("```"):
        return stripped.removeprefix("```").removesuffix("```").strip()
    return stripped


def _is_quota_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    text = str(error).lower()
    return status_code == 429 or "quota" in text or "resource_exhausted" in text


def _summarize_error_response(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return _truncate_error(response.text or response.reason_phrase)

    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return _truncate_error(response.text or response.reason_phrase)

    status = str(error.get("status", "")).strip()
    message = str(error.get("message", "")).strip()
    if response.status_code == 429 or status == "RESOURCE_EXHAUSTED":
        retry_delay = _extract_retry_delay(error)
        retry_suffix = f"; retry after {retry_delay}" if retry_delay else ""
        return f"quota exceeded{retry_suffix}"

    first_line = message.splitlines()[0] if message else response.reason_phrase
    return _truncate_error(first_line)


def _extract_retry_delay(error: dict[str, Any]) -> str:
    details = error.get("details")
    if not isinstance(details, list):
        return ""
    for detail in details:
        if not isinstance(detail, dict):
            continue
        retry_delay = detail.get("retryDelay")
        if isinstance(retry_delay, str):
            return retry_delay
    return ""


def _truncate_error(text: str) -> str:
    if len(text) <= MAX_ERROR_SUMMARY_CHARS:
        return text
    return f"{text[: MAX_ERROR_SUMMARY_CHARS - 3].rstrip()}..."
