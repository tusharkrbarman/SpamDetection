"""OpenAI implementation for transcript spam classification."""

import json
import logging

from openai import AsyncOpenAI

from src.classification_result import ClassificationResult

logger = logging.getLogger("openai-classifier")
logger.setLevel(logging.INFO)

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a spam call classifier. Analyze the transcript and determine if the call is spam or not.

Return ONLY a valid JSON object with this exact structure:
{
    "is_spam": true or false,
    "confidence": a float between 0.0 and 1.0,
    "reason": "A brief one-sentence explanation of why this is or isn't spam",
    "evidence_lines": ["exact line(s) from the transcript that indicate spam, or empty array if not spam"]
}

Spam indicators:
- Trying to sell products, services, insurance, loans, investments
- Claiming to be from a bank, government, or tech support unsolicited
- Asking for personal/financial information
- Urgent threats about accounts, penalties, or legal action
- Prize/lottery winnings that require action

If there is insufficient transcript to classify, set is_spam to false with low confidence.
"""


class OpenAIClassifier:
    """Classifies transcripts using OpenAI."""

    def __init__(self, api_key: str, model: str | None = None):
        self.model = model or "gpt-4o-mini"
        self.client = AsyncOpenAI(api_key=api_key)

    async def classify(self, transcript: str) -> ClassificationResult:
        if not transcript.strip():
            return ClassificationResult(
                is_spam=False,
                confidence=0.0,
                reason="No transcript to analyze",
                evidence_lines=[],
                full_transcript=transcript,
            )

        try:
            logger.info("Using OpenAI with model: %s", self.model)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Classify this call transcript:\n\n{transcript}",
                    },
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            logger.info("Classification raw response: %s", raw)
            result = json.loads(raw)

            return ClassificationResult(
                is_spam=_parse_bool(result.get("is_spam", False)),
                confidence=float(result.get("confidence", 0.0)),
                reason=str(result.get("reason", "")),
                evidence_lines=_parse_evidence_lines(result.get("evidence_lines", [])),
                full_transcript=transcript,
            )

        except Exception as e:
            logger.error("OpenAI classification failed: %s", e, exc_info=True)
            return ClassificationResult(
                is_spam=False,
                confidence=0.0,
                reason=f"OpenAI classification error: {e}",
                evidence_lines=[],
                full_transcript=transcript,
            )

    async def close(self) -> None:
        await self.client.close()


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _parse_evidence_lines(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
