"""Kilo AI provider implementation for LLM classification."""

import json
import logging
from openai import AsyncOpenAI

from src.providers.base import LLMProvider, ClassificationResult

logger = logging.getLogger("kilo-provider")
logger.setLevel(logging.INFO)

# System prompt for classification (without JSON mode support)
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

IMPORTANT: Return ONLY valid JSON, no markdown formatting.
"""


class KiloProvider(LLMProvider):
    """Kilo AI provider for spam classification."""

    def __init__(self, api_key: str, base_url: str, model: str | None = None):
        """Initialize Kilo AI provider.

        Args:
            api_key: Kilo AI API key (should start with "eyJ")
            base_url: Kilo AI API base URL
            model: Optional model name (defaults to free Nemotron model)
        """
        super().__init__(api_key, base_url)
        self.model = model or "nvidia/nemotron-3-super-120b-a12b:free"
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0
        )

    async def classify(self, transcript: str) -> ClassificationResult:
        """Classify a transcript using Kilo AI.

        Args:
            transcript: The call transcript to classify

        Returns:
            ClassificationResult with classification details
        """
        if not transcript.strip():
            return ClassificationResult(
                is_spam=False,
                confidence=0.0,
                reason="No transcript to analyze",
                evidence_lines=[],
                full_transcript=transcript,
            )

        try:
            logger.info(f"Using Kilo AI with model: {self.model}")

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
            )

            raw = response.choices[0].message.content
            logger.info("Classification raw response: %s", raw)

            result = json.loads(raw)

            return ClassificationResult(
                is_spam=bool(result.get("is_spam", False)),
                confidence=float(result.get("confidence", 0.0)),
                reason=str(result.get("reason", "")),
                evidence_lines=result.get("evidence_lines", []),
                full_transcript=transcript,
            )

        except Exception as e:
            logger.error("Kilo AI classification failed: %s", e, exc_info=True)
            return ClassificationResult(
                is_spam=False,
                confidence=0.0,
                reason=f"Kilo AI classification error: {e}",
                evidence_lines=[],
                full_transcript=transcript,
            )

    async def close(self):
        """Close the Kilo AI client."""
        await self.client.close()

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "kilo"
