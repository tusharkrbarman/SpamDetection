import json
import logging
import os
from dataclasses import dataclass
from openai import AsyncOpenAI

logger = logging.getLogger("spam-classifier")
logger.setLevel(logging.INFO)

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a spam call classifier. Analyze the transcript and determine if the call is spam or not.

Return ONLY a valid JSON object with this exact structure:
{
    "is_spam": true or false,
    "confidence": a float between 0.0 and 1.0,
    "reason": "A brief one-sentence explanation of why this is or isn't spam,
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


@dataclass(frozen=True)
class ClassificationResult:
    is_spam: bool
    confidence: float
    reason: str
    evidence_lines: list[str]
    full_transcript: str


def _get_llm_client():
    """Get the appropriate LLM client based on available API keys."""
    ollama_key = os.environ.get("OLLAMA_API_KEY")
    ollama_base = os.environ.get("OLLAMA_API_BASE")
    
    kilo_key = os.environ.get("KILO_API_KEY")
    kilo_base = os.environ.get("KILO_API_BASE")
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_base = os.environ.get("OPENAI_API_BASE")
    
    # Try OpenRouter first if available (free models)
    if openai_key and openai_key.startswith("sk-or-") and openai_base:
        try:
            return AsyncOpenAI(
                base_url=openai_base,
                api_key=openai_key,
                max_retries=0
            ), "openrouter"
        except Exception as e:
            logger.warning(f"Failed to create OpenRouter client: {e}")
    
    # Fall back to Kilo AI if available (free models)
    if kilo_key and kilo_key.startswith("eyJ") and kilo_base:
        try:
            return AsyncOpenAI(
                base_url=kilo_base,
                api_key=kilo_key,
                max_retries=0
            ), "kilo"
        except Exception as e:
            logger.warning(f"Failed to create Kilo AI client: {e}")
    
    # Fall back to Ollama if available (free model)
    if ollama_key and ollama_base:
        try:
            return AsyncOpenAI(
                base_url=f"{ollama_base}/v1",
                api_key=ollama_key,
                max_retries=0
            ), "ollama"
        except Exception as e:
            logger.warning(f"Failed to create Ollama client: {e}")
    
    # Default to direct OpenAI if configured (paid, but available as fallback)
    if openai_key and not openai_key.startswith("sk-or-"):
        try:
            return AsyncOpenAI(api_key=openai_key), "openai"
        except Exception as e:
            logger.warning(f"Failed to create OpenAI client: {e}")
    
    raise ValueError("No LLM API credentials configured")


def _get_classification_model(provider: str) -> str:
    """Get the appropriate model name for the provider (using free models only)."""
    model_env = os.environ.get("SPAM_CLASSIFICATION_MODEL")
    
    if provider == "ollama":
        # Ollama cloud free model
        return model_env or "llama3.2:latest"
    elif provider == "kilo":
        # Kilo AI free models - use Nemotron 3 Super (NVIDIA)
        # From Kilo docs: nvidia/nemotron-3-super-120b-a12b:free
        return model_env or "nvidia/nemotron-3-super-120b-a12b:free"
    elif provider == "openrouter":
        # OpenRouter free models - Gemma 4 31B is the latest free model
        return model_env or "google/gemma-4-31b-it:free"
    else:
        return model_env or "google/gemma-4-31b-it:free"


async def classify_transcript(transcript: str) -> ClassificationResult:
    if not transcript.strip():
        return ClassificationResult(
            is_spam=False,
            confidence=0.0,
            reason="No transcript to analyze",
            evidence_lines=[],
            full_transcript=transcript,
        )

    try:
        client, provider = _get_llm_client()
        model = _get_classification_model(provider)
        
        logger.info(f"Using {provider} with model: {model}")
        
        # For Ollama and Kilo, we don't use response_format since they may not support JSON mode
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Classify this call transcript:\n\n{transcript}",
                },
            ],
            "temperature": 0.1,
        }
        
        if provider in ("ollama", "kilo"):
            # These providers don't support response_format, so we ask in the prompt
            kwargs["messages"][0]["content"] = CLASSIFICATION_SYSTEM_PROMPT + "\n\nIMPORTANT: Return ONLY valid JSON, no markdown formatting."
        else:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await client.chat.completions.create(**kwargs)

        raw = response.choices[0].message.content
        logger.info("Classification raw response: %s", raw)

        result = json.loads(raw)

        await client.close()

        return ClassificationResult(
            is_spam=bool(result.get("is_spam", False)),
            confidence=float(result.get("confidence", 0.0)),
            reason=str(result.get("reason", "")),
            evidence_lines=result.get("evidence_lines", []),
            full_transcript=transcript,
        )

    except Exception as e:
        logger.error("Classification failed: %s", e, exc_info=True)
        return ClassificationResult(
            is_spam=False,
            confidence=0.0,
            reason=f"Classification error: {e}",
            evidence_lines=[],
            full_transcript=transcript,
        )