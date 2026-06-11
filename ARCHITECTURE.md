# Architecture Documentation

## Overview

The Spam Call Detection System is a LiveKit-based voice agent that answers calls, stalls callers, classifies transcripts as spam or legitimate with the configured text-to-text provider, and sends Telegram alerts with evidence and an optional TRAI reporting handoff. Development config uses Gemini API free tier.

## System Flow

```text
Incoming call
  -> LiveKit room
  -> VoiceAgent
     -> STT: Sarvam
     -> Conversation LLM: Gemini or OpenAI
     -> TTS: Sarvam
     -> 20 second timer
  -> Transcript extraction
  -> GeminiClassifier or OpenAIClassifier
  -> ClassificationResult
  -> TelegramNotifier
  -> Optional TRAI SMS confirmation page
```

## Components

### Voice Agent

[src/agent.py](src/agent.py) runs the LiveKit worker. It starts the voice session, stalls the caller, extracts the transcript, and passes it to the classifier after the configured call duration.

### Spam Classifier

[src/spam_classifier.py](src/spam_classifier.py) validates transcript length, loads the configured provider credentials, creates `GeminiClassifier` or `OpenAIClassifier`, and returns a `ClassificationResult`.

### Gemini Development Adapter

[src/gemini_llm.py](src/gemini_llm.py) provides a small Gemini REST adapter for LiveKit LLM responses and spam classification. The development config uses `gemini-3.1-flash-lite`.

### OpenAI Classifier

[src/openai_classifier.py](src/openai_classifier.py) sends the transcript to OpenAI and asks for strict JSON:

```json
{
  "is_spam": true,
  "confidence": 0.95,
  "reason": "Unsolicited commercial offer",
  "evidence_lines": ["Caller: ..."]
}
```

The parser converts the model response into [src/classification_result.py](src/classification_result.py). It handles malformed or failed responses by returning a safe non-spam fallback with the error reason.

### Telegram Notifier

[src/telegram_notifier.py](src/telegram_notifier.py) formats the classification result for Telegram. For spam calls it includes the TRAI complaint draft and, when configured, a `Report to TRAI` confirmation button.

### TRAI Report Handoff

[src/trai_report.py](src/trai_report.py) builds the complaint draft and SMS URL. [src/report_server.py](src/report_server.py) hosts the optional confirmation page that opens the user's SMS composer. The backend does not silently send SMS.

## Configuration

Development uses Gemini free-tier API credentials:

```env
LLM_PROVIDER=gemini
SPAM_CLASSIFIER_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
SPAM_CLASSIFICATION_MODEL=gemini-3.1-flash-lite
```

To switch back to OpenAI, set both providers to `openai` and provide `OPENAI_API_KEY`.

The spam-detection settings in [configs/agent_config.toml](configs/agent_config.toml) control call duration, transcript length, and LLM timeout.

## Security Notes

- Keep `.env` out of version control.
- Do not log API keys or full secrets.
- The TRAI handoff requires explicit user confirmation before SMS submission.
- If the configured provider key is missing, classification returns a safe error result.
