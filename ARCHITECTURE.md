# Architecture Documentation

## Overview

The Spam Call Detection System is a LiveKit-based voice agent that answers calls, stalls callers, classifies transcripts as spam or legitimate with OpenAI, and sends Telegram alerts with evidence and an optional TRAI reporting handoff.

## System Flow

```text
Incoming call
  -> LiveKit room
  -> VoiceAgent
     -> STT: Sarvam or mock
     -> Conversation LLM: OpenAI or mock
     -> TTS: Sarvam or mock
     -> 12 second timer
  -> Transcript extraction
  -> OpenAIClassifier
  -> ClassificationResult
  -> TelegramNotifier
  -> Optional TRAI SMS confirmation page
```

## Components

### Voice Agent

[src/agent.py](src/agent.py) runs the LiveKit worker. It starts the voice session, stalls the caller, extracts the transcript, and passes it to the classifier after the configured call duration.

### Spam Classifier

[src/spam_classifier.py](src/spam_classifier.py) validates transcript length, loads OpenAI credentials from the environment, creates `OpenAIClassifier`, and returns a `ClassificationResult`.

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

OpenAI is the only LLM service:

```env
OPENAI_API_KEY=sk-your_openai_api_key
SPAM_CLASSIFICATION_MODEL=gpt-4o-mini
```

The spam-detection settings in [configs/agent_config.toml](configs/agent_config.toml) control call duration, transcript length, and LLM timeout.

## Security Notes

- Keep `.env` out of version control.
- Do not log API keys or full secrets.
- The TRAI handoff requires explicit user confirmation before SMS submission.
- If `OPENAI_API_KEY` is missing or set to a non-OpenAI key, classification returns a safe error result.
