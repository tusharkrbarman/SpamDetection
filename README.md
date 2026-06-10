# Spam Call Detection Agent

LiveKit voice agent that answers unknown calls, stalls the caller for 20 seconds, classifies the transcript as spam or legitimate, and sends you a Telegram alert with evidence.

## Stack

- **LiveKit Agents** - Voice call runtime
- **Sarvam `saaras:v3`** - Speech-to-text (STT)
- **Gemini `gemini-3.1-flash-lite`** - Development LLM and spam classification
- **Sarvam `bulbul:v3`** - Text-to-speech (TTS)
- **Telegram Bot API** - Spam alerts with evidence

## How It Works

1. **Call comes in** - Agent answers with "Hello, this line is open. How can I help you?"
2. **Stall phase (20s)** - Agent keeps the caller talking with short filler responses
3. **Call ends** - After 20 seconds, the agent disconnects
4. **Classification** - Full transcript is sent to LLM for spam classification
5. **Telegram alert** - You receive a formatted message with:
   - Spam/Legitimate verdict
   - Confidence score
   - Reason
   - Exact transcript lines that indicate spam (highlighted)
   - Full transcript for reference

## Text-to-Text LLM

Development config uses the Gemini API free tier for both stalling conversation and spam classification:

```bash
LLM_PROVIDER=gemini
SPAM_CLASSIFIER_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
SPAM_CLASSIFICATION_MODEL=gemini-3.1-flash-lite
```

You can switch back to OpenAI by setting both providers to `openai` and adding `OPENAI_API_KEY`.

### Custom Model Selection

You can override the default spam classification model:

```bash
SPAM_CLASSIFICATION_MODEL=custom-model-name
```

## Files

- `src/agent.py` - LiveKit worker entrypoint with spam detection pipeline
- `src/spam_classifier.py` - Provider-selecting transcript classifier
- `src/gemini_llm.py` - Gemini API adapter for development LLM/classification
- `src/telegram_notifier.py` - Telegram alert sender with formatted messages
- `src/trai_report.py` - TRAI complaint draft and SMS handoff helpers
- `src/report_server.py` - Optional confirmation page for Telegram report buttons
- `src/config.py` - Centralized configuration management
- `src/openai_classifier.py` - OpenAI spam classification implementation
- `src/classification_result.py` - Spam classification result model
- `exceptions.py` - Structured error handling
- `configs/agent_config.toml` - Runtime settings for STT, LLM, TTS, voice, and spam detection
- `docs/agent_instructions.md` - Prompt for the stalling voice agent

## Local Setup

1. Install dependencies:

```bash
uv sync
```

2. Create your env file:

```bash
cp .env.example .env
```

3. Fill in credentials (see below for Telegram and Gemini setup).

4. Start the worker:

```bash
uv run python -m src.agent start
```

## Telegram Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot`, follow prompts, copy the **bot token**
3. Start a chat with your new bot and send it any message
4. Visit `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` to find your `chat_id` (it's a number like `123456789`)
5. Add both to your `.env`:

```
TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
TELEGRAM_CHAT_ID=123456789
```

## TRAI Reporting Handoff

For spam calls, Telegram alerts now include a TRAI complaint draft in the SMS format:

```text
Reason, caller-or-sender, dd/mm/yy
```

The user still reviews and sends the SMS to `1909`; the backend does not silently send SMS from a SIM.

To show a Telegram `Report to TRAI` button, expose the optional confirmation server over HTTPS and set `TRAI_REPORT_CONFIRM_URL`:

```bash
uv run python -m src.report_server
```

```env
TRAI_REPORT_CONFIRM_URL=https://your-public-domain.example/report
```

The button opens the confirmation page, which then opens the phone SMS composer with `1909` and the complaint draft prefilled. A future Android companion app can replace this confirmation server/SMS handoff without changing the classifier.

## Environment Variables

### LiveKit Configuration
```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

### Sarvam AI (STT/TTS)
```
SARVAM_API_KEY=your_sarvam_api_key
```

### Text-to-Text LLM
```
LLM_PROVIDER=gemini
SPAM_CLASSIFIER_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
SPAM_CLASSIFICATION_MODEL=gemini-3.1-flash-lite
```

### Optional Configuration
```
SPAM_CLASSIFICATION_MODEL=custom-model-name
```

### Telegram Configuration
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### Optional TRAI Reporting
```
TRAI_REPORT_CONFIRM_URL=https://your-public-domain.example/report
TRAI_REPORT_SERVER_HOST=127.0.0.1
TRAI_REPORT_SERVER_PORT=8787
```

## Configuration

- `configs/agent_config.toml` - Tune STT, LLM, TTS, voice behavior, and spam detection settings
  - `[spam_detection].call_duration_seconds` - How long to keep the caller talking (default: 20)
  - `[spam_detection].max_transcript_length` - Maximum transcript length in characters (default: 100000)
  - `[spam_detection].llm_timeout` - LLM API timeout in seconds (default: 30.0)
- `docs/agent_instructions.md` - Change the stalling agent behavior

## Telephony Setup

To receive actual phone calls, you need to route them into LiveKit. The most common approach:

**Twilio SIP Trunking:**
1. Set up a Twilio phone number
2. Configure Twilio SIP trunk to point to your LiveKit SIP URI
3. LiveKit will create a room for each incoming call and your agent joins it

See [LiveKit SIP docs](https://docs.livekit.io/sip/) for detailed setup.

## Docker

```bash
docker build -t spam-detection-agent .
```

## Architecture

The development config uses Gemini for text-to-text LLM calls:

```
Incoming Call
    ↓
LiveKit Room
    ↓
VoiceAgent (src/agent.py)
    ├─→ STT (Sarvam/Mock) → Transcript
    ├─→ LLM (Gemini or OpenAI) → Stalling responses
    └─→ TTS (Sarvam/Mock) → Audio output
    ↓
After 20s timeout
    ↓
Extract Transcript
    ↓
SpamClassifier (spam_classifier.py)
    ├─→ Config (config.py)
    ├─→ GeminiClassifier or OpenAIClassifier
    │   └─→ classify transcript
    └─→ ClassificationResult
    ↓
TelegramNotifier (telegram_notifier.py)
    ├─→ Build HTML Message
    ├─→ Send to Telegram API
    └─→ Return success/failure
    ↓
User receives alert
```

## Testing

Run the test suite:

```bash
uv run pytest tests/ -v
```

Run specific test categories:

```bash
# Classifier tests
uv run pytest tests/test_openai_classifier.py -v

# Integration tests
uv run pytest tests/integration/ -v

# All tests with coverage
uv run pytest tests/ --cov=src --cov-report=html
```
