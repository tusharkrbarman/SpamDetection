# Spam Call Detection Agent

LiveKit voice agent that answers unknown calls, stalls the caller for 12 seconds, classifies the transcript as spam or legitimate, and sends you a Telegram alert with evidence.

## Stack

- **LiveKit Agents** - Voice call runtime
- **Sarvam `saaras:v3`** - Speech-to-text (STT)
- **OpenAI `gpt-5.4`** - Real-time conversation (stalling)
- **Multi-Provider LLM** - Spam classification with automatic fallback:
  - **OpenRouter** (free models: Gemma 4 31B)
  - **Kilo AI** (free models: Nemotron 3 Super)
  - **Ollama** (free models: Llama 3.2)
  - **OpenAI** (paid models: GPT-4o-mini)
- **Sarvam `bulbul:v3`** - Text-to-speech (TTS)
- **Telegram Bot API** - Spam alerts with evidence

## How It Works

1. **Call comes in** - Agent answers with "Hello, this line is open. How can I help you?"
2. **Stall phase (12s)** - Agent keeps the caller talking with short filler responses
3. **Call ends** - After 12 seconds, the agent disconnects
4. **Classification** - Full transcript is sent to LLM for spam classification
5. **Telegram alert** - You receive a formatted message with:
   - Spam/Legitimate verdict
   - Confidence score
   - Reason
   - Exact transcript lines that indicate spam (highlighted)
   - Full transcript for reference

## Multi-Provider Support

The system automatically selects the best available LLM provider based on your configured API keys:

### Provider Priority (in order):
1. **OpenRouter** - Free models (Gemma 4 31B)
2. **Kilo AI** - Free models (Nemotron 3 Super)
3. **Ollama** - Free models (Llama 3.2)
4. **OpenAI** - Paid models (GPT-4o-mini)

### Configuration

The system checks for provider credentials in this order:

**OpenRouter:**
```bash
OPENAI_API_KEY=sk-or-...  # Must start with "sk-or-"
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

**Kilo AI:**
```bash
KILO_API_KEY=eyJ...  # Must start with "eyJ"
KILO_API_BASE=https://api.kilo.ai
```

**Ollama:**
```bash
OLLAMA_API_KEY=your-ollama-key
OLLAMA_API_BASE=http://localhost:11434
```

**OpenAI:**
```bash
OPENAI_API_KEY=sk-...  # Must NOT start with "sk-or-"
```

### Custom Model Selection

You can override the default model for any provider:

```bash
SPAM_CLASSIFICATION_MODEL=custom-model-name
```

## Files

- `src/agent.py` - LiveKit worker entrypoint with spam detection pipeline
- `src/spam_classifier.py` - Multi-provider transcript classifier
- `src/telegram_notifier.py` - Telegram alert sender with formatted messages
- `src/config.py` - Centralized configuration management
- `src/providers/` - Provider abstraction layer
  - `base.py` - Base provider interface
  - `openrouter.py` - OpenRouter provider
  - `kilo.py` - Kilo AI provider
  - `ollama.py` - Ollama provider
  - `openai.py` - OpenAI provider
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

3. Fill in credentials (see below for Telegram setup and provider configuration).

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

### LLM Providers (configure at least one)

**Option 1: OpenRouter (Free)**
```
OPENAI_API_KEY=sk-or-your_openrouter_api_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

**Option 2: Kilo AI (Free)**
```
KILO_API_KEY=eyJyour_kilo_api_key
KILO_API_BASE=https://api.kilo.ai
```

**Option 3: Ollama (Free)**
```
OLLAMA_API_KEY=your_ollama_api_key
OLLAMA_API_BASE=http://localhost:11434
```

**Option 4: OpenAI (Paid)**
```
OPENAI_API_KEY=sk-your_openai_api_key
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

## Configuration

- `configs/agent_config.toml` - Tune STT, LLM, TTS, voice behavior, and spam detection settings
  - `[spam_detection].call_duration_seconds` - How long to keep the caller talking (default: 12)
  - `[spam_detection].provider_priority` - Provider priority order (default: ["openrouter", "kilo", "ollama", "openai"])
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

The system uses a provider abstraction pattern for LLM classification:

```
Incoming Call
    ↓
LiveKit Room
    ↓
VoiceAgent (src/agent.py)
    ├─→ STT (Sarvam/Mock) → Transcript
    ├─→ LLM (OpenAI/Mock) → Stalling responses
    └─→ TTS (Sarvam/Mock) → Audio output
    ↓
After 12s timeout
    ↓
Extract Transcript
    ↓
SpamClassifier (spam_classifier.py)
    ├─→ Config (config.py)
    ├─→ Provider Selection
    │   ├─→ OpenRouter (free)
    │   ├─→ Kilo AI (free)
    │   ├─→ Ollama (free)
    │   └─→ OpenAI (paid)
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
# Provider tests
uv run pytest tests/test_providers.py -v

# Integration tests
uv run pytest tests/integration/ -v

# All tests with coverage
uv run pytest tests/ --cov=src --cov-report=html
```
