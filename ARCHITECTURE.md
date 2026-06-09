# Architecture Documentation

## Overview

The Spam Call Detection System is a LiveKit-based voice agent that answers calls, stalls callers, classifies transcripts as spam/legitimate, and sends Telegram alerts. The system uses a provider abstraction pattern to support multiple LLM providers with automatic fallback.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Incoming Call                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LiveKit Room                              │
│  - Audio streaming                                              │
│  - Real-time communication                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VoiceAgent (agent.py)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STT (Speech-to-Text)                                    │  │
│  │  - Sarvam AI / Mock                                      │  │
│  │  - Transcribes audio to text                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LLM (Language Model)                                    │  │
│  │  - OpenAI / Mock                                         │  │
│  │  - Generates stalling responses                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TTS (Text-to-Speech)                                    │  │
│  │  - Sarvam AI / Mock                                      │  │
│  │  - Converts text to audio                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Call Timer                                              │  │
│  │  - 12 second timeout                                     │  │
│  │  - Triggers classification                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Extract Transcript                            │
│  - Collects all user/agent messages                             │
│  - Formats as readable text                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              SpamClassifier (spam_classifier.py)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Config (config.py)                                      │  │
│  │  - Loads configuration                                   │  │
│  │  - Validates transcript length                           │  │
│  │  - Selects provider                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Provider Selection                                      │  │
│  │  - Checks available credentials                          │  │
│  │  - Prioritizes free providers                            │  │
│  │  - Falls back to paid if needed                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Provider Abstraction (providers/)                       │  │
│  │  ┌──────────────┐  ┌──────────────┐                     │  │
│  │  │ OpenAI       │  │  ChatGPT      │                     │  │
│  │  │ Provider     │  │  Only         │                     │  │
│  │  └──────────────┘  └──────────────┘                     │  │
│  │  ┌──────────────┐  ┌──────────────┐                     │  │
│  │  │ Classification result     │                     │  │
│  │  │ is_spam/confidence/reason │                     │  │
│  │  └──────────────┘  └──────────────┘                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ClassificationResult                                     │  │
│  │  - is_spam: bool                                         │  │
│  │  - confidence: float                                     │  │
│  │  - reason: str                                           │  │
│  │  - evidence_lines: list[str]                             │  │
│  │  - full_transcript: str                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           TelegramNotifier (telegram_notifier.py)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Message Builder                                         │  │
│  │  - Formats HTML message                                   │  │
│  │  - Adds confidence bar                                    │  │
│  │  - Highlights evidence                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Telegram API                                            │  │
│  │  - Sends formatted message                               │  │
│  │  - Handles errors                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      User Receives Alert                          │
│  - Spam/Legitimate verdict                                      │
│  - Confidence score                                             │
│  - Reason                                                        │
│  - Evidence lines                                                │
│  - Full transcript                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. VoiceAgent (`agent.py`)

**Purpose**: Main LiveKit agent that handles voice calls.

**Key Components**:
- **STT**: Transcribes audio to text using Sarvam AI or mock
- **LLM**: Generates stalling responses using OpenAI or mock
- **TTS**: Converts text to audio using Sarvam AI or mock
- **Call Timer**: 12-second timeout to trigger classification

**Data Flow**:
```
Audio Input → STT → Transcript → LLM → Response → TTS → Audio Output
                                                    ↓
                                              Chat Context
                                                    ↓
                                              After 12s
                                                    ↓
                                         Extract Transcript
```

### 2. SpamClassifier (`spam_classifier.py`)

**Purpose**: Classifies transcripts as spam or legitimate using OpenAI/ChatGPT.

**Key Components**:
- **Config Management**: Loads and validates configuration
- **Provider Selection**: Uses OpenAIProvider only
- **Classification**: Calls OpenAI to classify transcript
- **Error Handling**: Graceful fallback on errors

**Provider**:
1. OpenAI / ChatGPT

**Data Flow**:
```
Transcript → Config Validation → Provider Selection → Classification → Result
```

### 3. Provider Abstraction (`providers/`)

**Purpose**: Abstract interface for the OpenAI classification provider.

**Base Interface** (`base.py`):
```python
class LLMProvider(ABC):
    async def classify(transcript: str) -> ClassificationResult
    async def close()
    def get_model_name() -> str
    def get_provider_name() -> str
```

**Implementation**:
- **OpenAIProvider**: OpenAI/ChatGPT classification

**Benefits**:
- Clear provider boundary
- Better testability with mocks
- Clear separation of concerns

### 4. Configuration Management (`config.py`)

**Purpose**: Centralized configuration management with type safety.

**Key Components**:
- **ProviderConfig**: Configuration for specific provider
- **SpamDetectionConfig**: Spam detection settings
- **TelegramConfig**: Telegram notification settings
- **AppConfig**: Main application configuration

**Features**:
- Type-safe dataclasses
- Environment variable loading
- TOML file support
- Validation and defaults

### 5. Error Handling (`exceptions.py`)

**Purpose**: Structured exception handling for better debugging.

**Exception Hierarchy**:
```
SpamDetectionError (base)
├── ProviderUnavailableError
├── ProviderInitializationError
├── ClassificationError
│   └── InvalidResponseError
├── TranscriptValidationError
├── ConfigurationError
└── TelegramNotificationError
```

**Benefits**:
- Clear error types
- Contextual information
- Better error messages
- Easier debugging

### 6. TelegramNotifier (`telegram_notifier.py`)

**Purpose**: Sends formatted Telegram alerts with classification results.

**Key Components**:
- **Message Builder**: Formats HTML messages
- **Telegram API**: Sends messages via bot API
- **Error Handling**: Graceful fallback on errors

**Message Format**:
```
🚨 SPAM CALL DETECTED

Confidence: 95%
Reason: Unsolicited financial product offer

Evidence from transcript:
• Caller: Hello, I'm calling from Bank of America...
• Caller: We have a special limited-time offer...

Full transcript:
[Full transcript text]

2024-01-15 10:30:45 UTC
```

## Data Models

### ClassificationResult

```python
@dataclass(frozen=True)
class ClassificationResult:
    is_spam: bool              # Whether the call is spam
    confidence: float          # Confidence score (0.0-1.0)
    reason: str                # Explanation of classification
    evidence_lines: list[str]  # Transcript lines indicating spam
    full_transcript: str       # Complete transcript
```

### ProviderConfig

```python
@dataclass(frozen=True)
class ProviderConfig:
    api_key: str               # API key for authentication
    base_url: str | None       # Optional base URL
    model: str | None          # Optional model name
```

### SpamDetectionConfig

```python
@dataclass(frozen=True)
class SpamDetectionConfig:
    call_duration_seconds: float        # Call duration before classification
    provider_priority: list[Provider]   # Provider priority order
    max_transcript_length: int          # Maximum transcript length
    llm_timeout: float                  # LLM API timeout
```

## Design Patterns

### 1. Provider Abstraction Pattern

**Purpose**: Decouple classification logic from specific LLM providers.

**Implementation**:
- Abstract base class defines interface
- Concrete implementations for each provider
- Factory method selects appropriate provider

**Benefits**:
- Easy to add new providers
- Consistent interface
- Better testability

### 2. Configuration Pattern

**Purpose**: Centralized, type-safe configuration management.

**Implementation**:
- Dataclasses for configuration
- Environment variable loading
- TOML file support
- Validation and defaults

**Benefits**:
- Type safety
- Single source of truth
- Easy to test

### 3. Error Handling Pattern

**Purpose**: Structured exception handling with context.

**Implementation**:
- Exception hierarchy
- Contextual information
- Clear error messages

**Benefits**:
- Better debugging
- Clear error types
- Easier error handling

## Security Considerations

### 1. API Key Management

- API keys stored in environment variables
- No hardcoded credentials in source code
- Key prefix validation for provider detection
- `.env` files excluded from version control

### 2. Input Validation

- Transcript length validation (max 100,000 characters)
- Empty transcript handling
- Whitespace-only transcript handling

### 3. Error Handling

- Graceful fallback on errors
- No sensitive information in error messages
- Proper exception logging

### 4. Resource Management

- Proper cleanup of LLM clients
- Try/finally blocks for resource cleanup
- Timeout configuration for API calls

## Performance Considerations

### 1. Provider Selection

- Prioritizes free providers
- Automatic fallback to paid providers
- Credential caching

### 2. Resource Management

- Async/await for non-blocking operations
- Proper client cleanup
- Timeout configuration

### 3. Error Handling

- Graceful fallback on errors
- No blocking operations
- Proper logging

## Testing Strategy

### 1. Unit Tests

- Provider implementations
- Configuration management
- Error handling
- Data models

### 2. Integration Tests

- Full pipeline tests
- Provider selection
- Error scenarios
- Resource cleanup

### 3. Test Coverage

- Provider-specific tests
- Multi-provider tests
- Error handling tests
- Configuration tests

## Deployment Considerations

### 1. Environment Variables

- Required: LiveKit credentials, at least one LLM provider
- Optional: Telegram credentials, custom model

### 2. Configuration

- TOML file for runtime settings
- Environment variables for credentials
- Default values for optional settings

### 3. Dependencies

- LiveKit Agents SDK
- OpenAI Python SDK
- httpx for HTTP requests
- python-dotenv for environment variables

## Future Enhancements

### 1. Additional Providers

- Add more LLM providers
- Support for custom providers
- Provider health monitoring

### 2. Advanced Features

- Multi-language support
- Custom classification rules
- Machine learning models
- Real-time classification

### 3. Monitoring

- Metrics collection
- Performance monitoring
- Error tracking
- Usage analytics

### 4. Scalability

- Horizontal scaling
- Load balancing
- Caching
- Rate limiting
