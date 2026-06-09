# High-Level Design: Spam Call Detection Agent

## Architecture Diagram

```mermaid
graph TB
    subgraph "Telephony Layer"
        Caller["📞 Spam Caller"]
        Twilio["Twilio Phone Number"]
        SIP["LiveKit SIP Trunk"]
    end

    subgraph "LiveKit Infrastructure"
        LKServer["LiveKit Server"]
        Room["Call Room"]
    end

    subgraph "Voice Agent Worker"
        Agent["VoiceAgent\n(Stalling Agent)"]
        STT["Sarvam STT\nsaaras:v3"]
        LLM["OpenAI LLM\ngpt-5.4"]
        TTS["Sarvam TTS\nbulbul:v3"]
        Timer["12s Call Timer"]
    end

    subgraph "Post-Call Pipeline"
        Extractor["Transcript\nExtractor"]
        Classifier["Spam Classifier\nGPT-4o-mini"]
        Notifier["Telegram\nNotifier"]
    end

    subgraph "External Services"
        Telegram["📱 Telegram Bot API"]
        User["👤 User Phone"]
    end

    Caller -->|Phone Call| Twilio
    Twilio -->|SIP INVITE| SIP
    SIP -->|Creates Room| LKServer
    LKServer -->|Joins Room| Room
    Room -->|Audio Stream| Agent

    Agent -->|Audio| STT
    STT -->|Transcript Text| Agent
    Agent -->|Prompt| LLM
    LLM -->|Response Text| Agent
    Agent -->|Text to Speak| TTS
    TTS -->|Audio| Agent

    Agent -->|Starts| Timer
    Timer -->|After 12s| Extractor

    Extractor -->|Chat Context| Extractor
    Extractor -->|Full Transcript| Classifier
    Classifier -->|Structured Result| Notifier
    Notifier -->|Formatted Alert| Telegram
    Telegram -->|Spam Alert Message| User

    classDef telephony fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef livekit fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef agent fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef pipeline fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef external fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class Caller,Twilio,SIP telephony
    class LKServer,Room livekit
    class Agent,STT,LLM,TTS,Timer agent
    class Extractor,Classifier,Notifier pipeline
    class Telegram,User external
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant C as 📞 Caller
    participant T as Twilio
    participant LK as LiveKit Server
    participant VA as VoiceAgent
    participant STT as Sarvam STT
    participant LLM as OpenAI GPT-5.4
    participant TTS as Sarvam TTS
    participant Timer as 12s Timer
    participant Clf as Spam Classifier
    participant TG as Telegram Bot
    participant U as 👤 User

    C->>T: Incoming call
    T->>LK: SIP INVITE → Create Room
    LK->>VA: Join Room
    VA->>TTS: "Hello, this line is open..."
    TTS-->>VA: Audio
    VA->>LK: Play greeting
    LK->>C: Audio greeting

    loop Stall Phase (12 seconds)
        C->>LK: Caller speaks
        LK->>VA: Audio stream
        VA->>STT: Audio chunks
        STT-->>VA: Transcript text
        VA->>LLM: Prompt with transcript
        LLM-->>VA: "I see, go on..."
        VA->>TTS: Response text
        TTS-->>VA: Audio
        VA->>LK: Play response
        LK->>C: Audio response
    end

    Timer-->>VA: 12s elapsed
    VA->>LK: Disconnect call
    LK-->>C: Call ended

    VA->>VA: Extract transcript from chat_ctx
    VA->>Clf: Full transcript text
    Clf->>Clf: Analyze with GPT-4o-mini
    Clf-->>VA: {is_spam, confidence, reason, evidence_lines}

    alt is_spam == true
        VA->>TG: Send formatted alert
        TG-->>U: 🚨 SPAM CALL DETECTED\n+ evidence lines
    else is_spam == false
        VA->>TG: Send alert
        TG-->>U: ✅ LEGITIMATE CALL
    end
```

## Component Details

### 1. Telephony Layer
| Component | Purpose |
|-----------|---------|
| Twilio | Provides phone number and SIP trunking |
| LiveKit SIP | Converts PSTN calls to WebRTC rooms |

### 2. Voice Agent (Real-time)
| Component | Model/Service | Role |
|-----------|---------------|------|
| STT | Sarvam `saaras:v3` | Converts caller audio to text |
| LLM | OpenAI `gpt-5.4` | Generates stalling responses |
| TTS | Sarvam `bulbul:v3` | Converts responses to audio |
| Timer | Internal | Triggers classification after 12s |

### 3. Post-Call Pipeline
| Component | Model | Input | Output |
|-----------|-------|-------|--------|
| Transcript Extractor | - | LiveKit chat_ctx | Formatted transcript string |
| Spam Classifier | OpenAI `gpt-4o-mini` | Transcript | JSON: {is_spam, confidence, reason, evidence_lines} |
| Telegram Notifier | Telegram Bot API | Classification result | Formatted HTML message |

### 4. Data Flow

```
Audio → STT → Text → LLM → Response → TTS → Audio (loop for 12s)
                                                    ↓
                                              Transcript
                                                    ↓
                                              Classifier
                                                    ↓
                                        {is_spam, confidence, reason, evidence}
                                                    ↓
                                             Telegram Alert
```

## Configuration Points

| Setting | File | Default | Description |
|---------|------|---------|-------------|
| Call duration | `agent_config.toml` | 12s | How long to stall caller |
| Classification model | `agent_config.toml` | gpt-4o-mini | Model for spam detection |
| Stalling behavior | `agent_instructions.md` | - | Prompt for voice agent |
| STT language | `agent_config.toml` | unknown | Auto-detect caller language |
| TTS voice | `agent_config.toml` | ajay | Voice for agent responses |
