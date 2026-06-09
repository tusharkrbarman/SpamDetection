import asyncio
import logging
import os
import tomllib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, llm as livekit_llm
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice import Agent, AgentSession, room_io
from livekit.plugins import noise_cancellation, silero

load_dotenv(override=True)

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "agent_config.toml"
INSTRUCTIONS_PATH = PROJECT_ROOT / "docs" / "agent_instructions.md"


def _clamp_pace(value: float | int | str | None) -> float:
    try:
        pace = float(value) if value is not None else 1.0
    except (TypeError, ValueError):
        return 1.0
    return min(max(pace, 0.5), 2.0)


def _clamp_temperature(value: float | int | str | None) -> float:
    try:
        temperature = float(value) if value is not None else 0.35
    except (TypeError, ValueError):
        return 0.35
    return min(max(temperature, 0.01), 1.0)


@dataclass(frozen=True)
class STTConfig:
    language: str = "unknown"
    mode: str = "transcribe"
    high_vad_sensitivity: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "STTConfig":
        data = data or {}
        return cls(
            language=str(data.get("language", "unknown")),
            mode=str(data.get("mode", "transcribe")),
            high_vad_sensitivity=bool(data.get("high_vad_sensitivity", True)),
        )


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-5.4"
    reasoning_effort: str = "none"
    verbosity: str = "low"
    max_output_tokens: int = 96
    use_websocket: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LLMConfig":
        data = data or {}
        max_output_tokens = int(data.get("max_output_tokens", 96))
        if max_output_tokens < 1:
            raise ValueError("llm.max_output_tokens must be greater than 0")
        return cls(
            model=str(data.get("model", "gpt-5.4")),
            reasoning_effort=str(data.get("reasoning_effort", "none")),
            verbosity=str(data.get("verbosity", "low")),
            max_output_tokens=max_output_tokens,
            use_websocket=bool(data.get("use_websocket", True)),
        )


@dataclass(frozen=True)
class TTSConfig:
    voice: str = "priya"
    pace: float = 1.0
    temperature: float = 0.35
    sample_rate: int = 16000
    target_language: str = "en-IN"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TTSConfig":
        data = data or {}
        return cls(
            voice=str(data.get("voice", "priya")),
            pace=_clamp_pace(data.get("pace", 1.0)),
            temperature=_clamp_temperature(data.get("temperature", 0.35)),
            sample_rate=int(data.get("sample_rate", 16000)),
            target_language=str(data.get("target_language", "en-IN")),
        )


@dataclass(frozen=True)
class VoiceConfig:
    min_endpointing_delay: float = 0.2
    max_endpointing_delay: float = 0.9
    min_interruption_duration: float = 0.7
    min_interruption_words: int = 1
    false_interruption_timeout: float = 1.5
    resume_false_interruption: bool = True
    preemptive_generation: bool = True
    user_away_timeout: float = 12.0
    min_consecutive_speech_delay: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VoiceConfig":
        data = data or {}
        return cls(
            min_endpointing_delay=float(data.get("min_endpointing_delay", 0.2)),
            max_endpointing_delay=float(data.get("max_endpointing_delay", 0.9)),
            min_interruption_duration=float(data.get("min_interruption_duration", 0.7)),
            min_interruption_words=int(data.get("min_interruption_words", 1)),
            false_interruption_timeout=float(data.get("false_interruption_timeout", 1.5)),
            resume_false_interruption=bool(data.get("resume_false_interruption", True)),
            preemptive_generation=bool(data.get("preemptive_generation", True)),
            user_away_timeout=float(data.get("user_away_timeout", 12.0)),
            min_consecutive_speech_delay=float(data.get("min_consecutive_speech_delay", 0.1)),
        )


@dataclass(frozen=True)
class SpamDetectionConfig:
    call_duration_seconds: float = 12.0
    classification_model: str = "gpt-4o-mini"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SpamDetectionConfig":
        data = data or {}
        return cls(
            call_duration_seconds=float(data.get("call_duration_seconds", 12.0)),
            classification_model=str(data.get("classification_model", "gpt-4o-mini")),
        )


@dataclass(frozen=True)
class AgentConfig:
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    voice: VoiceConfig = VoiceConfig()
    spam_detection: SpamDetectionConfig = SpamDetectionConfig()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AgentConfig":
        data = data or {}
        return cls(
            stt=STTConfig.from_dict(data.get("stt")),
            llm=LLMConfig.from_dict(data.get("llm")),
            tts=TTSConfig.from_dict(data.get("tts")),
            voice=VoiceConfig.from_dict(data.get("voice")),
            spam_detection=SpamDetectionConfig.from_dict(data.get("spam_detection")),
        )


class MockSTT:
    """Mock STT that returns predefined responses for testing"""
    def __init__(self, *args, **kwargs):
        pass

    async def recognize(self, buffer):
        # Return empty recognition for testing
        from livekit.agents import stt
        return stt.SpeechEvent(
            type=stt.SpeechEventType.INTERIM,
            alternatives=[stt.SpeechData(text="", language="")],
        )


class MockLLM(livekit_llm.LLM):
    """Mock LLM that returns stall phrases for testing"""
    def __init__(self) -> None:
        super().__init__()
        self._responses = [
            "I see",
            "Go on",
            "Can you tell me more?",
            "Okay, I'm listening",
            "Hmm, interesting",
            "Please continue",
            "I understand",
            "That's good to know",
        ]
        self._index = 0

    def chat(self, **kwargs):
        # Return a mock response
        from livekit.agents.llm import ChatChunk, ChoiceDelta
        response_text = self._responses[self._index % len(self._responses)]
        self._index += 1
        
        chunk = ChatChunk(
            id="mock",
            choices=[ChoiceDelta(delta={"role": "assistant", "content": response_text})],
            created=0,
            model="mock",
        )
        return AsyncIteratorMock([chunk])


class MockTTS:
    """Mock TTS that does nothing"""
    def __init__(self, *args, **kwargs):
        pass

    async def synthesize(self, text, **kwargs):
        from livekit.agents import tts
        return tts.SynthesizedAudio(
            data=b"", sample_rate=16000, num_channels=1
        )


class AsyncIteratorMock:
    def __init__(self, items):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def load_config() -> AgentConfig:
    with open(CONFIG_PATH, "rb") as config_file:
        return AgentConfig.from_dict(tomllib.load(config_file))


def load_instructions(path: str | Path = INSTRUCTIONS_PATH) -> str:
    with open(path, "r", encoding="utf-8") as instructions_file:
        return instructions_file.read().strip()


def create_stt(config: AgentConfig):
    """Create STT instance based on configuration."""
    api_key = os.environ.get("SARVAM_API_KEY")
    
    # Use mock STT if no Sarvam key
    if not api_key or api_key == "your_sarvam_api_key":
        logger.info("Using mock STT (no Sarvam API key)")
        return MockSTT()
    
    # Real Sarvam STT
    logger.info("Using Sarvam STT")
    from livekit.plugins import sarvam
    return sarvam.STT(
        language=config.stt.language,
        model="saarika:v2.5",
        mode=config.stt.mode,
        api_key=api_key,
        high_vad_sensitivity=config.stt.high_vad_sensitivity,
        sample_rate=config.tts.sample_rate,
    )


def create_llm(config: AgentConfig):
    """Create LLM instance based on configuration."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Use mock LLM if no OpenAI key
    if not api_key or api_key == "your_openai_api_key" or api_key.startswith("sk-or-"):
        logger.info("Using mock LLM (no OpenAI API key)")
        return MockLLM()
    
    # Real OpenAI LLM
    logger.info("Using OpenAI LLM")
    from livekit.plugins import openai as livekit_openai
    
    return livekit_openai.LLM(
        model=config.llm.model,
        api_key=api_key,
        reasoning_effort=config.llm.reasoning_effort,
        max_tokens=config.llm.max_output_tokens,
    )


def create_tts(config: AgentConfig):
    """Create TTS instance based on configuration."""
    api_key = os.environ.get("SARVAM_API_KEY")
    
    # Use mock TTS if no Sarvam key
    if not api_key or api_key == "your_sarvam_api_key":
        logger.info("Using mock TTS (no Sarvam API key)")
        return MockTTS()
    
    # Real Sarvam TTS
    logger.info("Using Sarvam TTS")
    from livekit.plugins import sarvam
    return sarvam.TTS(
        target_language_code=config.tts.target_language,
        model="bulbul:v2",
        speaker=config.tts.voice,
        speech_sample_rate=config.tts.sample_rate,
        pace=config.tts.pace,
        temperature=config.tts.temperature,
        api_key=api_key,
    )


def extract_transcript_from_chat_ctx(chat_ctx) -> str:
    lines = []
    for msg in chat_ctx.messages:
        if msg.role == "user" and isinstance(msg.content, str):
            text = msg.content.strip()
            if text:
                lines.append(f"Caller: {text}")
        elif msg.role == "assistant" and isinstance(msg.content, str):
            text = msg.content.strip()
            if text:
                lines.append(f"Agent: {text}")
    return "\n".join(lines)


def extract_caller_number_from_room(room) -> str | None:
    participants = getattr(room, "remote_participants", None)
    if not participants:
        return None

    if isinstance(participants, dict):
        participant_iterable = participants.values()
    else:
        participant_iterable = participants

    for participant in participant_iterable:
        attributes = getattr(participant, "attributes", None) or {}
        number = (
            attributes.get("sip.phoneNumber")
            or attributes.get("phone_number")
            or attributes.get("caller_number")
        )
        if number:
            return str(number)

    return None


class MockClassificationResult:
    def __init__(self, is_spam: bool, confidence: float, reason: str, evidence_lines: list[str], transcript: str):
        self.is_spam = is_spam
        self.confidence = confidence
        self.reason = reason
        self.evidence_lines = evidence_lines
        self.full_transcript = transcript


async def mock_classify_transcript(transcript: str) -> MockClassificationResult:
    """Mock classification that alternates between spam and legit"""
    # Simple heuristic for demo
    spam_indicators = ["loan", "insurance", "credit card", "offer", "discount", "limited time", "act now"]
    transcript_lower = transcript.lower()
    
    spam_score = sum(1 for indicator in spam_indicators if indicator in transcript_lower)
    is_spam = spam_score > 0 and len(transcript) > 20
    
    if is_spam:
        evidence = [line for line in transcript.split('\n') 
                   if any(indicator in line.lower() for indicator in spam_indicators)]
        return MockClassificationResult(
            is_spam=True,
            confidence=0.85,
            reason="Detected sales/promotion language",
            evidence_lines=evidence[:2],  # Max 2 evidence lines
            transcript=transcript
        )
    else:
        return MockClassificationResult(
            is_spam=False,
            confidence=0.75,
            reason="No spam indicators detected",
            evidence_lines=[],
            transcript=transcript
        )


async def mock_send_spam_alert(result: MockClassificationResult) -> bool:
    """Mock Telegram alert - just print to console"""
    print("\n" + "="*60)
    print("🚨 MOCK TELEGRAM ALERT 🚨")
    print("="*60)
    print(f"Spam Status: {'SPAM' if result.is_spam else 'LEGITIMATE'}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Reason: {result.reason}")
    if result.evidence_lines:
        print("\nEvidence:")
        for line in result.evidence_lines:
            print(f"  • {line}")
    print(f"\nTranscript ({len(result.full_transcript)} chars):")
    print("-" * 40)
    print(result.full_transcript)
    print("-" * 40)
    print("="*60 + "\n")
    return True


class VoiceAgent(Agent):
    def __init__(
        self,
        *,
        config: AgentConfig | None = None,
        instructions: str | None = None,
        caller_number: str | None = None,
        call_ended_event: asyncio.Event | None = None,
    ) -> None:
        resolved_config = config or load_config()
        self._config = resolved_config
        self._caller_number = caller_number
        self._call_ended_event = call_ended_event or asyncio.Event()
        super().__init__(
            instructions=instructions or load_instructions(),
            stt=create_stt(resolved_config),
            llm=create_llm(resolved_config),
            tts=create_tts(resolved_config),
        )

    async def on_enter(self) -> None:
        self.session.say(
            "Hello, this line is open. How can I help you?",
            allow_interruptions=True,
            add_to_chat_ctx=True,
        )

        asyncio.create_task(self._end_call_after_timeout())

    async def _end_call_after_timeout(self) -> None:
        duration = self._config.spam_detection.call_duration_seconds
        logger.info("Call will end in %.1f seconds for classification", duration)
        await asyncio.sleep(duration)
        logger.info("Call duration reached, processing transcript...")
        await self._process_and_notify()
        self._call_ended_event.set()

    async def _process_and_notify(self) -> None:
        try:
            transcript = extract_transcript_from_chat_ctx(self.chat_ctx)
            logger.info("Extracted transcript (%d chars)", len(transcript))

            # Use real spam classifier
            from src.spam_classifier import classify_transcript
            result = await classify_transcript(transcript)

            if result.is_spam:
                logger.warning("SPAM detected: %s (confidence: %.2f)", result.reason, result.confidence)
            else:
                logger.info("Legitimate call: %s (confidence: %.2f)", result.reason, result.confidence)

            # Use real Telegram notifier if configured, otherwise console output
            from src.config import TelegramConfig
            telegram_config = TelegramConfig.from_env()
            
            if telegram_config.enabled:
                from src.telegram_notifier import send_spam_alert
                sent = await send_spam_alert(result, caller_number=self._caller_number)
                if sent:
                    logger.info("Telegram alert sent successfully")
                else:
                    logger.warning("Failed to send Telegram alert")
            else:
                # Fall back to console output
                await self._send_console_alert(result)

        except Exception as e:
            logger.error("Error in post-call processing: %s", e, exc_info=True)

    async def _send_console_alert(self, result) -> None:
        """Send alert to console (fallback when Telegram not configured)"""
        print("\n" + "="*60)
        print("🚨 SPAM ALERT 🚨" if result.is_spam else "✅ LEGITIMATE CALL")
        print("="*60)
        print(f"Spam Status: {'SPAM' if result.is_spam else 'LEGITIMATE'}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Reason: {result.reason}")
        if result.evidence_lines:
            print("\nEvidence:")
            for line in result.evidence_lines:
                print(f"  • {line}")
        print(f"\nTranscript ({len(result.full_transcript)} chars):")
        print("-" * 40)
        print(result.full_transcript)
        print("-" * 40)
        print("="*60 + "\n")


async def entrypoint(ctx: JobContext):
    logger.info("Worker joined room: %s", ctx.room.name)

    config = load_config()
    instructions = load_instructions()
    call_ended_event = asyncio.Event()
    caller_number = extract_caller_number_from_room(ctx.room)
    if caller_number:
        logger.info("Detected caller number from room attributes: %s", caller_number)

    vad = ctx.proc.userdata.get("vad")
    if vad is None:
        vad = silero.VAD.load(
            min_speech_duration=0.08,
            min_silence_duration=0.45,
            prefix_padding_duration=0.35,
            activation_threshold=0.55,
            sample_rate=16000,
        )
        ctx.proc.userdata["vad"] = vad

    session = AgentSession(
        vad=vad,
        allow_interruptions=True,
        min_interruption_duration=config.voice.min_interruption_duration,
        min_interruption_words=config.voice.min_interruption_words,
        min_endpointing_delay=config.voice.min_endpointing_delay,
        max_endpointing_delay=config.voice.max_endpointing_delay,
        false_interruption_timeout=config.voice.false_interruption_timeout,
        resume_false_interruption=config.voice.resume_false_interruption,
        min_consecutive_speech_delay=config.voice.min_consecutive_speech_delay,
        user_away_timeout=config.voice.user_away_timeout,
        preemptive_generation=config.voice.preemptive_generation,
        tts_text_transforms=["filter_markdown", "filter_emoji"],
    )

    agent = VoiceAgent(
        config=config,
        instructions=instructions,
        caller_number=caller_number,
        call_ended_event=call_ended_event,
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=room_io.RoomInputOptions(
            audio_sample_rate=16000,
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    await call_ended_event.wait()
    logger.info("Call processing complete, disconnecting from room")
    await ctx.room.disconnect()


def prewarm(proc) -> None:
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.08,
        min_silence_duration=0.45,
        prefix_padding_duration=0.35,
        activation_threshold=0.55,
        sample_rate=16000,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
