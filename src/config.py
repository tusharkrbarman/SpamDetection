"""Centralized configuration management for spam detection system."""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Provider(str, Enum):
    """Available LLM providers."""
    OPENROUTER = "openrouter"
    KILO = "kilo"
    OLLAMA = "ollama"
    OPENAI = "openai"


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a specific provider."""
    api_key: str
    base_url: str | None = None
    model: str | None = None

    @classmethod
    def from_env(cls, provider: Provider) -> "ProviderConfig | None":
        """Create provider config from environment variables.

        Args:
            provider: The provider to create config for

        Returns:
            ProviderConfig if credentials are available, None otherwise
        """
        if provider == Provider.OPENROUTER:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_API_BASE", "")
            model = os.environ.get("SPAM_CLASSIFICATION_MODEL")
            if api_key.startswith("sk-or-") and base_url:
                return cls(api_key=api_key, base_url=base_url, model=model)

        elif provider == Provider.KILO:
            api_key = os.environ.get("KILO_API_KEY", "")
            base_url = os.environ.get("KILO_API_BASE", "")
            model = os.environ.get("SPAM_CLASSIFICATION_MODEL")
            if api_key.startswith("eyJ") and base_url:
                return cls(api_key=api_key, base_url=base_url, model=model)

        elif provider == Provider.OLLAMA:
            api_key = os.environ.get("OLLAMA_API_KEY", "")
            base_url = os.environ.get("OLLAMA_API_BASE", "")
            model = os.environ.get("SPAM_CLASSIFICATION_MODEL")
            if api_key and base_url:
                return cls(api_key=api_key, base_url=base_url, model=model)

        elif provider == Provider.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            model = os.environ.get("SPAM_CLASSIFICATION_MODEL")
            if api_key and not api_key.startswith("sk-or-"):
                return cls(api_key=api_key, base_url=None, model=model)

        return None


@dataclass(frozen=True)
class SpamDetectionConfig:
    """Configuration for spam detection."""
    call_duration_seconds: float = 12.0
    provider_priority: list[Provider] = None
    max_transcript_length: int = 100_000
    llm_timeout: float = 30.0

    def __post_init__(self):
        """Set default provider priority if not provided."""
        if self.provider_priority is None:
            object.__setattr__(
                self,
                "provider_priority",
                [Provider.OPENROUTER, Provider.KILO, Provider.OLLAMA, Provider.OPENAI],
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SpamDetectionConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            SpamDetectionConfig instance
        """
        data = data or {}
        provider_priority = data.get("provider_priority")
        if provider_priority:
            # Convert string list to Provider enum
            provider_priority = [Provider(p) for p in provider_priority]

        return cls(
            call_duration_seconds=float(data.get("call_duration_seconds", 12.0)),
            provider_priority=provider_priority,
            max_transcript_length=int(data.get("max_transcript_length", 100_000)),
            llm_timeout=float(data.get("llm_timeout", 30.0)),
        )

    @classmethod
    def from_toml(cls, path: Path) -> "SpamDetectionConfig":
        """Load config from TOML file.

        Args:
            path: Path to TOML configuration file

        Returns:
            SpamDetectionConfig instance
        """
        import tomllib

        with open(path, "rb") as config_file:
            data = tomllib.load(config_file)
            return cls.from_dict(data.get("spam_detection"))

    def get_available_provider(self) -> ProviderConfig | None:
        """Get the first available provider based on priority.

        Returns:
            ProviderConfig for the first available provider, None if none available
        """
        for provider in self.provider_priority:
            config = ProviderConfig.from_env(provider)
            if config:
                return config
        return None


@dataclass(frozen=True)
class TelegramConfig:
    """Configuration for Telegram notifications."""
    bot_token: str | None = None
    chat_id: str | None = None
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        """Create Telegram config from environment variables.

        Returns:
            TelegramConfig instance
        """
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        enabled = bool(bot_token and chat_id and bot_token != "your_telegram_bot_token")

        return cls(bot_token=bot_token, chat_id=chat_id, enabled=enabled)


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    spam_detection: SpamDetectionConfig
    telegram: TelegramConfig

    @classmethod
    def load(cls, config_path: Path | None = None) -> "AppConfig":
        """Load application configuration.

        Args:
            config_path: Optional path to TOML configuration file

        Returns:
            AppConfig instance
        """
        if config_path and config_path.exists():
            spam_detection = SpamDetectionConfig.from_toml(config_path)
        else:
            spam_detection = SpamDetectionConfig()

        telegram = TelegramConfig.from_env()

        return cls(spam_detection=spam_detection, telegram=telegram)
