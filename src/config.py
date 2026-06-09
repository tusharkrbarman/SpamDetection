"""Centralized configuration management for spam detection system."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpamDetectionConfig:
    """Configuration for spam detection."""
    call_duration_seconds: float = 12.0
    max_transcript_length: int = 100_000
    llm_timeout: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SpamDetectionConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            SpamDetectionConfig instance
        """
        data = data or {}
        return cls(
            call_duration_seconds=float(data.get("call_duration_seconds", 12.0)),
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
