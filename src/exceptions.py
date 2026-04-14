"""Structured exception handling for spam detection system."""


class SpamDetectionError(Exception):
    """Base exception for spam detection errors."""

    def __init__(self, message: str, provider: str | None = None):
        """Initialize the error.

        Args:
            message: Error message
            provider: Optional provider name where the error occurred
        """
        self.provider = provider
        super().__init__(message)

    def __str__(self) -> str:
        """String representation with provider context."""
        if self.provider:
            return f"[{self.provider}] {super().__str__()}"
        return super().__str__()


class ProviderUnavailableError(SpamDetectionError):
    """Raised when no provider is available or configured."""

    def __init__(self, message: str = "No LLM provider credentials configured"):
        """Initialize the error.

        Args:
            message: Error message
        """
        super().__init__(message)


class ProviderInitializationError(SpamDetectionError):
    """Raised when provider initialization fails."""

    def __init__(self, provider: str, message: str):
        """Initialize the error.

        Args:
            provider: Provider name
            message: Error message
        """
        super().__init__(message, provider=provider)


class ClassificationError(SpamDetectionError):
    """Raised when classification fails."""

    def __init__(self, provider: str, message: str, transcript_length: int | None = None):
        """Initialize the error.

        Args:
            provider: Provider name
            message: Error message
            transcript_length: Optional transcript length for context
        """
        self.transcript_length = transcript_length
        super().__init__(message, provider=provider)

    def __str__(self) -> str:
        """String representation with transcript context."""
        base = super().__str__()
        if self.transcript_length is not None:
            return f"{base} (transcript length: {self.transcript_length})"
        return base


class InvalidResponseError(ClassificationError):
    """Raised when LLM response is invalid or malformed."""

    def __init__(self, provider: str, message: str = "Invalid LLM response format"):
        """Initialize the error.

        Args:
            provider: Provider name
            message: Error message
        """
        super().__init__(provider, message)


class TranscriptValidationError(SpamDetectionError):
    """Raised when transcript validation fails."""

    def __init__(self, message: str, transcript_length: int | None = None):
        """Initialize the error.

        Args:
            message: Error message
            transcript_length: Optional transcript length for context
        """
        self.transcript_length = transcript_length
        super().__init__(message)

    def __str__(self) -> str:
        """String representation with transcript context."""
        base = super().__str__()
        if self.transcript_length is not None:
            return f"{base} (transcript length: {self.transcript_length})"
        return base


class ConfigurationError(SpamDetectionError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str | None = None):
        """Initialize the error.

        Args:
            message: Error message
            config_key: Optional configuration key that caused the error
        """
        self.config_key = config_key
        super().__init__(message)

    def __str__(self) -> str:
        """String representation with config context."""
        base = super().__str__()
        if self.config_key:
            return f"{base} (config key: {self.config_key})"
        return base


class TelegramNotificationError(SpamDetectionError):
    """Raised when Telegram notification fails."""

    def __init__(self, message: str, status_code: int | None = None):
        """Initialize the error.

        Args:
            message: Error message
            status_code: Optional HTTP status code
        """
        self.status_code = status_code
        super().__init__(message)

    def __str__(self) -> str:
        """String representation with status code context."""
        base = super().__str__()
        if self.status_code is not None:
            return f"{base} (status: {self.status_code})"
        return base
