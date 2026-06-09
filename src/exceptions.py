"""Structured exception handling for spam detection system."""


class SpamDetectionError(Exception):
    """Base exception for spam detection errors."""

    def __init__(self, message: str, component: str | None = None):
        """Initialize the error.

        Args:
            message: Error message
            component: Optional component name where the error occurred
        """
        self.component = component
        super().__init__(message)

    def __str__(self) -> str:
        """String representation with component context."""
        if self.component:
            return f"[{self.component}] {super().__str__()}"
        return super().__str__()


class OpenAIConfigurationError(SpamDetectionError):
    """Raised when OpenAI credentials or settings are missing."""

    def __init__(self, message: str = "OpenAI API credentials are not configured"):
        """Initialize the error.

        Args:
            message: Error message
        """
        super().__init__(message, component="openai")


class ClassificationError(SpamDetectionError):
    """Raised when classification fails."""

    def __init__(self, message: str, transcript_length: int | None = None):
        """Initialize the error.

        Args:
            message: Error message
            transcript_length: Optional transcript length for context
        """
        self.transcript_length = transcript_length
        super().__init__(message, component="openai")

    def __str__(self) -> str:
        """String representation with transcript context."""
        base = super().__str__()
        if self.transcript_length is not None:
            return f"{base} (transcript length: {self.transcript_length})"
        return base


class InvalidResponseError(ClassificationError):
    """Raised when LLM response is invalid or malformed."""

    def __init__(self, message: str = "Invalid LLM response format"):
        """Initialize the error.

        Args:
            message: Error message
        """
        super().__init__(message)


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
