"""Domain-specific exception hierarchy."""


class DomainError(Exception):
    """Base class for every error raised by the lor2c domain."""


class ConfigurationError(DomainError):
    """Raised when settings or specifications violate an invariant."""


class ScheduleError(DomainError):
    """Raised when residual routing is asked to do something inconsistent."""


class AdapterError(DomainError):
    """Raised when an adapter is missing or malformed."""


class TemplateError(DomainError):
    """Raised when a prompt template cannot render or parse text."""
