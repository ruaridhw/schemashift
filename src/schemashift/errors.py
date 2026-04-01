"""Custom exception hierarchy for schemashift."""


class SchemaShiftError(Exception):
    """Base exception for all schemashift errors."""


class ConfigValidationError(SchemaShiftError):
    """Raised when a config file or model fails validation."""


class DSLSyntaxError(SchemaShiftError):
    """Raised when a DSL expression cannot be parsed."""

    def __init__(self, message: str, expression: str = "", position: int = -1) -> None:
        super().__init__(message)
        self.expression = expression
        self.position = position


class DSLRuntimeError(SchemaShiftError):
    """Raised when a valid DSL expression fails at evaluation time."""

    def __init__(self, message: str, expression: str = "", target: str = "") -> None:
        super().__init__(message)
        self.expression = expression
        self.target = target


class FormatDetectionError(SchemaShiftError):
    """Raised when automatic format detection fails."""


class AmbiguousFormatError(FormatDetectionError):
    """Raised when multiple formats match with equal confidence."""

    def __init__(self, message: str, candidates: list[str] | None = None) -> None:
        super().__init__(message)
        self.candidates: list[str] = candidates if candidates is not None else []


class SchemaValidationError(SchemaShiftError):
    """Raised when a DataFrame does not conform to a TargetSchema."""


class UnsupportedFileError(SchemaShiftError):
    """Raised when a file extension is not supported by any reader."""


class LLMGenerationError(SchemaShiftError):
    """Raised when LLM-assisted config generation fails."""


class ReaderError(SchemaShiftError):
    """Raised when reading a file fails at the I/O or parsing level."""
