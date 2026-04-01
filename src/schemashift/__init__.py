"""schemashift — Declarative file format transformer."""

from schemashift.detection import detect_format
from schemashift.errors import (
    AmbiguousFormatError,
    ConfigValidationError,
    DSLRuntimeError,
    DSLSyntaxError,
    FormatDetectionError,
    LLMGenerationError,
    ReaderError,
    SchemaShiftError,
    SchemaValidationError,
    UnsupportedFileError,
)
from schemashift.models import ColumnMapping, FormatConfig, ReaderConfig
from schemashift.readers import read_file
from schemashift.registry import DictRegistry, FileSystemRegistry, Registry
from schemashift.target_schema import TargetSchema
from schemashift.transform import auto_transform, dry_run, transform, validate_config

__all__ = [
    "ColumnMapping",
    "FormatConfig",
    "ReaderConfig",
    "TargetSchema",
    "DictRegistry",
    "FileSystemRegistry",
    "Registry",
    "auto_transform",
    "dry_run",
    "transform",
    "validate_config",
    "detect_format",
    "read_file",
    "SchemaShiftError",
    "ConfigValidationError",
    "DSLSyntaxError",
    "DSLRuntimeError",
    "FormatDetectionError",
    "AmbiguousFormatError",
    "SchemaValidationError",
    "UnsupportedFileError",
    "LLMGenerationError",
    "ReaderError",
]
