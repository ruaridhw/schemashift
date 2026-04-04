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
    ReviewRejectedError,
    SchemaShiftError,
    SchemaValidationError,
    UnsupportedFileError,
)
from schemashift.llm import load_default_llm
from schemashift.models import ColumnMapping, FormatConfig, ReaderConfig
from schemashift.orchestration import auto_transform, smart_transform
from schemashift.readers import read_file
from schemashift.registry import DictRegistry, FileSystemRegistry, Registry
from schemashift.schema import get_schema, get_schema_path
from schemashift.target_schema import TargetSchema
from schemashift.transform import dry_run, transform, validate_config

__all__ = [
    "AmbiguousFormatError",
    "ColumnMapping",
    "ConfigValidationError",
    "DSLRuntimeError",
    "DSLSyntaxError",
    "DictRegistry",
    "FileSystemRegistry",
    "FormatConfig",
    "FormatDetectionError",
    "LLMGenerationError",
    "ReaderConfig",
    "ReaderError",
    "Registry",
    "ReviewRejectedError",
    "SchemaShiftError",
    "SchemaValidationError",
    "TargetSchema",
    "UnsupportedFileError",
    "auto_transform",
    "detect_format",
    "dry_run",
    "get_schema",
    "get_schema_path",
    "load_default_llm",
    "read_file",
    "smart_transform",
    "transform",
    "validate_config",
]
