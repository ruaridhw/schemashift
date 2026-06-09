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
from schemashift.models import ColumnMapping, ReaderConfig, TransformSpec
from schemashift.orchestration import smart_transform
from schemashift.readers import read_file
from schemashift.registry import DictRegistry, FileSystemRegistry, Registry
from schemashift.result import FailureInfo, TransformResult
from schemashift.schema import get_schema, get_schema_path
from schemashift.transform import transform, validate_config
from schemashift.validation import ColumnConstraints, SchemaConfig

__all__ = [
    "AmbiguousFormatError",
    "ColumnConstraints",
    "ColumnMapping",
    "ConfigValidationError",
    "DSLRuntimeError",
    "DSLSyntaxError",
    "DictRegistry",
    "FailureInfo",
    "FileSystemRegistry",
    "FormatDetectionError",
    "LLMGenerationError",
    "ReaderConfig",
    "ReaderError",
    "Registry",
    "ReviewRejectedError",
    "SchemaConfig",
    "SchemaShiftError",
    "SchemaValidationError",
    "TransformResult",
    "TransformSpec",
    "UnsupportedFileError",
    "detect_format",
    "get_schema",
    "get_schema_path",
    "load_default_llm",
    "read_file",
    "smart_transform",
    "transform",
    "validate_config",
]
