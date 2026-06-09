"""Higher-level orchestration flows built on top of the core transform engine."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import dataframely as dy
import polars as pl

from schemashift.errors import FormatDetectionError, ReviewRejectedError
from schemashift.models import ReaderConfig, TransformSpec
from schemashift.readers import read_header
from schemashift.registry import Registry
from schemashift.result import TransformResult
from schemashift.transform import transform
from schemashift.validation import SchemaConfig

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def smart_transform(
    path: Path,
    registry: Registry,
    target_schema: SchemaConfig | None = None,
    schema: SchemaConfig | type[dy.Schema] | None = None,
    llm: "BaseChatModel | None" = None,
    review_fn: Callable[[TransformSpec, pl.DataFrame], TransformSpec | None] | None = None,
    auto_register: bool = False,
    example_configs: list[TransformSpec] | None = None,
    max_retries: int = 2,
    n_sample_rows: int = 15,
    reader_config: ReaderConfig | None = None,
    *,
    strict: bool = False,
) -> TransformResult:
    """Full detect-or-generate flow.

    1. Try auto-detect from registry.
    2. If miss and LLM available: generate config.
    3. If review_fn provided: pass config + sample to reviewer.
    4. If auto_register: save to registry.
    5. Apply config and validate against schema.

    Args:
        path: Source file path.
        registry: Registry to search and optionally register to.
        target_schema: Deprecated. Use ``schema`` instead.
        schema: Output schema for validation.
        llm: LangChain BaseChatModel.
        review_fn: callback(config, sample_df) -> config | None. None = reject.
        auto_register: Register LLM-generated config automatically.
        example_configs: Example configs for LLM prompt.
        max_retries: Max LLM retries.
        n_sample_rows: Rows to sample for LLM.
        reader_config: Optional reader configuration forwarded to all file reads.
        strict: If True, raise on validation failures.

    Returns:
        A :class:`TransformResult` with valid rows and failure details.

    Raises:
        FormatDetectionError: No match and no LLM.
        ValueError: LLM needed but target_schema not provided.
        LLMGenerationError: LLM fails after all retries.
        ReviewRejectedError: review_fn returned None.
        SchemaValidationError: When *strict* is True and validation fails.
    """
    config = _resolve_config(
        path=path,
        registry=registry,
        target_schema=target_schema,
        llm=llm,
        review_fn=review_fn,
        auto_register=auto_register,
        example_configs=example_configs,
        max_retries=max_retries,
        n_sample_rows=n_sample_rows,
        reader_config=reader_config,
    )
    return transform(path, config, schema=schema, strict=strict)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_config(
    path: Path,
    registry: Registry,
    reader_config: ReaderConfig | None = None,
) -> TransformSpec | None:
    from schemashift.detection import detect_format  # noqa: PLC0415

    return detect_format(read_header(path, reader_config), registry)


def _resolve_config(
    path: Path,
    registry: Registry,
    target_schema: SchemaConfig | None,
    llm: "BaseChatModel | None",
    review_fn: Callable[[TransformSpec, pl.DataFrame], TransformSpec | None] | None,
    auto_register: bool,
    example_configs: list[TransformSpec] | None,
    max_retries: int,
    n_sample_rows: int,
    reader_config: ReaderConfig | None = None,
) -> TransformSpec:
    config = _detect_config(path, registry, reader_config)
    if config is not None:
        return config

    columns = read_header(path, reader_config)
    if llm is None:
        raise FormatDetectionError(
            f"No registered config matches '{path}' and no LLM is configured. File columns: {columns}"
        )
    if target_schema is None:
        raise ValueError("target_schema is required for LLM config generation")

    from schemashift.llm import generate_config  # noqa: PLC0415

    generated = generate_config(
        path=path,
        target_schema=target_schema,
        llm=llm,
        example_configs=example_configs,
        max_retries=max_retries,
        n_sample_rows=n_sample_rows,
        reader_config=reader_config,
    )
    reviewed = _review_generated_config(path, generated, review_fn)
    if auto_register:
        registry.register(reviewed)
    return reviewed


def _review_generated_config(
    path: Path,
    config: TransformSpec,
    review_fn: Callable[[TransformSpec, pl.DataFrame], TransformSpec | None] | None,
) -> TransformSpec:
    if review_fn is None:
        return config

    result = transform(path, config, n_rows=10)
    reviewed = review_fn(config, result.valid)
    if reviewed is None:
        raise ReviewRejectedError("Config was rejected by review_fn")
    return reviewed
