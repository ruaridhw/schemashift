"""Higher-level orchestration flows built on top of the core transform engine."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from schemashift.errors import FormatDetectionError, ReviewRejectedError
from schemashift.models import FormatConfig, ReaderConfig
from schemashift.readers import read_header
from schemashift.registry import Registry
from schemashift.transform import _transform, transform

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from schemashift.target_schema import TargetSchema


def auto_transform(
    path: str | Path,
    registry: Registry,
    reader_config: ReaderConfig | None = None,
) -> pl.DataFrame:
    """Auto-detect the format from the registry and transform the file.

    Args:
        path: Path to the source file.
        registry: Registry to search for a matching config.
        reader_config: Optional reader configuration used when reading the file
            header for format detection.

    Returns:
        A transformed :class:`polars.DataFrame`.

    Raises:
        FormatDetectionError: When no config matches the file's columns.
        AmbiguousFormatError: When multiple configs match.
    """
    config = _detect_config(path, registry, reader_config)
    if config is None:
        columns = read_header(path, reader_config)
        raise FormatDetectionError(
            f"No registered config matches the columns found in '{path}'. Columns present: {columns}"
        )
    return _transform(path, config).collect()  # ty: ignore[return-value]


def smart_transform(
    path: str | Path,
    registry: Registry,
    target_schema: "TargetSchema | None" = None,
    llm: "BaseChatModel | None" = None,
    review_fn: Callable[[FormatConfig, pl.DataFrame], FormatConfig | None] | None = None,
    auto_register: bool = False,
    example_configs: list[FormatConfig] | None = None,
    max_retries: int = 2,
    n_sample_rows: int = 15,
    reader_config: ReaderConfig | None = None,
) -> pl.DataFrame:
    """Full detect-or-generate flow.

    1. Try auto-detect from registry.
    2. If miss and LLM available: generate config.
    3. If review_fn provided: pass config + sample to reviewer.
    4. If auto_register: save to registry.
    5. Apply config; optionally validate against target_schema.

    Args:
        path: Source file path.
        registry: Registry to search and optionally register to.
        target_schema: Required for LLM generation and output validation.
        llm: LangChain BaseChatModel.
        review_fn: callback(config, sample_df) -> config | None. None = reject.
        auto_register: Register LLM-generated config automatically.
        example_configs: Example configs for LLM prompt.
        max_retries: Max LLM retries.
        n_sample_rows: Rows to sample for LLM.
        reader_config: Optional reader configuration forwarded to all file reads.

    Returns:
        Transformed :class:`polars.DataFrame`.

    Raises:
        FormatDetectionError: No match and no LLM.
        ValueError: LLM needed but target_schema not provided.
        LLMGenerationError: LLM fails after all retries.
        ReviewRejectedError: review_fn returned None.
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
    lf = _transform(path, config)
    if target_schema is not None:
        target_schema.validate_lazy(lf)
    df: pl.DataFrame = lf.collect()  # ty: ignore[assignment]
    if target_schema is not None:
        target_schema.validate_eager(df)
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_config(
    path: str | Path,
    registry: Registry,
    reader_config: ReaderConfig | None = None,
) -> FormatConfig | None:
    from schemashift.detection import detect_format  # noqa: PLC0415

    return detect_format(read_header(path, reader_config), registry)


def _resolve_config(
    path: str | Path,
    registry: Registry,
    target_schema: "TargetSchema | None",
    llm: "BaseChatModel | None",
    review_fn: Callable[[FormatConfig, pl.DataFrame], FormatConfig | None] | None,
    auto_register: bool,
    example_configs: list[FormatConfig] | None,
    max_retries: int,
    n_sample_rows: int,
    reader_config: ReaderConfig | None = None,
) -> FormatConfig:
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
        path=str(path),
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
    path: str | Path,
    config: FormatConfig,
    review_fn: Callable[[FormatConfig, pl.DataFrame], FormatConfig | None] | None,
) -> FormatConfig:
    if review_fn is None:
        return config

    sample_df = transform(path, config, n_rows=10)
    reviewed = review_fn(config, sample_df)
    if reviewed is None:
        raise ReviewRejectedError("Config was rejected by review_fn")
    return reviewed
