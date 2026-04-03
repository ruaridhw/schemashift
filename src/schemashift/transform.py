"""Core transform engine: apply a FormatConfig to a file."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from schemashift.dsl import parse_and_compile
from schemashift.dtypes import DTYPE_MAP
from schemashift.errors import DSLRuntimeError, DSLSyntaxError, FormatDetectionError
from schemashift.models import ColumnMapping, FormatConfig, ReaderConfig
from schemashift.readers import read_file
from schemashift.registry import Registry

if TYPE_CHECKING:
    from schemashift.target_schema import TargetSchema


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transform(
    path: str | Path,
    config: FormatConfig,
    reader_config: ReaderConfig | None = None,
) -> pl.LazyFrame:
    """Apply a FormatConfig to a file and return a transformed LazyFrame.

    Steps:
    1. Read the file using :func:`~schemashift.readers.read_file`.
    2. Build one Polars expression per ColumnMapping.
    3. Optionally drop columns not referenced by any mapping.
    4. Apply dtype casting where specified.
    5. Apply fill_null where specified.

    Args:
        path: Path to the source file.
        config: FormatConfig describing how to map columns.
        reader_config: Optional low-level reader options (overrides
            ``config.reader`` when provided).

    Returns:
        A :class:`polars.LazyFrame` with the transformed columns.

    Raises:
        DSLRuntimeError: When a DSL expression fails to evaluate.
        ReaderError: When the file cannot be read.
    """
    effective_reader = reader_config if reader_config is not None else config.reader
    lf = read_file(path, effective_reader)
    expressions = _build_expressions(config)

    if config.drop_unmapped:
        return lf.select(expressions)
    return lf.with_columns(expressions)


def validate_config(config: FormatConfig) -> list[str]:
    """Validate a FormatConfig by parsing all DSL expressions and checking dtypes.

    Args:
        config: The FormatConfig to validate.

    Returns:
        A list of error message strings. An empty list means the config is valid.
    """
    errors: list[str] = []

    for mapping in config.columns:
        if mapping.expr is not None:
            try:
                parse_and_compile(mapping.expr)
            except DSLSyntaxError as exc:
                errors.append(f"Column '{mapping.target}': DSL syntax error in expr {mapping.expr!r}: {exc}")
            except Exception as exc:
                errors.append(f"Column '{mapping.target}': unexpected error parsing expr {mapping.expr!r}: {exc}")

        if mapping.dtype is not None and mapping.dtype not in DTYPE_MAP:
            errors.append(
                f"Column '{mapping.target}': unknown dtype {mapping.dtype!r}. Valid dtypes: {sorted(DTYPE_MAP)}"
            )

    return errors


def dry_run(config: FormatConfig, path: str | Path, n_rows: int = 10) -> pl.DataFrame:
    """Apply config to the first *n_rows* of a file and return the result.

    Args:
        config: FormatConfig to apply.
        path: Path to the sample file.
        n_rows: Number of rows to collect.

    Returns:
        A :class:`polars.DataFrame` containing the transformed rows.

    Raises:
        Any error raised by :func:`transform` or Polars collection.
    """
    lf = transform(path, config)
    result: pl.DataFrame = lf.limit(n_rows).collect()  # ty: ignore[invalid-assignment]
    return result


def auto_transform(
    path: str | Path,
    registry: Registry,
) -> pl.LazyFrame:
    """Auto-detect the format from the registry and transform the file.

    Args:
        path: Path to the source file.
        registry: Registry to search for a matching config.

    Returns:
        A transformed :class:`polars.LazyFrame`.

    Raises:
        FormatDetectionError: When no config matches the file's columns.
        AmbiguousFormatError: When multiple configs match.
    """
    from schemashift.detection import detect_format
    from schemashift.readers import read_header

    columns = read_header(path)
    config = detect_format(columns, registry)

    if config is None:
        raise FormatDetectionError(
            f"No registered config matches the columns found in '{path}'. Columns present: {columns}"
        )

    return transform(path, config)


def smart_transform(
    path: str | Path,
    registry: Registry,
    target_schema: "TargetSchema | None" = None,
    llm: Any = None,  # ANNOT: the typing here should be stronger
    review_fn: Callable[[FormatConfig, pl.DataFrame], FormatConfig | None] | None = None,
    auto_register: bool = False,
    example_configs: list[FormatConfig] | None = None,
    max_retries: int = 2,
    n_sample_rows: int = 15,
) -> pl.LazyFrame:
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

    Returns:
        Transformed pl.LazyFrame.

    Raises:
        FormatDetectionError: No match and no LLM, or review_fn rejected.
        ValueError: LLM needed but target_schema not provided.
        LLMGenerationError: LLM fails after all retries.
    """
    from schemashift.detection import detect_format
    from schemashift.readers import read_header

    # Try registry
    columns = read_header(path)
    config = detect_format(columns, registry)

    if config is not None:
        lf = transform(path, config)
        if target_schema is not None:
            target_schema.validate_lazy(lf)
        return lf

    # No match — need LLM
    if llm is None:
        raise FormatDetectionError(
            f"No registered config matches '{path}' and no LLM is configured. File columns: {columns}"
        )
    if target_schema is None:
        raise ValueError("target_schema is required for LLM config generation")

    from schemashift.llm import generate_config as _llm_generate

    config = _llm_generate(
        path=str(path),
        target_schema=target_schema,
        llm=llm,
        example_configs=example_configs,
        max_retries=max_retries,
        n_sample_rows=n_sample_rows,
    )

    # Review callback
    if review_fn is not None:
        sample_df = dry_run(config, path, n_rows=10)
        reviewed = review_fn(config, sample_df)
        if reviewed is None:
            raise FormatDetectionError("Config was rejected by review_fn")
        config = reviewed

    if auto_register:
        registry.register(config)

    lf = transform(path, config)
    if target_schema is not None:
        target_schema.validate_lazy(lf)
    return lf


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_expressions(config: FormatConfig) -> list[pl.Expr]:
    """Build a list of Polars expressions for the given config's column mappings."""
    expressions: list[pl.Expr] = []

    for mapping in config.columns:
        expr = _mapping_to_expr(mapping)
        expressions.append(expr)

    return expressions


def _mapping_to_expr(mapping: ColumnMapping) -> pl.Expr:
    """Convert a single ColumnMapping to a Polars expression."""
    if mapping.source is not None:
        expr = pl.col(mapping.source).alias(mapping.target)
    elif mapping.expr is not None:
        try:
            compiled = parse_and_compile(mapping.expr)
        except DSLSyntaxError as exc:
            raise DSLRuntimeError(
                f"DSL syntax error for target '{mapping.target}': {exc}",
                expression=mapping.expr,
                target=mapping.target,
            ) from exc
        expr = compiled.alias(mapping.target)
    elif mapping.constant is not None:
        # constant
        expr = pl.lit(mapping.constant).alias(mapping.target)
    else:
        raise ValueError(f"ColumnMapping {mapping} has neither source nor expr nor constant")

    # Apply dtype casting
    if mapping.dtype is not None:
        dtype = DTYPE_MAP.get(mapping.dtype)
        if dtype is not None:
            expr = expr.cast(dtype)

    # Apply fill_null
    if mapping.fillna is not None:
        expr = expr.fill_null(pl.lit(mapping.fillna))

    return expr
