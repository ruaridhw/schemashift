"""Core transform engine: apply a FormatConfig to a file."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from schemashift.dsl import parse_and_compile
from schemashift.errors import DSLRuntimeError, DSLSyntaxError, FormatDetectionError
from schemashift.models import ColumnMapping, FormatConfig, ReaderConfig
from schemashift.readers import read_file
from schemashift.registry import Registry

# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

DTYPE_MAP: dict[str, pl.PolarsDataType] = {
    "str": pl.Utf8,
    "string": pl.Utf8,
    "utf8": pl.Utf8,
    "int8": pl.Int8,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "uint8": pl.UInt8,
    "uint16": pl.UInt16,
    "uint32": pl.UInt32,
    "uint64": pl.UInt64,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "bool": pl.Boolean,
    "boolean": pl.Boolean,
    "datetime": pl.Datetime,
    "date": pl.Date,
    "time": pl.Time,
    "duration": pl.Duration,
    "binary": pl.Binary,
    "categorical": pl.Categorical,
}


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

    lf = lf.select(expressions) if config.drop_unmapped else lf.with_columns(expressions)

    return lf


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
                errors.append(
                    f"Column '{mapping.target}': DSL syntax error in expr "
                    f"{mapping.expr!r}: {exc}"
                )
            except Exception as exc:
                errors.append(
                    f"Column '{mapping.target}': unexpected error parsing expr "
                    f"{mapping.expr!r}: {exc}"
                )

        if mapping.dtype is not None and mapping.dtype not in DTYPE_MAP:
            errors.append(
                f"Column '{mapping.target}': unknown dtype {mapping.dtype!r}. "
                f"Valid dtypes: {sorted(DTYPE_MAP)}"
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
    return lf.limit(n_rows).collect()


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
            f"No registered config matches the columns found in '{path}'. "
            f"Columns present: {columns}"
        )

    return transform(path, config)


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
    else:
        # constant
        expr = pl.lit(mapping.constant).alias(mapping.target)

    # Apply dtype casting
    if mapping.dtype is not None:
        dtype = DTYPE_MAP.get(mapping.dtype)
        if dtype is not None:
            expr = expr.cast(dtype)

    # Apply fill_null
    if mapping.fillna is not None:
        expr = expr.fill_null(pl.lit(mapping.fillna))

    return expr
