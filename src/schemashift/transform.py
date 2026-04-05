"""Core transform engine: apply a FormatConfig to a file."""

from pathlib import Path

import polars as pl

from schemashift.dsl import parse_and_compile
from schemashift.dtypes import DTYPE_MAP
from schemashift.errors import DSLSyntaxError
from schemashift.models import ColumnMapping, FormatConfig
from schemashift.readers import read_file

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transform(
    path: str | Path,
    config: FormatConfig,
    n_rows: int | None = None,
) -> pl.DataFrame:
    """Apply a FormatConfig to a file and return the transformed data.

    Args:
        path: Path to the source file.
        config: FormatConfig describing how to map columns.
        n_rows: If given, collect only the first *n_rows* rows (useful for
            previewing or validating a config without reading the whole file).

    Returns:
        A :class:`polars.DataFrame` with the transformed columns.

    Raises:
        DSLRuntimeError: When a DSL expression fails to evaluate.
        ReaderError: When the file cannot be read.
    """
    lf = _transform(path, config)
    if n_rows is not None:
        return lf.limit(n_rows).collect()  # ty: ignore[return-value]
    return lf.collect()  # ty: ignore[return-value]


def _transform(
    path: str | Path,
    config: FormatConfig,
) -> pl.LazyFrame:
    """Internal: apply a FormatConfig and return a LazyFrame (not collected).

    Used by the CLI for streaming sinks and by orchestration functions that
    need to validate the schema lazily before collecting.
    """
    lf = read_file(path, config.reader)
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
        compiled = parse_and_compile(mapping.expr)
        expr = compiled.alias(mapping.target)
    elif mapping.has_constant():
        # constant (may be None, broadcasting nulls)
        expr = pl.lit(mapping.constant).alias(mapping.target)
    else:
        raise ValueError(f"ColumnMapping {mapping} has neither source nor expr nor constant")

    # Apply dtype casting
    if mapping.dtype is not None:
        dtype = DTYPE_MAP.get(mapping.dtype)
        if dtype is not None:
            expr = expr.cast(dtype)

    # Apply fill_null
    if mapping.has_fillna():
        expr = expr.fill_null(pl.lit(mapping.fillna))

    return expr
