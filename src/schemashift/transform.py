"""Core transform engine: apply a TransformSpec to a file."""

import logging
from pathlib import Path

import dataframely as dy
import polars as pl

from schemashift.dsl import parse_and_compile
from schemashift.dtypes import DTYPE_MAP
from schemashift.errors import DSLSyntaxError, SchemaValidationError
from schemashift.models import ColumnMapping, TransformSpec
from schemashift.readers import read_file
from schemashift.result import FailureInfo, TransformResult
from schemashift.validation import SchemaConfig, resolve_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transform(
    path: Path,
    config: TransformSpec,
    schema: SchemaConfig | type[dy.Schema] | None = None,
    *,
    strict: bool = False,
    n_rows: int | None = None,
) -> TransformResult:
    """Apply a TransformSpec to a file and return transformed + validated data.

    Args:
        path: Path to the source file.
        config: TransformSpec describing how to map columns.
        schema: Override output schema. Falls back to ``config.output_schema``.
        strict: If True, raise :class:`SchemaValidationError` when any validation
            failures exist.
        n_rows: If given, collect only the first *n_rows* rows (useful for
            previewing or validating a config without reading the whole file).

    Returns:
        A :class:`TransformResult` with valid rows and failure details.

    Raises:
        SchemaValidationError: When *strict* is True and validation fails.
        DSLRuntimeError: When a DSL expression fails to evaluate.
        ReaderError: When the file cannot be read.
    """
    # --- Build and apply expressions ---
    lf = read_file(path, config.reader)
    expressions, expression_errors = _build_expressions_lenient(config)

    lf = lf.select(expressions) if config.drop_unmapped else lf.with_columns(expressions)

    if n_rows is not None:
        lf = lf.limit(n_rows)

    df: pl.DataFrame = lf.collect()  # ty: ignore[invalid-assignment]

    # --- Resolve schema and validate ---
    resolved_schema = schema or config.output_schema
    if resolved_schema is not None:
        dy_schema = resolve_schema(resolved_schema)
        filter_result = dy_schema.filter(df)
        valid_df = filter_result.result
        schema_failures = filter_result.failure
    else:
        valid_df = df
        schema_failures = None

    failures = FailureInfo(
        schema_failures=schema_failures,
        expression_errors=expression_errors,
    )
    result = TransformResult(valid=valid_df, failures=failures)

    if strict and result.failures.has_failures:
        raise SchemaValidationError(
            _format_failure_message(failures),
            failures=failures,
        )

    return result


def _transform(
    path: Path,
    config: TransformSpec,
) -> pl.LazyFrame:
    """Internal: apply a TransformSpec and return a LazyFrame (not collected).

    Used by the CLI for streaming sinks and by orchestration functions that
    need to validate the schema lazily before collecting.
    """
    lf = read_file(path, config.reader)
    expressions = _build_expressions(config)

    if config.drop_unmapped:
        return lf.select(expressions)
    return lf.with_columns(expressions)


def validate_config(config: TransformSpec) -> list[str]:
    """Validate a TransformSpec by parsing all DSL expressions and checking dtypes.

    Args:
        config: The TransformSpec to validate.

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


def _build_expressions(config: TransformSpec) -> list[pl.Expr]:
    """Build a list of Polars expressions for the given config's column mappings."""
    return [_mapping_to_expr(mapping) for mapping in config.columns]


def _build_expressions_lenient(config: TransformSpec) -> tuple[list[pl.Expr], dict[str, str]]:
    """Build expressions with lenient error handling.

    Returns a tuple of (expressions, expression_errors). Failed expressions
    produce a null-literal column so the pipeline can continue and collect
    all errors at once.
    """
    expressions: list[pl.Expr] = []
    expression_errors: dict[str, str] = {}

    for mapping in config.columns:
        try:
            expr = _mapping_to_expr(mapping, strict_cast=False)
            expressions.append(expr)
        except Exception as exc:
            logger.warning("Expression error for column '%s': %s", mapping.target, exc)
            expression_errors[mapping.target] = str(exc)
            expressions.append(pl.lit(None).alias(mapping.target))

    return expressions, expression_errors


def _mapping_to_expr(mapping: ColumnMapping, *, strict_cast: bool = True) -> pl.Expr:
    """Convert a single ColumnMapping to a Polars expression.

    Args:
        mapping: The column mapping to convert.
        strict_cast: If False, use non-strict casting (partial failures become null).
    """
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
            expr = expr.cast(dtype, strict=strict_cast)

    # Apply fill_null
    if mapping.has_fillna():
        expr = expr.fill_null(pl.lit(mapping.fillna))

    return expr


def _format_failure_message(failures: FailureInfo) -> str:
    """Format a human-readable message from FailureInfo for SchemaValidationError."""
    parts: list[str] = ["Schema validation failed:"]
    for col, msg in failures.expression_errors.items():
        parts.append(f"  - Expression error in column '{col}': {msg}")
    for rule, count in (failures.counts or {}).items():
        if not rule.startswith("expression_error:"):
            parts.append(f"  - Rule '{rule}' failed for {count} row(s)")
    return "\n".join(parts)
