"""Transform result types with unified failure reporting."""

from __future__ import annotations

from dataclasses import dataclass, field

import dataframely as dy  # noqa: TC002  # used at runtime in property methods
import polars as pl  # noqa: TC002  # used at runtime in property methods


@dataclass(frozen=True)
class FailureInfo:
    """Unified row-level failure details from transform + validation.

    Combines schema constraint violations (from dataframely) with
    DSL expression errors that prevented column evaluation entirely.
    """

    schema_failures: dy.FailureInfo | None = None
    """Row-level constraint violations from dy.Schema.filter()."""

    expression_errors: dict[str, str] = field(default_factory=dict)
    """DSL expressions that failed entirely: {column_name: error_message}."""

    @property
    def invalid(self) -> pl.DataFrame | None:
        """Original invalid rows (without rule columns)."""
        if self.schema_failures is not None:
            return self.schema_failures.invalid()
        return None

    @property
    def details(self) -> pl.DataFrame | None:
        """Invalid rows with per-rule valid/invalid/unknown status."""
        if self.schema_failures is not None:
            return self.schema_failures.details()
        return None

    @property
    def counts(self) -> dict[str, int]:
        """Failure counts per rule/error.

        Schema rule failures use dy's naming convention (e.g. ``"amount|min"``).
        Expression errors are keyed as ``"expression_error:{column}"``.
        """
        result: dict[str, int] = {}
        if self.schema_failures is not None:
            result.update(self.schema_failures.counts())
        for col in self.expression_errors:
            result[f"expression_error:{col}"] = -1  # all rows affected
        return result

    @property
    def has_failures(self) -> bool:
        """True if any expression errors or schema violations exist."""
        if self.expression_errors:
            return True
        if self.schema_failures is not None:
            invalid = self.schema_failures.invalid()
            return invalid.height > 0
        return False


@dataclass(frozen=True)
class TransformResult:
    """Result of a transform + validation pipeline.

    Always contains the valid rows and failure details. When no schema
    violations or expression errors exist, ``failures.has_failures`` is False.
    """

    valid: pl.DataFrame
    """Rows that passed all validation rules."""

    failures: FailureInfo
    """All failure details (always present, check ``has_failures``)."""

    @property
    def all_valid(self) -> bool:
        """True if no failures of any kind."""
        return not self.failures.has_failures


__all__ = [
    "FailureInfo",
    "TransformResult",
]
