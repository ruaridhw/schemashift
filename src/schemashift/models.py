"""Pydantic v2 models for schemashift configuration."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .errors import ConfigValidationError

_VALID_DTYPES: frozenset[str] = frozenset(
    {
        "str",
        "utf8",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "date",
        "datetime",
        "time",
        "duration",
        "binary",
        "categorical",
        "null",
    }
)

# Matches col("column_name") in DSL expressions.
_COL_PATTERN: re.Pattern[str] = re.compile(r'col\("([^"]+)"\)')


class ColumnMapping(BaseModel):
    """Describes how to produce one output column."""

    model_config = {"from_attributes": True}

    target: str = Field(description="Output column name.")
    source: str | None = Field(default=None, description="Source column name to rename.")
    expr: str | None = Field(default=None, description="DSL expression string.")
    constant: Any = Field(default=None, description="Literal value broadcast to all rows.")
    dtype: str | None = Field(default=None, description="Optional Polars dtype cast applied after mapping.")
    fillna: Any = Field(default=None, description="Value used to fill nulls after mapping.")

    @model_validator(mode="after")
    def _exactly_one_source_set(self) -> ColumnMapping:
        set_fields = sum(
            [
                self.source is not None,
                self.expr is not None,
                self.constant is not None,
            ]
        )
        if set_fields != 1:
            raise ConfigValidationError(
                f"ColumnMapping '{self.target}': exactly one of 'source', 'expr', or "
                f"'constant' must be set, but {set_fields} were provided."
            )
        return self

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, value: str | None) -> str | None:
        if value is not None and value not in _VALID_DTYPES:
            raise ConfigValidationError(f"Invalid dtype '{value}'. Valid values are: {sorted(_VALID_DTYPES)}")
        return value


class ReaderConfig(BaseModel):
    """Low-level options passed to the file reader."""

    model_config = {"from_attributes": True}

    skip_rows: int = Field(default=0, description="Number of rows to skip before the header.")
    sheet_name: str | int | None = Field(default=None, description="Sheet name or 1-based index for Excel files.")
    separator: str | None = Field(default=None, description="Column separator for CSV files; None auto-detects.")
    encoding: str = Field(default="utf-8", description="File encoding.")


class FormatConfig(BaseModel):
    """Top-level configuration for a single source format."""

    model_config = {"from_attributes": True}

    name: str = Field(description="Unique config identifier.")
    description: str = Field(default="", description="Human-readable description of this format.")
    version: int = Field(default=1, description="Config schema version.")
    target_schema: str | None = Field(default=None, description="Name of the target schema this config maps to.")
    reader: ReaderConfig = Field(default_factory=ReaderConfig, description="Low-level reader options.")
    columns: list[ColumnMapping] = Field(description="Column mappings from source to target.")
    drop_unmapped: bool = Field(default=True, description="If True, drop source columns not listed in mappings.")

    @model_validator(mode="after")
    def _unique_target_names(self) -> FormatConfig:
        targets = [col.target for col in self.columns]
        seen: set[str] = set()
        duplicates: list[str] = []
        for t in targets:
            if t in seen:
                duplicates.append(t)
            seen.add(t)
        if duplicates:
            raise ConfigValidationError(
                f"FormatConfig '{self.name}': duplicate target column names: {sorted(set(duplicates))}"
            )
        return self

    def source_columns(self) -> set[str]:
        """Return all source column names referenced in this config.

        Includes direct 'source' fields and col("...") references inside 'expr' fields.
        """
        cols: set[str] = set()
        for mapping in self.columns:
            if mapping.source is not None:
                cols.add(mapping.source)
            if mapping.expr is not None:
                cols.update(_COL_PATTERN.findall(mapping.expr))
        return cols
