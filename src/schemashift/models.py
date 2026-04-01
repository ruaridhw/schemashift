"""Pydantic v2 models for schemashift configuration."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

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

    target: str
    source: str | None = None
    expr: str | None = None
    constant: Any = None
    dtype: str | None = None
    fillna: Any = None

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
            raise ConfigValidationError(
                f"Invalid dtype '{value}'. "
                f"Valid values are: {sorted(_VALID_DTYPES)}"
            )
        return value


class ReaderConfig(BaseModel):
    """Low-level options passed to the file reader."""

    model_config = {"from_attributes": True}

    skip_rows: int = 0
    sheet_name: str | int | None = None
    separator: str | None = None
    encoding: str = "utf-8"


class FormatConfig(BaseModel):
    """Top-level configuration for a single source format."""

    model_config = {"from_attributes": True}

    name: str
    description: str = ""
    version: int = 1
    reader: ReaderConfig = ReaderConfig()
    columns: list[ColumnMapping]
    drop_unmapped: bool = True

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
                f"FormatConfig '{self.name}': duplicate target column names: "
                f"{sorted(set(duplicates))}"
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
