"""Pydantic v2 models for schemashift configuration."""

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .dtypes import DTYPE_MAP
from .errors import ConfigValidationError

# Matches col("column_name") in DSL expressions.
_COL_PATTERN: re.Pattern[str] = re.compile(r'col\("([^"]+)"\)')

_DTYPE_JSON_SCHEMA: dict = {
    "anyOf": [
        {"enum": sorted(DTYPE_MAP), "type": "string"},
        {"type": "null"},
    ]
}


class ColumnMapping(BaseModel):
    """Describes how to produce one output column."""

    model_config = {"from_attributes": True}

    target: str = Field(description="Name of the output column produced by this mapping.")
    source: str | None = Field(
        default=None, description="Source column name in the input file. Mutually exclusive with 'expr' and 'constant'."
    )
    expr: str | None = Field(
        default=None,
        description=(
            'DSL expression to compute this column (e.g. col("price") * 1.2).'
            " Mutually exclusive with 'source' and 'constant'."
        ),
    )
    constant: Any = Field(
        default=None, description="Literal constant broadcast to all rows. Mutually exclusive with 'source' and 'expr'."
    )
    dtype: str | None = Field(
        default=None,
        description="Target Polars dtype to cast this column to after mapping.",
        json_schema_extra=_DTYPE_JSON_SCHEMA,
    )
    fillna: Any = Field(default=None, description="Fill-value applied to nulls after mapping.")

    @model_validator(mode="after")
    def _exactly_one_source_set(self) -> "ColumnMapping":
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
        if value is not None and value not in DTYPE_MAP:
            raise ConfigValidationError(f"Invalid dtype '{value}'. Valid values are: {sorted(DTYPE_MAP)}")
        return value


class ReaderConfig(BaseModel):
    """Low-level options passed to the file reader."""

    model_config = {"from_attributes": True}

    skip_rows: int = Field(default=0, description="Number of rows to skip before the header.", ge=0)
    sheet_name: str | int | None = Field(
        default=None, description="Excel sheet name (string) or 0-based index (int). Ignored for non-Excel files."
    )
    separator: str | None = Field(default=None, description="CSV field delimiter. Auto-detected when null.")
    encoding: str = Field(default="utf-8", description="Character encoding of the source file.")


class FormatConfig(BaseModel):
    """Top-level configuration for a single source format."""

    model_config = {"from_attributes": True}

    name: str = Field(description="Unique identifier for this format configuration.")
    description: str = Field(default="", description="Human-readable description of this format.")
    version: int = Field(default=1, description="Config version. Increment on breaking changes.", ge=1)
    target_schema: str | None = Field(default=None, description="Name of the TargetSchema to validate against.")
    reader: ReaderConfig = Field(default_factory=ReaderConfig, description="Low-level reader options.")
    columns: list[ColumnMapping] = Field(description="Ordered list of column mappings defining the output schema.")
    drop_unmapped: bool = Field(default=True, description="Drop source columns not referenced by any mapping.")

    @model_validator(mode="after")
    def _unique_target_names(self) -> "FormatConfig":
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
