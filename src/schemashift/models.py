"""Pydantic v2 models for schemashift configuration."""

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .dsl import collect_col_refs, parse_dsl
from .dtypes import DTYPE_MAP, DType
from .errors import ConfigValidationError

# Sentinel for optional Any-typed fields where None is a valid user-supplied value.
# Using PydanticUndefined as a default makes Pydantic treat the field as required,
# so we define our own sentinel object instead.
_UNSET: Any = object()


_DTYPE_JSON_SCHEMA: dict[str, Any] = {
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
        default=_UNSET,
        description="Literal constant broadcast to all rows. Mutually exclusive with 'source' and 'expr'.",
    )
    dtype: DType | None = Field(
        default=None,
        description="Target Polars dtype to cast this column to after mapping.",
        json_schema_extra=_DTYPE_JSON_SCHEMA,
    )
    fillna: Any = Field(default=_UNSET, description="Fill-value applied to nulls after mapping.")

    def has_constant(self) -> bool:
        return self.constant is not _UNSET

    def has_fillna(self) -> bool:
        return self.fillna is not _UNSET

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to omit sentinel-valued fields from the output dict."""
        d = super().model_dump(**kwargs)
        if d.get("constant") is _UNSET:
            d.pop("constant", None)
        if d.get("fillna") is _UNSET:
            d.pop("fillna", None)
        return d

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override to omit sentinel-valued fields from the JSON output."""
        return json.dumps(self.model_dump(**kwargs))

    @model_validator(mode="after")
    def _exactly_one_source_set(self) -> "ColumnMapping":
        set_fields = sum(
            [
                self.source is not None,
                self.expr is not None,
                self.has_constant(),
            ]
        )
        if set_fields != 1:
            raise ConfigValidationError(
                f"ColumnMapping '{self.target}': exactly one of 'source', 'expr', or "
                f"'constant' must be set, but {set_fields} were provided."
            )
        return self


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

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to serialise ColumnMapping fields respecting the _UNSET sentinel."""
        d = super().model_dump(**kwargs)
        # Replace raw column dicts with ones produced by ColumnMapping.model_dump()
        d["columns"] = [col.model_dump(**kwargs) for col in self.columns]
        return d

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override to serialise ColumnMapping fields respecting the _UNSET sentinel."""
        indent = kwargs.pop("indent", None)
        return json.dumps(self.model_dump(**kwargs), indent=indent)

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

        Includes direct 'source' fields and all col("...") references inside 'expr' fields,
        extracted by walking the DSL AST rather than regex-matching strings.
        """
        cols: set[str] = set()
        for mapping in self.columns:
            if mapping.source is not None:
                cols.add(mapping.source)
            if mapping.expr is not None:
                cols.update(collect_col_refs(parse_dsl(mapping.expr)))
        return cols
