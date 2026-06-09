"""Schema validation via dataframely — config-driven dy.Schema construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dataframely as dy
import yaml
from pydantic import BaseModel, Field, model_validator

from .dtypes import DType  # noqa: TC001  # runtime import needed for Pydantic field type

# ---------------------------------------------------------------------------
# Mapping from DType strings to dataframely column classes
# ---------------------------------------------------------------------------

_DY_COLUMN_MAP: dict[str, type[dy.Column]] = {
    # String
    "str": dy.String,
    "string": dy.String,
    "utf8": dy.String,
    # Signed integers
    "int8": dy.Int8,
    "int16": dy.Int16,
    "int32": dy.Int32,
    "int64": dy.Int64,
    "integer": dy.Int64,
    # Unsigned integers
    "uint8": dy.UInt8,
    "uint16": dy.UInt16,
    "uint32": dy.UInt32,
    "uint64": dy.UInt64,
    # Floats
    "float32": dy.Float32,
    "float64": dy.Float64,
    "number": dy.Float64,
    # Boolean
    "bool": dy.Bool,
    "boolean": dy.Bool,
    # Temporal
    "date": dy.Date,
    "datetime": dy.Datetime,
    "time": dy.Time,
    "duration": dy.Duration,
    # Other
    "binary": dy.Binary,
    "categorical": dy.Categorical,
}

# Parameters accepted by each dy column class family.
# Used to filter ColumnConstraints fields before passing to the constructor.
_INTEGER_TYPES = frozenset({"int8", "int16", "int32", "int64", "integer", "uint8", "uint16", "uint32", "uint64"})
_FLOAT_TYPES = frozenset({"float32", "float64", "number"})
_STRING_TYPES = frozenset({"str", "string", "utf8"})
_TEMPORAL_TYPES = frozenset({"date", "datetime", "time", "duration"})

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ColumnConstraints(BaseModel):
    """Per-column schema definition mapping to a dataframely column type + constraints."""

    model_config = {"from_attributes": True}

    type: DType = Field(description="Polars dtype string (e.g. 'int64', 'str', 'float64').")
    nullable: bool = Field(default=False, description="Whether null values are allowed.")
    primary_key: bool = Field(default=False, description="Whether this column is a primary key.")
    description: str = Field(default="", description="Human-readable description.")

    # Numeric constraints (int/float/temporal)
    min: Any = Field(default=None, description="Minimum value (inclusive).")
    max: Any = Field(default=None, description="Maximum value (inclusive).")
    min_exclusive: Any = Field(default=None, description="Minimum value (exclusive).")
    max_exclusive: Any = Field(default=None, description="Maximum value (exclusive).")

    # Integer-only
    is_in: list[Any] | None = Field(
        default=None, description="Allowed values (integers only natively; strings use check)."
    )

    # String constraints
    min_length: int | None = Field(default=None, description="Minimum string length.", ge=0)
    max_length: int | None = Field(default=None, description="Maximum string length.", ge=0)
    regex: str | None = Field(default=None, description="Regex pattern the string must match.")

    @model_validator(mode="after")
    def _validate_constraints_for_type(self) -> ColumnConstraints:
        """Ensure constraints are valid for the column type."""
        t = self.type
        if t in _STRING_TYPES:
            for field in ("min", "max", "min_exclusive", "max_exclusive"):
                if getattr(self, field) is not None:
                    raise ValueError(f"Constraint '{field}' is not valid for string type '{t}'")
        if t not in _STRING_TYPES:
            for field in ("min_length", "max_length", "regex"):
                if getattr(self, field) is not None:
                    raise ValueError(f"Constraint '{field}' is only valid for string types, not '{t}'")
        if self.is_in is not None and t not in _INTEGER_TYPES:
            # is_in is natively supported only on integer types in dy;
            # for non-integer types we use the check mechanism, which is fine
            pass
        return self


class SchemaConfig(BaseModel):
    """Declarative schema definition loadable from YAML/JSON."""

    model_config = {"from_attributes": True}

    name: str = Field(description="Schema identifier.")
    description: str = Field(default="", description="Human-readable description.")
    columns: dict[str, ColumnConstraints] = Field(description="Column name → constraints mapping.")

    @classmethod
    def from_yaml(cls, path: Path) -> SchemaConfig:
        """Load a SchemaConfig from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Factory: SchemaConfig → dy.Schema class
# ---------------------------------------------------------------------------


def _add_range_kwargs(kwargs: dict[str, Any], constraints: ColumnConstraints) -> None:
    """Add min/max/min_exclusive/max_exclusive kwargs if set."""
    for field in ("min", "max", "min_exclusive", "max_exclusive"):
        value = getattr(constraints, field)
        if value is not None:
            kwargs[field] = value


def _add_is_in_check(kwargs: dict[str, Any], constraints: ColumnConstraints) -> None:
    """Add is_in as a check callable (for types that don't support it natively)."""
    if constraints.is_in is not None:
        allowed = constraints.is_in
        kwargs["check"] = {"is_in": lambda expr, _a=allowed: expr.is_in(_a)}


def _build_column(name: str, constraints: ColumnConstraints) -> dy.Column:
    """Build a dataframely Column instance from ColumnConstraints."""
    dy_cls = _DY_COLUMN_MAP.get(constraints.type)
    if dy_cls is None:
        raise ValueError(f"No dataframely column type for dtype '{constraints.type}'")

    kwargs: dict[str, Any] = {
        "nullable": constraints.nullable,
        "primary_key": constraints.primary_key,
    }

    t = constraints.type

    if t in _STRING_TYPES:
        if constraints.min_length is not None:
            kwargs["min_length"] = constraints.min_length
        if constraints.max_length is not None:
            kwargs["max_length"] = constraints.max_length
        if constraints.regex is not None:
            kwargs["regex"] = constraints.regex
        _add_is_in_check(kwargs, constraints)
    elif t in _INTEGER_TYPES:
        _add_range_kwargs(kwargs, constraints)
        if constraints.is_in is not None:
            kwargs["is_in"] = constraints.is_in
    elif t in _FLOAT_TYPES:
        _add_range_kwargs(kwargs, constraints)
        _add_is_in_check(kwargs, constraints)
    elif t in _TEMPORAL_TYPES:
        _add_range_kwargs(kwargs, constraints)

    return dy_cls(**kwargs)


def build_dy_schema(config: SchemaConfig) -> type[dy.Schema]:
    """Dynamically construct a dy.Schema subclass from a SchemaConfig."""
    attrs: dict[str, Any] = {}
    for col_name, constraints in config.columns.items():
        attrs[col_name] = _build_column(col_name, constraints)
    return type(config.name, (dy.Schema,), attrs)


def resolve_schema(schema: SchemaConfig | type[dy.Schema]) -> type[dy.Schema]:
    """Normalise a schema argument to a dy.Schema class.

    Accepts either a SchemaConfig (built into a dy.Schema) or a dy.Schema class directly.
    """
    if isinstance(schema, SchemaConfig):
        return build_dy_schema(schema)
    if isinstance(schema, type) and issubclass(schema, dy.Schema):
        return schema
    raise TypeError(f"Expected SchemaConfig or dy.Schema subclass, got {type(schema)}")


__all__ = [
    "ColumnConstraints",
    "SchemaConfig",
    "build_dy_schema",
    "resolve_schema",
]
