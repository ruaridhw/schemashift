"""Shared dtype utilities: string-to-Polars-type mapping and conversion."""

from __future__ import annotations

from typing import Literal

import polars as pl

from .errors import SchemaValidationError

DType = Literal[
    "str",
    "string",
    "utf8",
    "int8",
    "int16",
    "int32",
    "int64",
    "integer",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "number",
    "bool",
    "boolean",
    "date",
    "datetime",
    "time",
    "duration",
    "binary",
    "categorical",
    "null",
]

DTYPE_MAP: dict[DType, type[pl.DataType]] = {
    # String
    "str": pl.Utf8,
    "string": pl.Utf8,
    "utf8": pl.Utf8,
    # Signed integers
    "int8": pl.Int8,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "integer": pl.Int64,  # JSON Schema alias
    # Unsigned integers
    "uint8": pl.UInt8,
    "uint16": pl.UInt16,
    "uint32": pl.UInt32,
    "uint64": pl.UInt64,
    # Floats
    "float32": pl.Float32,
    "float64": pl.Float64,
    "number": pl.Float64,  # JSON Schema alias
    # Boolean
    "bool": pl.Boolean,
    "boolean": pl.Boolean,
    # Temporal
    "date": pl.Date,
    "datetime": pl.Datetime,
    "time": pl.Time,
    "duration": pl.Duration,
    # Other
    "binary": pl.Binary,
    "categorical": pl.Categorical,
    "null": pl.Null,
}


def polars_dtype(type_str: DType) -> type[pl.DataType]:
    """Convert a dtype string (e.g. ``'float64'``) to the corresponding Polars DataType class.

    Raises:
        SchemaValidationError: if *type_str* is not a recognised dtype alias.
    """
    try:
        return DTYPE_MAP[type_str]
    except KeyError as exc:
        raise SchemaValidationError(f"Unknown type string '{type_str}'. Valid values: {sorted(DTYPE_MAP)}") from exc
