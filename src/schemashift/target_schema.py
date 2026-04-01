"""Target schema definitions and validation helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml
from pydantic import BaseModel

from .errors import SchemaValidationError

_DTYPE_MAP: dict[str, pl.DataType] = {
    "str": pl.Utf8,
    "utf8": pl.Utf8,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "int8": pl.Int8,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "uint8": pl.UInt8,
    "uint16": pl.UInt16,
    "uint32": pl.UInt32,
    "uint64": pl.UInt64,
    "bool": pl.Boolean,
    "date": pl.Date,
    "datetime": pl.Datetime,
    "time": pl.Time,
    "duration": pl.Duration,
    "binary": pl.Binary,
    "categorical": pl.Categorical,
}


class TargetColumn(BaseModel):
    """Describes a single column in a target schema."""

    model_config = {"from_attributes": True}

    name: str
    type: str
    required: bool = True
    description: str = ""


class TargetSchema(BaseModel):
    """Defines the expected shape and types of an output DataFrame."""

    model_config = {"from_attributes": True}

    name: str
    description: str = ""
    columns: list[TargetColumn]

    @classmethod
    def from_yaml(cls, path: str | Path) -> TargetSchema:
        """Load a TargetSchema from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def required_columns(self) -> list[str]:
        """Return names of columns marked as required."""
        return [col.name for col in self.columns if col.required]

    def polars_dtype(self, type_str: str) -> pl.DataType:
        """Convert a type string like 'float64' to the corresponding Polars DataType.

        Raises:
            SchemaValidationError: if the type string is not recognised.
        """
        try:
            return _DTYPE_MAP[type_str]
        except KeyError as exc:
            raise SchemaValidationError(
                f"Unknown type string '{type_str}'. "
                f"Valid values: {sorted(_DTYPE_MAP)}"
            ) from exc

    def validate_lazy(self, lf: pl.LazyFrame) -> None:
        """Validate column names and dtypes against the LazyFrame schema.

        Does not collect data — checks structural metadata only.

        Raises:
            SchemaValidationError: listing all missing columns and type mismatches.
        """
        schema = lf.collect_schema()
        actual_names = set(schema.names())
        errors: list[str] = []

        for col in self.columns:
            if col.name not in actual_names:
                errors.append(f"Missing column: '{col.name}'")
                continue
            expected_dtype = self.polars_dtype(col.type)
            actual_dtype = schema[col.name]
            # Compare base type class rather than instances to avoid parameter mismatches
            # (e.g. Datetime(time_unit=...) vs Datetime).
            if not _dtypes_compatible(actual_dtype, expected_dtype):
                errors.append(
                    f"Column '{col.name}': expected dtype {expected_dtype}, "
                    f"got {actual_dtype}"
                )

        if errors:
            raise SchemaValidationError(
                "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def validate_eager(self, df: pl.DataFrame) -> None:
        """Full validation: columns, dtypes, and null checks on required columns.

        Raises:
            SchemaValidationError: with details of all issues found.
        """
        self.validate_lazy(df.lazy())

        errors: list[str] = []
        for col in self.columns:
            if col.required and col.name in df.columns:
                null_count = df[col.name].null_count()
                if null_count > 0:
                    errors.append(
                        f"Required column '{col.name}' has {null_count} null value(s)"
                    )

        if errors:
            raise SchemaValidationError(
                "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )


def _dtypes_compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    """Return True when *actual* satisfies *expected*.

    ``expected`` is a DataType class (e.g. ``pl.Int64``), while ``actual`` is
    a DataType instance returned from a LazyFrame schema.  Using ``isinstance``
    handles both exact matches and parameterised types like
    ``Datetime(time_unit='us')`` comparing equal to the bare ``Datetime`` class.
    """
    # expected may be a class (DataTypeClass) — isinstance works for both.
    return isinstance(actual, expected)  # type: ignore[arg-type]
