"""Target schema definitions and validation helpers."""

from pathlib import Path

import polars as pl
import yaml
from pydantic import BaseModel, Field

from .dtypes import DTYPE_MAP, DType, polars_dtype
from .errors import SchemaValidationError


class TargetColumn(BaseModel):
    """Describes a single column in a target schema."""

    model_config = {"from_attributes": True}

    name: str = Field(description="Output column name.")
    type: DType = Field(description="Polars dtype string (e.g. 'int64', 'str', 'float64').")
    required: bool = Field(default=True, description="Whether this column must be present and non-null in output.")
    description: str = Field(default="", description="Human-readable description of the column's purpose.")


class TargetSchema(BaseModel):
    """Defines the expected shape and types of an output DataFrame."""

    model_config = {"from_attributes": True}

    name: str = Field(description="Schema identifier.")
    description: str = Field(default="", description="Human-readable description of the schema's purpose.")
    columns: list[TargetColumn] = Field(description="Ordered list of expected output columns.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TargetSchema":
        """Load a TargetSchema from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def required_columns(self) -> list[str]:
        """Return names of columns marked as required."""
        return [col.name for col in self.columns if col.required]

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
                if col.required:
                    errors.append(f"Missing column: '{col.name}'")
                continue
            expected_dtype = polars_dtype(col.type)
            actual_dtype = schema[col.name]
            # Compare base type class rather than instances to avoid parameter mismatches
            # (e.g. Datetime(time_unit=...) vs Datetime).
            if not _dtypes_compatible(actual_dtype, expected_dtype):
                errors.append(f"Column '{col.name}': expected dtype {expected_dtype}, got {actual_dtype}")

        if errors:
            raise SchemaValidationError("Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

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
                    errors.append(f"Required column '{col.name}' has {null_count} null value(s)")

        if errors:
            raise SchemaValidationError("Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def _dtypes_compatible(actual: pl.DataType, expected: type[pl.DataType]) -> bool:
    """Return True when *actual* satisfies *expected*.

    ``expected`` is a DataType class (e.g. ``pl.Int64``), while ``actual`` is
    a DataType instance returned from a LazyFrame schema.  Using ``isinstance``
    handles parameterised types like ``Datetime(time_unit='us')`` matching the
    bare ``Datetime`` class.
    """
    return isinstance(actual, expected)


__all__ = ["DTYPE_MAP", "DType", "polars_dtype", "TargetColumn", "TargetSchema"]
