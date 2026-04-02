"""Tests for schemashift.target_schema."""

from pathlib import Path

import polars as pl
import pytest

from schemashift.errors import SchemaValidationError
from schemashift.target_schema import TargetColumn, TargetSchema

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_SCHEMA_YAML = FIXTURES_DIR / "test_schema.yaml"


@pytest.fixture()
def test_schema() -> TargetSchema:
    return TargetSchema.from_yaml(TEST_SCHEMA_YAML)


@pytest.fixture()
def matching_df() -> pl.DataFrame:
    """DataFrame whose schema exactly matches test_schema.yaml."""
    return pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.Int64),
            "name": pl.Series(["Alice", "Bob", "Carol"], dtype=pl.Utf8),
            "amount": pl.Series([1.0, 2.5, 3.0], dtype=pl.Float64),
            "category": pl.Series(["food", "transport", "food"], dtype=pl.Utf8),
        }
    )


class TestTargetSchemaFromYaml:
    def test_loads_name(self, test_schema):
        assert test_schema.name == "test_schema"

    def test_loads_description(self, test_schema):
        assert test_schema.description == "Schema used in unit tests"

    def test_loads_four_columns(self, test_schema):
        assert len(test_schema.columns) == 4

    def test_column_names(self, test_schema):
        names = [col.name for col in test_schema.columns]
        assert names == ["id", "name", "amount", "category"]

    def test_column_types(self, test_schema):
        types = {col.name: col.type for col in test_schema.columns}
        assert types["id"] == "int64"
        assert types["name"] == "str"
        assert types["amount"] == "float64"
        assert types["category"] == "str"

    def test_required_flags(self, test_schema):
        req = {col.name: col.required for col in test_schema.columns}
        assert req["id"] is True
        assert req["name"] is True
        assert req["amount"] is False
        assert req["category"] is False

    def test_from_path_string(self):
        schema = TargetSchema.from_yaml(str(TEST_SCHEMA_YAML))
        assert schema.name == "test_schema"


class TestRequiredColumns:
    def test_returns_only_required(self, test_schema):
        assert test_schema.required_columns() == ["id", "name"]

    def test_all_required(self):
        schema = TargetSchema(
            name="s",
            columns=[
                TargetColumn(name="a", type="str", required=True),
                TargetColumn(name="b", type="int64", required=True),
            ],
        )
        assert schema.required_columns() == ["a", "b"]

    def test_none_required(self):
        schema = TargetSchema(
            name="s",
            columns=[
                TargetColumn(name="x", type="str", required=False),
            ],
        )
        assert schema.required_columns() == []


class TestPolarsDtype:
    @pytest.mark.parametrize(
        "type_str, expected",
        [
            ("str", pl.Utf8),
            ("utf8", pl.Utf8),
            ("float32", pl.Float32),
            ("float64", pl.Float64),
            ("int8", pl.Int8),
            ("int16", pl.Int16),
            ("int32", pl.Int32),
            ("int64", pl.Int64),
            ("uint8", pl.UInt8),
            ("uint16", pl.UInt16),
            ("uint32", pl.UInt32),
            ("uint64", pl.UInt64),
            ("bool", pl.Boolean),
            ("date", pl.Date),
            ("datetime", pl.Datetime),
            ("time", pl.Time),
            ("duration", pl.Duration),
            ("binary", pl.Binary),
            ("categorical", pl.Categorical),
        ],
    )
    def test_dtype_mapping(self, test_schema, type_str, expected):
        result = test_schema.polars_dtype(type_str)
        assert type(result) is type(expected)

    def test_unknown_type_raises_schema_validation_error(self, test_schema):
        with pytest.raises(SchemaValidationError, match="Unknown type string"):
            test_schema.polars_dtype("bigint")


class TestValidateLazy:
    def test_valid_lazy_frame_passes(self, test_schema, matching_df):
        test_schema.validate_lazy(matching_df.lazy())  # should not raise

    def test_missing_column_raises(self, test_schema, matching_df):
        lf = matching_df.drop("id").lazy()
        with pytest.raises(SchemaValidationError, match="Missing column: 'id'"):
            test_schema.validate_lazy(lf)

    def test_wrong_dtype_raises(self, test_schema):
        df = pl.DataFrame(
            {
                "id": pl.Series(["1", "2"], dtype=pl.Utf8),  # wrong: should be Int64
                "name": pl.Series(["Alice", "Bob"], dtype=pl.Utf8),
                "amount": pl.Series([1.0, 2.0], dtype=pl.Float64),
                "category": pl.Series(["food", "food"], dtype=pl.Utf8),
            }
        )
        with pytest.raises(SchemaValidationError, match="Column 'id'"):
            test_schema.validate_lazy(df.lazy())

    def test_multiple_errors_reported_together(self, test_schema):
        df = pl.DataFrame(
            {
                "id": pl.Series(["1"], dtype=pl.Utf8),
                "amount": pl.Series([1.0], dtype=pl.Float64),
                # 'name' and 'category' missing
            }
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            test_schema.validate_lazy(df.lazy())
        message = str(exc_info.value)
        assert "Missing column: 'name'" in message
        assert "Missing column: 'category'" in message


class TestValidateEager:
    def test_valid_data_passes(self, test_schema, matching_df):
        test_schema.validate_eager(matching_df)  # should not raise

    def test_null_in_required_column_raises(self, test_schema, matching_df):
        df_with_null = matching_df.with_columns(pl.Series("id", [1, None, 3], dtype=pl.Int64))
        with pytest.raises(SchemaValidationError, match="Required column 'id'"):
            test_schema.validate_eager(df_with_null)

    def test_null_in_optional_column_passes(self, test_schema, matching_df):
        df_with_null = matching_df.with_columns(pl.Series("amount", [1.0, None, 3.0], dtype=pl.Float64))
        test_schema.validate_eager(df_with_null)  # should not raise

    def test_missing_column_raises(self, test_schema, matching_df):
        df = matching_df.drop("name")
        with pytest.raises(SchemaValidationError, match="Missing column: 'name'"):
            test_schema.validate_eager(df)

    def test_multiple_null_columns_reported(self, test_schema, matching_df):
        df = matching_df.with_columns(
            pl.Series("id", [None, None, 3], dtype=pl.Int64),
            pl.Series("name", [None, "Bob", None], dtype=pl.Utf8),
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            test_schema.validate_eager(df)
        msg = str(exc_info.value)
        assert "Required column 'id'" in msg
        assert "Required column 'name'" in msg
