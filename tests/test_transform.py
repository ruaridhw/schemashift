"""Tests for the transform engine."""

from pathlib import Path

import polars as pl
import pytest

from schemashift.models import ColumnMapping, TransformSpec
from schemashift.result import TransformResult
from schemashift.transform import transform, validate_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "csv" / "sample.csv"
NUMBERS_CSV = FIXTURES / "csv" / "numbers.csv"
NULLABLE_CSV = FIXTURES / "csv" / "nullable.csv"


def _source_config(drop_unmapped: bool = True) -> TransformSpec:
    return TransformSpec(
        name="source_test",
        columns=[
            ColumnMapping(target="identifier", source="id"),
            ColumnMapping(target="customer", source="name"),
        ],
        drop_unmapped=drop_unmapped,
    )


# ---------------------------------------------------------------------------
# Source mapping
# ---------------------------------------------------------------------------


class TestTransformSourceMapping:
    def test_basic_source_mapping_renames_columns(self) -> None:
        config = _source_config()
        result = transform(SAMPLE_CSV, config)
        assert "identifier" in result.valid.columns
        assert "customer" in result.valid.columns

    def test_basic_source_mapping_row_count(self) -> None:
        config = _source_config()
        result = transform(SAMPLE_CSV, config)
        assert len(result.valid) == 5

    def test_basic_source_mapping_values(self) -> None:
        config = _source_config()
        result = transform(SAMPLE_CSV, config)
        assert result.valid["identifier"].to_list() == [1, 2, 3, 4, 5]
        assert result.valid["customer"].to_list() == ["Alice", "Bob", "Carol", "Dave", "Eve"]

    def test_drop_unmapped_true_excludes_original_columns(self) -> None:
        config = _source_config(drop_unmapped=True)
        result = transform(SAMPLE_CSV, config)
        assert set(result.valid.columns) == {"identifier", "customer"}

    def test_drop_unmapped_false_keeps_original_columns(self) -> None:
        config = _source_config(drop_unmapped=False)
        result = transform(SAMPLE_CSV, config)
        # Original columns should still be present alongside mapped ones
        assert "identifier" in result.valid.columns
        assert "customer" in result.valid.columns
        assert "amount" in result.valid.columns
        assert "category" in result.valid.columns
        assert "active" in result.valid.columns


# ---------------------------------------------------------------------------
# Expr mapping
# ---------------------------------------------------------------------------


class TestTransformExprMapping:
    def test_expr_divides_column_by_constant(self) -> None:
        config = TransformSpec(
            name="expr_test",
            columns=[ColumnMapping(target="x_div", expr='col("x") / 10')],
        )
        result = transform(NUMBERS_CSV, config)
        assert result.valid["x_div"].to_list() == pytest.approx([1.0, 2.0, 3.0])

    def test_expr_multiplies_two_columns(self) -> None:
        config = TransformSpec(
            name="mul_test",
            columns=[ColumnMapping(target="product", expr='col("x") * col("y")')],
        )
        result = transform(NUMBERS_CSV, config)
        assert result.valid["product"].to_list() == pytest.approx([20.0, 80.0, 180.0])

    def test_expr_arithmetic_result_correct(self) -> None:
        config = TransformSpec(
            name="arith",
            columns=[ColumnMapping(target="sum_col", expr='col("x") + col("y")')],
        )
        result = transform(NUMBERS_CSV, config)
        assert result.valid["sum_col"].to_list() == pytest.approx([12.0, 24.0, 36.0])

    def test_invalid_dsl_syntax_captured_in_expression_errors(self) -> None:
        config = TransformSpec(
            name="bad_expr",
            columns=[ColumnMapping(target="sum_col", expr='col("x"')],
        )
        result = transform(NUMBERS_CSV, config)
        assert "sum_col" in result.failures.expression_errors
        assert result.failures.has_failures


# ---------------------------------------------------------------------------
# Constant mapping
# ---------------------------------------------------------------------------


class TestTransformConstantMapping:
    def test_constant_string_populates_all_rows(self) -> None:
        config = TransformSpec(
            name="const_test",
            columns=[
                ColumnMapping(target="id_out", source="id"),
                ColumnMapping(target="tag", constant="processed"),
            ],
        )
        result = transform(SAMPLE_CSV, config)
        assert result.valid["tag"].to_list() == ["processed"] * 5

    def test_constant_integer(self) -> None:
        # Include a source column so the LazyFrame has a real row count when
        # drop_unmapped=True; a literal-only select yields 1 row in Polars.
        config = TransformSpec(
            name="const_int",
            columns=[
                ColumnMapping(target="id_out", source="id"),
                ColumnMapping(target="version", constant=42),
            ],
        )
        result = transform(SAMPLE_CSV, config)
        assert result.valid["version"].to_list() == [42] * 5

    def test_constant_none_broadcasts_nulls(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("a\n1\n2\n")
        config = TransformSpec(
            name="test",
            columns=[ColumnMapping(target="flag", constant=None)],
        )
        result = transform(str(csv), config)
        assert result.valid["flag"].is_null().all()


# ---------------------------------------------------------------------------
# Dtype casting
# ---------------------------------------------------------------------------


class TestTransformDtypeCasting:
    def test_cast_to_str(self) -> None:
        config = TransformSpec(
            name="cast_test",
            columns=[ColumnMapping(target="id_str", source="id", dtype="str")],
        )
        result = transform(SAMPLE_CSV, config)
        assert result.valid["id_str"].dtype == pl.Utf8

    def test_cast_to_float64(self) -> None:
        config = TransformSpec(
            name="cast_float",
            columns=[ColumnMapping(target="id_float", source="id", dtype="float64")],
        )
        result = transform(SAMPLE_CSV, config)
        assert result.valid["id_float"].dtype == pl.Float64
        assert result.valid["id_float"].to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_cast_to_int32(self) -> None:
        config = TransformSpec(
            name="cast_int32",
            columns=[ColumnMapping(target="id_i32", source="id", dtype="int32")],
        )
        result = transform(SAMPLE_CSV, config)
        assert result.valid["id_i32"].dtype == pl.Int32


# ---------------------------------------------------------------------------
# Fill null
# ---------------------------------------------------------------------------


class TestTransformFillNull:
    def test_fillna_replaces_nulls(self) -> None:
        config = TransformSpec(
            name="fillna_test",
            columns=[ColumnMapping(target="value_filled", source="value", fillna=0)],
        )
        result = transform(NULLABLE_CSV, config)
        # Row 2 (index 1) has a null value → should be filled with 0
        filled_values = result.valid["value_filled"].to_list()
        assert filled_values[1] == pytest.approx(0)
        assert filled_values[0] == pytest.approx(10)
        assert filled_values[2] == pytest.approx(30)


# ---------------------------------------------------------------------------
# Validate config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config_returns_no_errors(self) -> None:
        config = TransformSpec(
            name="valid",
            columns=[
                ColumnMapping(target="out", source="in"),
                ColumnMapping(target="computed", expr='col("x") + 1'),
            ],
        )
        errors = validate_config(config)
        assert errors == []

    def test_bad_dsl_expression_returns_error(self) -> None:
        config = TransformSpec(
            name="bad_dsl",
            columns=[ColumnMapping(target="bad", expr="col( /invalid/ )")],
        )
        errors = validate_config(config)
        assert len(errors) == 1
        assert "bad" in errors[0]

    def test_multiple_bad_expressions_returns_multiple_errors(self) -> None:
        config = TransformSpec(
            name="multi_bad",
            columns=[
                ColumnMapping(target="bad1", expr="col( /broken/ )"),
                ColumnMapping(target="bad2", expr="col( @wrong@ )"),
            ],
        )
        errors = validate_config(config)
        assert len(errors) == 2

    def test_constant_and_source_mappings_are_valid(self) -> None:
        config = TransformSpec(
            name="constants",
            columns=[
                ColumnMapping(target="a", source="x"),
                ColumnMapping(target="b", constant="hello"),
            ],
        )
        errors = validate_config(config)
        assert errors == []


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestTransformNRows:
    def test_transform_returns_transform_result(self) -> None:
        config = TransformSpec(
            name="dr_test",
            columns=[ColumnMapping(target="id_out", source="id")],
        )
        result = transform(SAMPLE_CSV, config, n_rows=3)
        assert isinstance(result, TransformResult)

    def test_n_rows_limits_output(self) -> None:
        config = TransformSpec(
            name="dr_rows",
            columns=[ColumnMapping(target="id_out", source="id")],
        )
        result = transform(SAMPLE_CSV, config, n_rows=2)
        assert len(result.valid) == 2

    def test_n_rows_none_returns_all_rows(self) -> None:
        config = TransformSpec(
            name="dr_cols",
            columns=[
                ColumnMapping(target="id_out", source="id"),
                ColumnMapping(target="name_out", source="name"),
            ],
        )
        result = transform(SAMPLE_CSV, config)
        assert set(result.valid.columns) == {"id_out", "name_out"}
        assert len(result.valid) == 5

    def test_n_rows_values(self) -> None:
        config = TransformSpec(
            name="dr_vals",
            columns=[ColumnMapping(target="id_out", source="id")],
        )
        result = transform(SAMPLE_CSV, config, n_rows=3)
        assert result.valid["id_out"].to_list() == [1, 2, 3]
