"""Tests for the transform engine."""

from pathlib import Path

import polars as pl
import pytest

from schemashift.errors import AmbiguousFormatError, FormatDetectionError
from schemashift.models import ColumnMapping, FormatConfig
from schemashift.registry import DictRegistry
from schemashift.transform import auto_transform, dry_run, transform, validate_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "csv" / "sample.csv"
NUMBERS_CSV = FIXTURES / "csv" / "numbers.csv"
NULLABLE_CSV = FIXTURES / "csv" / "nullable.csv"


def _source_config(drop_unmapped: bool = True) -> FormatConfig:
    return FormatConfig(
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
        df = transform(SAMPLE_CSV, config).collect()
        assert "identifier" in df.columns
        assert "customer" in df.columns

    def test_basic_source_mapping_row_count(self) -> None:
        config = _source_config()
        df = transform(SAMPLE_CSV, config).collect()
        assert len(df) == 5

    def test_basic_source_mapping_values(self) -> None:
        config = _source_config()
        df = transform(SAMPLE_CSV, config).collect()
        assert df["identifier"].to_list() == [1, 2, 3, 4, 5]
        assert df["customer"].to_list() == ["Alice", "Bob", "Carol", "Dave", "Eve"]

    def test_drop_unmapped_true_excludes_original_columns(self) -> None:
        config = _source_config(drop_unmapped=True)
        df = transform(SAMPLE_CSV, config).collect()
        assert set(df.columns) == {"identifier", "customer"}

    def test_drop_unmapped_false_keeps_original_columns(self) -> None:
        config = _source_config(drop_unmapped=False)
        df = transform(SAMPLE_CSV, config).collect()
        # Original columns should still be present alongside mapped ones
        assert "identifier" in df.columns
        assert "customer" in df.columns
        assert "amount" in df.columns
        assert "category" in df.columns
        assert "active" in df.columns


# ---------------------------------------------------------------------------
# Expr mapping
# ---------------------------------------------------------------------------


class TestTransformExprMapping:
    def test_expr_divides_column_by_constant(self) -> None:
        config = FormatConfig(
            name="expr_test",
            columns=[ColumnMapping(target="x_div", expr='col("x") / 10')],
        )
        df = transform(NUMBERS_CSV, config).collect()
        assert df["x_div"].to_list() == pytest.approx([1.0, 2.0, 3.0])

    def test_expr_multiplies_two_columns(self) -> None:
        config = FormatConfig(
            name="mul_test",
            columns=[ColumnMapping(target="product", expr='col("x") * col("y")')],
        )
        df = transform(NUMBERS_CSV, config).collect()
        assert df["product"].to_list() == pytest.approx([20.0, 80.0, 180.0])

    def test_expr_arithmetic_result_correct(self) -> None:
        config = FormatConfig(
            name="arith",
            columns=[ColumnMapping(target="sum_col", expr='col("x") + col("y")')],
        )
        df = transform(NUMBERS_CSV, config).collect()
        assert df["sum_col"].to_list() == pytest.approx([12.0, 24.0, 36.0])


# ---------------------------------------------------------------------------
# Constant mapping
# ---------------------------------------------------------------------------


class TestTransformConstantMapping:
    def test_constant_string_populates_all_rows(self) -> None:
        config = FormatConfig(
            name="const_test",
            columns=[
                ColumnMapping(target="id_out", source="id"),
                ColumnMapping(target="tag", constant="processed"),
            ],
        )
        df = transform(SAMPLE_CSV, config).collect()
        assert df["tag"].to_list() == ["processed"] * 5

    def test_constant_integer(self) -> None:
        # Include a source column so the LazyFrame has a real row count when
        # drop_unmapped=True; a literal-only select yields 1 row in Polars.
        config = FormatConfig(
            name="const_int",
            columns=[
                ColumnMapping(target="id_out", source="id"),
                ColumnMapping(target="version", constant=42),
            ],
        )
        df = transform(SAMPLE_CSV, config).collect()
        assert df["version"].to_list() == [42] * 5


# ---------------------------------------------------------------------------
# Dtype casting
# ---------------------------------------------------------------------------


class TestTransformDtypeCasting:
    def test_cast_to_str(self) -> None:
        config = FormatConfig(
            name="cast_test",
            columns=[ColumnMapping(target="id_str", source="id", dtype="str")],
        )
        df = transform(SAMPLE_CSV, config).collect()
        assert df["id_str"].dtype == pl.Utf8

    def test_cast_to_float64(self) -> None:
        config = FormatConfig(
            name="cast_float",
            columns=[ColumnMapping(target="id_float", source="id", dtype="float64")],
        )
        df = transform(SAMPLE_CSV, config).collect()
        assert df["id_float"].dtype == pl.Float64
        assert df["id_float"].to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_cast_to_int32(self) -> None:
        config = FormatConfig(
            name="cast_int32",
            columns=[ColumnMapping(target="id_i32", source="id", dtype="int32")],
        )
        df = transform(SAMPLE_CSV, config).collect()
        assert df["id_i32"].dtype == pl.Int32


# ---------------------------------------------------------------------------
# Fill null
# ---------------------------------------------------------------------------


class TestTransformFillNull:
    def test_fillna_replaces_nulls(self) -> None:
        config = FormatConfig(
            name="fillna_test",
            columns=[ColumnMapping(target="value_filled", source="value", fillna=0)],
        )
        df = transform(NULLABLE_CSV, config).collect()
        # Row 2 (index 1) has a null value → should be filled with 0
        filled_values = df["value_filled"].to_list()
        assert filled_values[1] == pytest.approx(0)
        assert filled_values[0] == pytest.approx(10)
        assert filled_values[2] == pytest.approx(30)


# ---------------------------------------------------------------------------
# Validate config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config_returns_no_errors(self) -> None:
        config = FormatConfig(
            name="valid",
            columns=[
                ColumnMapping(target="out", source="in"),
                ColumnMapping(target="computed", expr='col("x") + 1'),
            ],
        )
        errors = validate_config(config)
        assert errors == []

    def test_bad_dsl_expression_returns_error(self) -> None:
        config = FormatConfig(
            name="bad_dsl",
            columns=[ColumnMapping(target="bad", expr="col( /invalid/ )")],
        )
        errors = validate_config(config)
        assert len(errors) == 1
        assert "bad" in errors[0]

    def test_multiple_bad_expressions_returns_multiple_errors(self) -> None:
        config = FormatConfig(
            name="multi_bad",
            columns=[
                ColumnMapping(target="bad1", expr="col( /broken/ )"),
                ColumnMapping(target="bad2", expr="col( @wrong@ )"),
            ],
        )
        errors = validate_config(config)
        assert len(errors) == 2

    def test_constant_and_source_mappings_are_valid(self) -> None:
        config = FormatConfig(
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


class TestDryRun:
    def test_dry_run_returns_dataframe(self) -> None:
        config = FormatConfig(
            name="dr_test",
            columns=[ColumnMapping(target="id_out", source="id")],
        )
        df = dry_run(config, SAMPLE_CSV, n_rows=3)
        assert isinstance(df, pl.DataFrame)

    def test_dry_run_respects_n_rows(self) -> None:
        config = FormatConfig(
            name="dr_rows",
            columns=[ColumnMapping(target="id_out", source="id")],
        )
        df = dry_run(config, SAMPLE_CSV, n_rows=2)
        assert len(df) == 2

    def test_dry_run_correct_columns(self) -> None:
        config = FormatConfig(
            name="dr_cols",
            columns=[
                ColumnMapping(target="id_out", source="id"),
                ColumnMapping(target="name_out", source="name"),
            ],
        )
        df = dry_run(config, SAMPLE_CSV)
        assert set(df.columns) == {"id_out", "name_out"}

    def test_dry_run_values(self) -> None:
        config = FormatConfig(
            name="dr_vals",
            columns=[ColumnMapping(target="id_out", source="id")],
        )
        df = dry_run(config, SAMPLE_CSV, n_rows=3)
        assert df["id_out"].to_list() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Auto transform
# ---------------------------------------------------------------------------


class TestAutoTransform:
    def test_auto_transform_detects_and_applies_config(self) -> None:
        reg = DictRegistry()
        config = FormatConfig(
            name="auto_cfg",
            columns=[
                ColumnMapping(target="id_out", source="id"),
                ColumnMapping(target="name_out", source="name"),
            ],
        )
        reg.register(config)
        df = auto_transform(SAMPLE_CSV, reg).collect()
        assert set(df.columns) == {"id_out", "name_out"}
        assert len(df) == 5

    def test_auto_transform_raises_when_no_match(self) -> None:
        reg = DictRegistry()
        config = FormatConfig(
            name="unmatched",
            columns=[ColumnMapping(target="out", source="nonexistent_col_xyz")],
        )
        reg.register(config)

        with pytest.raises(FormatDetectionError):
            auto_transform(SAMPLE_CSV, reg)

    def test_auto_transform_raises_when_ambiguous(self) -> None:
        reg = DictRegistry()
        reg.register(
            FormatConfig(
                name="cfg1",
                columns=[ColumnMapping(target="out", source="id")],
            )
        )
        reg.register(
            FormatConfig(
                name="cfg2",
                columns=[ColumnMapping(target="out", source="id")],
            )
        )

        with pytest.raises(AmbiguousFormatError):
            auto_transform(SAMPLE_CSV, reg)
