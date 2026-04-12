"""Tests for the DSL compiler — parse_and_compile against real Polars DataFrames."""

from datetime import timedelta

import polars as pl
import pytest

from schemashift.dsl import parse_and_compile
from schemashift.errors import DSLSyntaxError

# ---------------------------------------------------------------------------
# Column reference
# ---------------------------------------------------------------------------


class TestColRef:
    def test_integer_column(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        result = df.select(parse_and_compile('col("X")').alias("out"))["out"].to_list()
        assert result == [1, 2, 3]

    def test_string_column(self) -> None:
        df = pl.DataFrame({"Name": ["alice", "bob"]})
        result = df.select(parse_and_compile('col("Name")').alias("out"))["out"].to_list()
        assert result == ["alice", "bob"]

    def test_column_name_with_spaces(self) -> None:
        df = pl.DataFrame({"First Name": ["alice", "bob"]})
        result = df.select(parse_and_compile('col("First Name")').alias("out"))["out"].to_list()
        assert result == ["alice", "bob"]


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_division(self) -> None:
        df = pl.DataFrame({"X": [10, 20, 30]})
        result = df.select(parse_and_compile('col("X") / 1000').alias("out"))["out"].to_list()
        assert result == pytest.approx([0.01, 0.02, 0.03])

    def test_addition(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        result = df.select(parse_and_compile('col("X") + 10').alias("out"))["out"].to_list()
        assert result == [11, 12, 13]

    def test_subtraction(self) -> None:
        df = pl.DataFrame({"X": [10, 20, 30]})
        result = df.select(parse_and_compile('col("X") - 5').alias("out"))["out"].to_list()
        assert result == [5, 15, 25]

    def test_multiplication(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        result = df.select(parse_and_compile('col("X") * 3').alias("out"))["out"].to_list()
        assert result == [3, 6, 9]

    def test_modulo(self) -> None:
        df = pl.DataFrame({"X": [10, 11, 12]})
        result = df.select(parse_and_compile('col("X") % 3').alias("out"))["out"].to_list()
        assert result == [1, 2, 0]

    def test_complex_multi_column(self) -> None:
        df = pl.DataFrame({"Price": [100, 200, 300], "Qty": [1, 2, 3]})
        result = df.select(parse_and_compile('col("Price") * col("Qty") / 1000').alias("out"))["out"].to_list()
        assert result == pytest.approx([0.1, 0.4, 0.9])

    def test_operator_precedence_mul_before_add(self) -> None:
        df = pl.DataFrame({"X": [2, 4, 6], "Y": [3, 3, 3]})
        result = df.select(parse_and_compile('col("X") + col("Y") * 2').alias("out"))["out"].to_list()
        assert result == [8, 10, 12]

    def test_parentheses_override_precedence(self) -> None:
        df = pl.DataFrame({"X": [2, 4, 6]})
        result = df.select(parse_and_compile('(col("X") + 1) * 2').alias("out"))["out"].to_list()
        assert result == [6, 10, 14]


# ---------------------------------------------------------------------------
# Unary minus
# ---------------------------------------------------------------------------


class TestUnaryMinus:
    def test_negate_column(self) -> None:
        df = pl.DataFrame({"X": [1, -2, 3]})
        result = df.select(parse_and_compile('-col("X")').alias("out"))["out"].to_list()
        assert result == [-1, 2, -3]

    def test_negate_expression(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        result = df.select(parse_and_compile('-(col("X") + 1)').alias("out"))["out"].to_list()
        assert result == [-2, -3, -4]


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_equal(self) -> None:
        df = pl.DataFrame({"X": ["a", "b", "a"]})
        result = df.select(parse_and_compile('col("X") == "a"').alias("out"))["out"].to_list()
        assert result == [True, False, True]

    def test_not_equal(self) -> None:
        df = pl.DataFrame({"X": ["a", "b", "c"]})
        result = df.select(parse_and_compile('col("X") != "a"').alias("out"))["out"].to_list()
        assert result == [False, True, True]

    def test_greater_than(self) -> None:
        df = pl.DataFrame({"X": [1, 5, 10]})
        result = df.select(parse_and_compile('col("X") > 4').alias("out"))["out"].to_list()
        assert result == [False, True, True]

    def test_less_than(self) -> None:
        df = pl.DataFrame({"X": [1, 5, 10]})
        result = df.select(parse_and_compile('col("X") < 6').alias("out"))["out"].to_list()
        assert result == [True, True, False]

    def test_greater_equal(self) -> None:
        df = pl.DataFrame({"X": [3, 5, 7]})
        result = df.select(parse_and_compile('col("X") >= 5').alias("out"))["out"].to_list()
        assert result == [False, True, True]

    def test_less_equal(self) -> None:
        df = pl.DataFrame({"X": [3, 5, 7]})
        result = df.select(parse_and_compile('col("X") <= 5').alias("out"))["out"].to_list()
        assert result == [True, True, False]


# ---------------------------------------------------------------------------
# abs and round
# ---------------------------------------------------------------------------


class TestAbsAndRound:
    def test_abs(self) -> None:
        df = pl.DataFrame({"X": [-1.0, 2.0, -3.5]})
        result = df.select(parse_and_compile('col("X").abs()').alias("out"))["out"].to_list()
        assert result == pytest.approx([1.0, 2.0, 3.5])

    def test_round(self) -> None:
        df = pl.DataFrame({"X": [1.235, 2.345, 3.456]})
        result = df.select(parse_and_compile('col("X").round(2)').alias("out"))["out"].to_list()
        assert result == pytest.approx([1.24, 2.35, 3.46], abs=1e-6)

    def test_round_to_zero_decimals(self) -> None:
        df = pl.DataFrame({"X": [1.5, 2.5, 3.4]})
        result = df.select(parse_and_compile('col("X").round(0)').alias("out"))["out"].to_list()
        assert result == pytest.approx([2.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# fill_null and is_null
# ---------------------------------------------------------------------------


class TestNullHandling:
    def test_fill_null_with_integer(self) -> None:
        df = pl.DataFrame({"X": [1, None, 3]})
        result = df.select(parse_and_compile('col("X").fill_null(0)').alias("out"))["out"].to_list()
        assert result == [1, 0, 3]

    def test_fill_null_with_string(self) -> None:
        df = pl.DataFrame({"X": ["a", None, "c"]})
        result = df.select(parse_and_compile('col("X").fill_null("unknown")').alias("out"))["out"].to_list()
        assert result == ["a", "unknown", "c"]

    def test_is_null(self) -> None:
        df = pl.DataFrame({"X": [1, None, 3]})
        result = df.select(parse_and_compile('col("X").is_null()').alias("out"))["out"].to_list()
        assert result == [False, True, False]


# ---------------------------------------------------------------------------
# cast
# ---------------------------------------------------------------------------


class TestCast:
    def test_cast_int_to_float64(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        result = df.select(parse_and_compile('col("X").cast("float64")').alias("out"))
        assert result["out"].dtype == pl.Float64
        assert result["out"].to_list() == pytest.approx([1.0, 2.0, 3.0])

    def test_cast_float_to_int32(self) -> None:
        df = pl.DataFrame({"X": [1.9, 2.1, 3.7]})
        result = df.select(parse_and_compile('col("X").cast("int32")').alias("out"))
        assert result["out"].dtype == pl.Int32
        assert result["out"].to_list() == [1, 2, 3]

    def test_cast_int_to_str(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        result = df.select(parse_and_compile('col("X").cast("str")').alias("out"))
        assert result["out"].dtype == pl.Utf8

    def test_cast_str_to_utf8_alias(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        result = df.select(parse_and_compile('col("X").cast("utf8")').alias("out"))
        assert result["out"].dtype == pl.Utf8

    def test_cast_to_boolean(self) -> None:
        df = pl.DataFrame({"X": [0, 1, 1]})
        result = df.select(parse_and_compile('col("X").cast("bool")').alias("out"))
        assert result["out"].dtype == pl.Boolean


# ---------------------------------------------------------------------------
# String methods
# ---------------------------------------------------------------------------


class TestStringMethods:
    def test_str_lower(self) -> None:
        df = pl.DataFrame({"X": ["HELLO", "WORLD"]})
        result = df.select(parse_and_compile('col("X").str.lower()').alias("out"))["out"].to_list()
        assert result == ["hello", "world"]

    def test_str_upper(self) -> None:
        df = pl.DataFrame({"X": ["hello", "world"]})
        result = df.select(parse_and_compile('col("X").str.upper()').alias("out"))["out"].to_list()
        assert result == ["HELLO", "WORLD"]

    def test_str_to_lowercase(self) -> None:
        df = pl.DataFrame({"X": ["HELLO"]})
        result = df.select(parse_and_compile('col("X").str.to_lowercase()').alias("out"))["out"].to_list()
        assert result == ["hello"]

    def test_str_to_uppercase(self) -> None:
        df = pl.DataFrame({"X": ["hello"]})
        result = df.select(parse_and_compile('col("X").str.to_uppercase()').alias("out"))["out"].to_list()
        assert result == ["HELLO"]

    def test_str_strip(self) -> None:
        df = pl.DataFrame({"X": ["  hello  ", " world"]})
        result = df.select(parse_and_compile('col("X").str.strip()').alias("out"))["out"].to_list()
        assert result == ["hello", "world"]

    def test_str_replace(self) -> None:
        df = pl.DataFrame({"X": ["hello world", "world"]})
        result = df.select(parse_and_compile('col("X").str.replace("world", "earth")').alias("out"))["out"].to_list()
        assert result == ["hello earth", "earth"]

    def test_str_contains(self) -> None:
        df = pl.DataFrame({"X": ["foobar", "baz", "foo"]})
        result = df.select(parse_and_compile('col("X").str.contains("foo")').alias("out"))["out"].to_list()
        assert result == [True, False, True]

    def test_str_starts_with(self) -> None:
        df = pl.DataFrame({"X": ["hello", "world", "help"]})
        result = df.select(parse_and_compile('col("X").str.starts_with("hel")').alias("out"))[  # ignore:typo
            "out"
        ].to_list()
        assert result == [True, False, True]

    def test_str_ends_with(self) -> None:
        df = pl.DataFrame({"X": ["hello", "world", "jello"]})
        result = df.select(parse_and_compile('col("X").str.ends_with("llo")').alias("out"))["out"].to_list()
        assert result == [True, False, True]

    def test_str_slice(self) -> None:
        df = pl.DataFrame({"X": ["hello", "world"]})
        result = df.select(parse_and_compile('col("X").str.slice(1, 3)').alias("out"))["out"].to_list()
        assert result == ["ell", "orl"]

    def test_str_lengths(self) -> None:
        df = pl.DataFrame({"X": ["hi", "hello", "hey"]})
        result = df.select(parse_and_compile('col("X").str.lengths()').alias("out"))["out"].to_list()
        assert result == [2, 5, 3]

    def test_str_to_datetime(self) -> None:
        df = pl.DataFrame({"X": ["2024-01-15", "2024-06-30"]})
        result = df.select(parse_and_compile('col("X").str.to_datetime("%Y-%m-%d")').alias("out"))
        assert result["out"].dtype == pl.Datetime
        assert result["out"][0].year == 2024
        assert result["out"][0].month == 1
        assert result["out"][0].day == 15

    def test_str_extract(self) -> None:
        df = pl.DataFrame({"X": ["abc123def", "xyz456ghi"]})
        # Use escaped backslash inside the DSL string literal.
        result = df.select(parse_and_compile('col("X").str.extract("(\\\\d+)", 1)').alias("out"))["out"].to_list()
        assert result == ["123", "456"]

    def test_str_strip_then_lower_chained(self) -> None:
        df = pl.DataFrame({"X": ["  HELLO  ", " WORLD "]})
        result = df.select(parse_and_compile('col("X").str.strip().str.lower()').alias("out"))["out"].to_list()
        assert result == ["hello", "world"]


# ---------------------------------------------------------------------------
# Datetime methods
# ---------------------------------------------------------------------------


class TestDatetimeMethods:
    @pytest.fixture
    def date_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"ts": ["2024-03-15", "2023-11-07"]},
        ).with_columns(pl.col("ts").str.to_datetime("%Y-%m-%d").alias("ts"))

    def test_dt_year(self, date_df: pl.DataFrame) -> None:
        result = date_df.select(parse_and_compile('col("ts").dt.year()').alias("out"))
        assert result["out"].to_list() == [2024, 2023]

    def test_dt_month(self, date_df: pl.DataFrame) -> None:
        result = date_df.select(parse_and_compile('col("ts").dt.month()').alias("out"))
        assert result["out"].to_list() == [3, 11]

    def test_dt_day(self, date_df: pl.DataFrame) -> None:
        result = date_df.select(parse_and_compile('col("ts").dt.day()').alias("out"))
        assert result["out"].to_list() == [15, 7]

    def test_dt_strftime(self, date_df: pl.DataFrame) -> None:
        result = date_df.select(parse_and_compile('col("ts").dt.strftime("%Y")').alias("out"))
        assert result["out"].to_list() == ["2024", "2023"]

    def test_dt_strftime_full_format(self, date_df: pl.DataFrame) -> None:
        result = date_df.select(parse_and_compile('col("ts").dt.strftime("%d/%m/%Y")').alias("out"))
        assert result["out"].to_list() == ["15/03/2024", "07/11/2023"]


# ---------------------------------------------------------------------------
# When / otherwise
# ---------------------------------------------------------------------------


class TestWhenOtherwise:
    def test_simple_when_otherwise(self) -> None:
        df = pl.DataFrame({"X": ["a", "b", "c"]})
        result = df.select(parse_and_compile('when(col("X") == "a", 1).otherwise(0)').alias("out"))["out"].to_list()
        assert result == [1, 0, 0]

    def test_chained_when_otherwise(self) -> None:
        df = pl.DataFrame({"X": ["a", "b", "c"]})
        result = df.select(
            parse_and_compile('when(col("X") == "a", 1).when(col("X") == "b", 2).otherwise(3)').alias("out")
        )["out"].to_list()
        assert result == [1, 2, 3]

    def test_when_with_numeric_comparison(self) -> None:
        df = pl.DataFrame({"Score": [90, 70, 50]})
        result = df.select(parse_and_compile('when(col("Score") >= 80, "pass").otherwise("fail")').alias("out"))[
            "out"
        ].to_list()
        assert result == ["pass", "fail", "fail"]

    def test_when_value_is_expression(self) -> None:
        df = pl.DataFrame({"X": [2, -1, 4]})
        result = df.select(parse_and_compile('when(col("X") > 0, col("X") * 10).otherwise(0)').alias("out"))[
            "out"
        ].to_list()
        assert result == [20, 0, 40]

    def test_when_otherwise_with_null(self) -> None:
        df = pl.DataFrame({"X": [1, None, 3]})
        result = df.select(parse_and_compile('when(col("X").is_null(), -1).otherwise(col("X"))').alias("out"))[
            "out"
        ].to_list()
        assert result == [1, -1, 3]


# ---------------------------------------------------------------------------
# Literal values in expressions
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_string_literal(self) -> None:
        # pl.lit broadcasts to the DataFrame length only when combined with columns.
        # When selecting a bare literal Polars returns one row.
        result = pl.select(parse_and_compile('"hello"').alias("out"))["out"].to_list()
        assert result == ["hello"]

    def test_integer_literal(self) -> None:
        result = pl.select(parse_and_compile("42").alias("out"))["out"].to_list()
        assert result == [42]

    def test_null_literal(self) -> None:
        result = pl.select(parse_and_compile("null").alias("out"))["out"].to_list()
        assert result == [None]

    def test_boolean_true(self) -> None:
        result = pl.select(parse_and_compile("true").alias("out"))["out"].to_list()
        assert result == [True]

    def test_boolean_false(self) -> None:
        result = pl.select(parse_and_compile("false").alias("out"))["out"].to_list()
        assert result == [False]


# ---------------------------------------------------------------------------
# Logical operators (compiled)
# ---------------------------------------------------------------------------


class TestLogicalCompiled:
    def test_logical_and(self) -> None:
        df = pl.DataFrame({"X": [1, 5, 10], "Y": [8, 3, 7]})
        result = df.select(parse_and_compile('col("X") > 3 & col("Y") > 5').alias("out"))["out"].to_list()
        assert result == [False, False, True]

    def test_logical_or(self) -> None:
        df = pl.DataFrame({"X": [1, 5, 10]})
        result = df.select(parse_and_compile('col("X") == 1 | col("X") == 10').alias("out"))["out"].to_list()
        assert result == [True, False, True]

    def test_logical_not(self) -> None:
        df = pl.DataFrame({"flag": [True, False, True]})
        result = df.select(parse_and_compile('not col("flag")').alias("out"))["out"].to_list()
        assert result == [False, True, False]


# ---------------------------------------------------------------------------
# Datetime hour/minute/second/timestamp compiled
# ---------------------------------------------------------------------------


class TestDatetimeMethodsExtended:
    @pytest.fixture
    def datetime_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"ts": ["2024-03-15 14:35:52", "2023-11-07 09:05:01"]},
        ).with_columns(pl.col("ts").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ts"))

    def test_dt_hour(self, datetime_df: pl.DataFrame) -> None:
        result = datetime_df.select(parse_and_compile('col("ts").dt.hour()').alias("out"))
        assert result["out"].to_list() == [14, 9]

    def test_dt_minute(self, datetime_df: pl.DataFrame) -> None:
        result = datetime_df.select(parse_and_compile('col("ts").dt.minute()').alias("out"))
        assert result["out"].to_list() == [35, 5]

    def test_dt_second(self, datetime_df: pl.DataFrame) -> None:
        result = datetime_df.select(parse_and_compile('col("ts").dt.second()').alias("out"))
        assert result["out"].to_list() == [52, 1]

    def test_dt_timestamp(self, datetime_df: pl.DataFrame) -> None:
        result = datetime_df.select(parse_and_compile('col("ts").dt.timestamp()').alias("out"))
        ts_list = result["out"].to_list()
        assert len(ts_list) == 2
        # Timestamps should be positive integer milliseconds since epoch.
        assert all(t > 0 for t in ts_list)  # type: ignore[operator]
        # 2024 row should have a larger timestamp than the 2023 row.
        assert ts_list[1] < ts_list[0]

    def test_dt_timestamp_explicit_unit(self, datetime_df: pl.DataFrame) -> None:
        result = datetime_df.select(parse_and_compile('col("ts").dt.timestamp("us")').alias("out"))
        ts_list = result["out"].to_list()
        assert all(t > 0 for t in ts_list)  # type: ignore[operator]

    def test_dt_timestamp_invalid_unit_raises(self, datetime_df: pl.DataFrame) -> None:
        with pytest.raises(DSLSyntaxError, match="unit"):
            datetime_df.select(parse_and_compile('col("ts").dt.timestamp("s")').alias("out"))


# ---------------------------------------------------------------------------
# Compiler error paths
# ---------------------------------------------------------------------------


class TestCompilerErrorPaths:
    def test_cast_invalid_type_raises_dsl_syntax_error(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_and_compile('col("X").cast("pandas")')


# ---------------------------------------------------------------------------
# Coalesce
# ---------------------------------------------------------------------------


class TestCoalesce:
    def test_coalesce_three_args_returns_first_non_null(self) -> None:
        df = pl.DataFrame({"x": [1, None, None], "y": [None, 2, None]}, schema={"x": pl.Int64, "y": pl.Int64})
        result = df.select(parse_and_compile('coalesce(col("x"), col("y"), 0)').alias("out"))["out"].to_list()
        assert result == [1, 2, 0]

    def test_str_replace_regex_substitutes_pattern(self) -> None:
        df = pl.DataFrame({"name": ["abc123", "def456"]})
        result = df.select(parse_and_compile(r'col("name").str.replace_regex("\\d+", "NUM")').alias("out"))[
            "out"
        ].to_list()
        assert result == ["abcNUM", "defNUM"]

    def test_str_replace_literal_still_works(self) -> None:
        df = pl.DataFrame({"name": ["a.b", "c.d"]})
        result = df.select(parse_and_compile('col("name").str.replace(".", "W")').alias("out"))["out"].to_list()
        assert result == ["aWb", "cWd"]


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_country_to_iso2(self) -> None:
        df = pl.DataFrame({"Country": ["United Kingdom", "France", "Nonexistent"]})
        result = df.select(parse_and_compile('lookup(col("Country"), "country_to_iso2")').alias("out"))["out"].to_list()
        assert result == ["GB", "FR", "Nonexistent"]

    def test_unknown_table_raises(self) -> None:
        with pytest.raises(DSLSyntaxError, match="Unknown lookup table"):
            parse_and_compile('lookup(col("X"), "nonexistent")')


# ---------------------------------------------------------------------------
# CustomLookup
# ---------------------------------------------------------------------------


class TestCustomLookup:
    def test_string_to_string(self) -> None:
        df = pl.DataFrame({"X": ["A", "B", "C"]})
        result = df.select(parse_and_compile('custom_lookup(col("X"), {"A": "Active", "B": "Inactive"})').alias("out"))[
            "out"
        ].to_list()
        assert result == ["Active", "Inactive", "C"]

    def test_numeric_key_to_string(self) -> None:
        df = pl.DataFrame({"X": ["1", "2", "3"]})
        result = df.select(parse_and_compile('custom_lookup(col("X"), {"1": "One", "2": "Two"})').alias("out"))[
            "out"
        ].to_list()
        assert result == ["One", "Two", "3"]

    def test_bool_values(self) -> None:
        df = pl.DataFrame({"X": ["Y", "N", "Y"]})
        result = df.select(parse_and_compile('custom_lookup(col("X"), {"Y": "yes", "N": "no"})').alias("out"))[
            "out"
        ].to_list()
        assert result == ["yes", "no", "yes"]

    def test_with_base_table_extends(self) -> None:
        df = pl.DataFrame({"Country": ["United Kingdom", "Türkiye"]})
        result = df.select(
            parse_and_compile('custom_lookup(col("Country"), {"Türkiye": "TR"}, "country_to_iso2")').alias("out")
        )["out"].to_list()
        assert result == ["GB", "TR"]

    def test_unknown_base_table_raises(self) -> None:
        with pytest.raises(DSLSyntaxError, match="Unknown base table"):
            parse_and_compile('custom_lookup(col("X"), {"a": "b"}, "nonexistent")')


# ---------------------------------------------------------------------------
# New direct methods (is_not_null, floor, ceil, clip, pow, sqrt)
# ---------------------------------------------------------------------------


class TestNewDirectMethods:
    def test_is_not_null(self) -> None:
        df = pl.DataFrame({"X": [1, None, 3]})
        result = df.select(parse_and_compile('col("X").is_not_null()').alias("out"))["out"].to_list()
        assert result == [True, False, True]

    def test_floor(self) -> None:
        df = pl.DataFrame({"X": [1.7, 2.3, -1.1]})
        result = df.select(parse_and_compile('col("X").floor()').alias("out"))["out"].to_list()
        assert result == pytest.approx([1.0, 2.0, -2.0])

    def test_ceil(self) -> None:
        df = pl.DataFrame({"X": [1.1, 2.9, -1.1]})
        result = df.select(parse_and_compile('col("X").ceil()').alias("out"))["out"].to_list()
        assert result == pytest.approx([2.0, 3.0, -1.0])

    def test_sqrt(self) -> None:
        df = pl.DataFrame({"X": [4.0, 9.0, 16.0]})
        result = df.select(parse_and_compile('col("X").sqrt()').alias("out"))["out"].to_list()
        assert result == pytest.approx([2.0, 3.0, 4.0])

    def test_clip(self) -> None:
        df = pl.DataFrame({"X": [-5, 3, 15]})
        result = df.select(parse_and_compile('col("X").clip(0, 10)').alias("out"))["out"].to_list()
        assert result == [0, 3, 10]

    def test_pow(self) -> None:
        df = pl.DataFrame({"X": [2.0, 3.0, 4.0]})
        result = df.select(parse_and_compile('col("X").pow(2)').alias("out"))["out"].to_list()
        assert result == pytest.approx([4.0, 9.0, 16.0])


# ---------------------------------------------------------------------------
# New string methods (replace_all, lstrip, rstrip, count_matches, zfill)
# ---------------------------------------------------------------------------


class TestNewStringMethods:
    def test_str_replace_all(self) -> None:
        df = pl.DataFrame({"X": ["a.b.c", "d.e"]})
        result = df.select(parse_and_compile('col("X").str.replace_all(".", "-")').alias("out"))["out"].to_list()
        assert result == ["a-b-c", "d-e"]

    def test_str_lstrip(self) -> None:
        df = pl.DataFrame({"X": ["  hello  ", "  world"]})
        result = df.select(parse_and_compile('col("X").str.lstrip()').alias("out"))["out"].to_list()
        assert result == ["hello  ", "world"]

    def test_str_rstrip(self) -> None:
        df = pl.DataFrame({"X": ["  hello  ", "world  "]})
        result = df.select(parse_and_compile('col("X").str.rstrip()').alias("out"))["out"].to_list()
        assert result == ["  hello", "world"]

    def test_str_count_matches(self) -> None:
        df = pl.DataFrame({"X": ["aababc", "xyz"]})
        result = df.select(parse_and_compile('col("X").str.count_matches("a")').alias("out"))["out"].to_list()
        assert result == [3, 0]

    def test_str_zfill(self) -> None:
        df = pl.DataFrame({"X": ["42", "7", "100"]})
        result = df.select(parse_and_compile('col("X").str.zfill(5)').alias("out"))["out"].to_list()
        assert result == ["00042", "00007", "00100"]


# ---------------------------------------------------------------------------
# New datetime methods (quarter, week, weekday, ordinal_day, truncate,
#                       total_seconds, millisecond, microsecond)
# ---------------------------------------------------------------------------


class TestNewDatetimeMethods:
    @pytest.fixture
    def dt_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"ts": ["2024-03-15 14:35:52.123", "2023-11-07 09:05:01.456"]},
        ).with_columns(pl.col("ts").str.to_datetime("%Y-%m-%d %H:%M:%S%.f").alias("ts"))

    def test_dt_quarter(self, dt_df: pl.DataFrame) -> None:
        result = dt_df.select(parse_and_compile('col("ts").dt.quarter()').alias("out"))
        assert result["out"].to_list() == [1, 4]

    def test_dt_week(self, dt_df: pl.DataFrame) -> None:
        result = dt_df.select(parse_and_compile('col("ts").dt.week()').alias("out"))
        assert result["out"].to_list() == [11, 45]

    def test_dt_weekday(self, dt_df: pl.DataFrame) -> None:
        # 2024-03-15 is a Friday (4), 2023-11-07 is a Tuesday (1)
        result = dt_df.select(parse_and_compile('col("ts").dt.weekday()').alias("out"))
        assert result["out"].to_list() == [5, 2]

    def test_dt_ordinal_day(self, dt_df: pl.DataFrame) -> None:
        result = dt_df.select(parse_and_compile('col("ts").dt.ordinal_day()').alias("out"))
        assert result["out"].to_list() == [75, 311]

    def test_dt_truncate(self, dt_df: pl.DataFrame) -> None:
        result = dt_df.select(parse_and_compile('col("ts").dt.truncate("1d")').alias("out"))
        dates = [str(d)[:10] for d in result["out"].to_list()]
        assert dates == ["2024-03-15", "2023-11-07"]

    def test_dt_millisecond(self, dt_df: pl.DataFrame) -> None:
        result = dt_df.select(parse_and_compile('col("ts").dt.millisecond()').alias("out"))
        assert result["out"].to_list() == [123, 456]

    def test_dt_microsecond(self, dt_df: pl.DataFrame) -> None:
        result = dt_df.select(parse_and_compile('col("ts").dt.microsecond()').alias("out"))
        assert result["out"].to_list() == [123000, 456000]

    def test_dt_total_seconds(self) -> None:
        df = pl.DataFrame({"dur": [timedelta(hours=2), timedelta(minutes=30)]})
        result = df.select(parse_and_compile('col("dur").dt.total_seconds()').alias("out"))
        assert result["out"].to_list() == [7200, 1800]
