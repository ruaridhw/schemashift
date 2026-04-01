"""Tests for the DSL parser — grammar productions, AST shapes, and error handling."""

from __future__ import annotations

import pytest

from schemashift.dsl.ast_nodes import (
    BinaryOp,
    ColRef,
    Literal,
    MethodCall,
    UnaryOp,
    WhenChain,
    WhenClause,
)
from schemashift.dsl.parser import parse_dsl
from schemashift.errors import DSLSyntaxError

# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_integer_literal(self) -> None:
        assert parse_dsl("42") == Literal(42)

    def test_zero_literal(self) -> None:
        assert parse_dsl("0") == Literal(0)

    def test_float_literal(self) -> None:
        assert parse_dsl("3.14") == Literal(3.14)

    def test_float_leading_dot(self) -> None:
        assert parse_dsl(".5") == Literal(0.5)

    def test_float_trailing_dot(self) -> None:
        assert parse_dsl("1.") == Literal(1.0)

    def test_single_quoted_string(self) -> None:
        assert parse_dsl("'hello'") == Literal("hello")

    def test_double_quoted_string(self) -> None:
        assert parse_dsl('"world"') == Literal("world")

    def test_string_with_spaces(self) -> None:
        assert parse_dsl('"hello world"') == Literal("hello world")

    def test_boolean_true(self) -> None:
        assert parse_dsl("true") == Literal(True)

    def test_boolean_false(self) -> None:
        assert parse_dsl("false") == Literal(False)

    def test_null_literal(self) -> None:
        assert parse_dsl("null") == Literal(None)


# ---------------------------------------------------------------------------
# Column references
# ---------------------------------------------------------------------------


class TestColRef:
    def test_simple_col_ref(self) -> None:
        assert parse_dsl('col("Name")') == ColRef("Name")

    def test_col_ref_with_spaces_in_name(self) -> None:
        assert parse_dsl('col("First Name")') == ColRef("First Name")

    def test_col_ref_single_quoted(self) -> None:
        assert parse_dsl("col('Amount')") == ColRef("Amount")

    def test_col_ref_numeric_like_name(self) -> None:
        assert parse_dsl('col("col1")') == ColRef("col1")


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_addition(self) -> None:
        assert parse_dsl('col("X") + 1') == BinaryOp("+", ColRef("X"), Literal(1))

    def test_subtraction(self) -> None:
        assert parse_dsl('col("X") - 1') == BinaryOp("-", ColRef("X"), Literal(1))

    def test_multiplication(self) -> None:
        assert parse_dsl('col("X") * 2') == BinaryOp("*", ColRef("X"), Literal(2))

    def test_division(self) -> None:
        assert parse_dsl('col("X") / 1000') == BinaryOp("/", ColRef("X"), Literal(1000))

    def test_modulo(self) -> None:
        assert parse_dsl('col("X") % 3') == BinaryOp("%", ColRef("X"), Literal(3))

    def test_multiplication_before_addition(self) -> None:
        # col("X") + col("Y") * 2 => col("X") + (col("Y") * 2)
        node = parse_dsl('col("X") + col("Y") * 2')
        assert node == BinaryOp(
            "+",
            ColRef("X"),
            BinaryOp("*", ColRef("Y"), Literal(2)),
        )

    def test_addition_left_associative(self) -> None:
        # 1 + 2 + 3 => (1 + 2) + 3
        node = parse_dsl("1 + 2 + 3")
        assert node == BinaryOp("+", BinaryOp("+", Literal(1), Literal(2)), Literal(3))

    def test_complex_arithmetic(self) -> None:
        # col("X") * col("Y") / 1000
        node = parse_dsl('col("X") * col("Y") / 1000')
        assert node == BinaryOp(
            "/",
            BinaryOp("*", ColRef("X"), ColRef("Y")),
            Literal(1000),
        )


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_equal(self) -> None:
        assert parse_dsl('col("X") == "foo"') == BinaryOp("==", ColRef("X"), Literal("foo"))

    def test_not_equal(self) -> None:
        assert parse_dsl('col("X") != "bar"') == BinaryOp("!=", ColRef("X"), Literal("bar"))

    def test_greater_than(self) -> None:
        assert parse_dsl('col("X") > 0') == BinaryOp(">", ColRef("X"), Literal(0))

    def test_less_than(self) -> None:
        assert parse_dsl('col("X") < 10') == BinaryOp("<", ColRef("X"), Literal(10))

    def test_greater_equal(self) -> None:
        assert parse_dsl('col("X") >= 5') == BinaryOp(">=", ColRef("X"), Literal(5))

    def test_less_equal(self) -> None:
        assert parse_dsl('col("X") <= 100') == BinaryOp("<=", ColRef("X"), Literal(100))


# ---------------------------------------------------------------------------
# Logical operators
# ---------------------------------------------------------------------------


class TestLogical:
    def test_logical_and(self) -> None:
        node = parse_dsl('col("X") > 0 & col("Y") < 10')
        assert node == BinaryOp(
            "&",
            BinaryOp(">", ColRef("X"), Literal(0)),
            BinaryOp("<", ColRef("Y"), Literal(10)),
        )

    def test_logical_or(self) -> None:
        node = parse_dsl('col("X") == 1 | col("X") == 2')
        assert node == BinaryOp(
            "|",
            BinaryOp("==", ColRef("X"), Literal(1)),
            BinaryOp("==", ColRef("X"), Literal(2)),
        )

    def test_logical_and_left_associative(self) -> None:
        node = parse_dsl('col("A") > 0 & col("B") > 0 & col("C") > 0')
        assert node == BinaryOp(
            "&",
            BinaryOp(
                "&",
                BinaryOp(">", ColRef("A"), Literal(0)),
                BinaryOp(">", ColRef("B"), Literal(0)),
            ),
            BinaryOp(">", ColRef("C"), Literal(0)),
        )


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------


class TestUnary:
    def test_negation_literal(self) -> None:
        assert parse_dsl("-1") == UnaryOp("-", Literal(1))

    def test_negation_col(self) -> None:
        assert parse_dsl('-col("X")') == UnaryOp("-", ColRef("X"))

    def test_negation_parenthesised_expression(self) -> None:
        node = parse_dsl('-(col("X") + 1)')
        assert node == UnaryOp("-", BinaryOp("+", ColRef("X"), Literal(1)))

    def test_double_negation(self) -> None:
        assert parse_dsl("-(-1)") == UnaryOp("-", UnaryOp("-", Literal(1)))


# ---------------------------------------------------------------------------
# Parenthesised expressions
# ---------------------------------------------------------------------------


class TestParentheses:
    def test_simple_group(self) -> None:
        assert parse_dsl("(1 + 2)") == BinaryOp("+", Literal(1), Literal(2))

    def test_changes_precedence(self) -> None:
        # (col("X") + 1) * 2
        node = parse_dsl('(col("X") + 1) * 2')
        assert node == BinaryOp(
            "*",
            BinaryOp("+", ColRef("X"), Literal(1)),
            Literal(2),
        )

    def test_nested_parens(self) -> None:
        node = parse_dsl("((1 + 2))")
        assert node == BinaryOp("+", Literal(1), Literal(2))


# ---------------------------------------------------------------------------
# Direct methods
# ---------------------------------------------------------------------------


class TestDirectMethods:
    def test_abs(self) -> None:
        node = parse_dsl('col("X").abs()')
        assert node == MethodCall(ColRef("X"), "abs", ())

    def test_is_null(self) -> None:
        node = parse_dsl('col("X").is_null()')
        assert node == MethodCall(ColRef("X"), "is_null", ())

    def test_round(self) -> None:
        node = parse_dsl('col("X").round(2)')
        assert node == MethodCall(ColRef("X"), "round", (Literal(2),))

    def test_cast(self) -> None:
        node = parse_dsl('col("X").cast("float64")')
        assert node == MethodCall(ColRef("X"), "cast", (Literal("float64"),))

    def test_fill_null_with_zero(self) -> None:
        node = parse_dsl('col("X").fill_null(0)')
        assert node == MethodCall(ColRef("X"), "fill_null", (Literal(0),))

    def test_fill_null_with_string(self) -> None:
        node = parse_dsl('col("X").fill_null("n/a")')
        assert node == MethodCall(ColRef("X"), "fill_null", (Literal("n/a"),))


# ---------------------------------------------------------------------------
# String methods
# ---------------------------------------------------------------------------


class TestStringMethods:
    def test_str_lower(self) -> None:
        node = parse_dsl('col("X").str.lower()')
        assert node == MethodCall(ColRef("X"), "str.lower", ())

    def test_str_upper(self) -> None:
        node = parse_dsl('col("X").str.upper()')
        assert node == MethodCall(ColRef("X"), "str.upper", ())

    def test_str_to_lowercase(self) -> None:
        node = parse_dsl('col("X").str.to_lowercase()')
        assert node == MethodCall(ColRef("X"), "str.to_lowercase", ())

    def test_str_to_uppercase(self) -> None:
        node = parse_dsl('col("X").str.to_uppercase()')
        assert node == MethodCall(ColRef("X"), "str.to_uppercase", ())

    def test_str_strip(self) -> None:
        node = parse_dsl('col("X").str.strip()')
        assert node == MethodCall(ColRef("X"), "str.strip", ())

    def test_str_replace(self) -> None:
        node = parse_dsl('col("X").str.replace("a", "b")')
        assert node == MethodCall(ColRef("X"), "str.replace", (Literal("a"), Literal("b")))

    def test_str_contains(self) -> None:
        node = parse_dsl('col("X").str.contains("foo")')
        assert node == MethodCall(ColRef("X"), "str.contains", (Literal("foo"),))

    def test_str_starts_with(self) -> None:
        node = parse_dsl('col("X").str.starts_with("pre")')
        assert node == MethodCall(ColRef("X"), "str.starts_with", (Literal("pre"),))

    def test_str_ends_with(self) -> None:
        node = parse_dsl('col("X").str.ends_with("suf")')
        assert node == MethodCall(ColRef("X"), "str.ends_with", (Literal("suf"),))

    def test_str_slice(self) -> None:
        node = parse_dsl('col("X").str.slice(0, 3)')
        assert node == MethodCall(ColRef("X"), "str.slice", (Literal(0), Literal(3)))

    def test_str_to_datetime(self) -> None:
        node = parse_dsl('col("X").str.to_datetime("%Y-%m-%d")')
        assert node == MethodCall(ColRef("X"), "str.to_datetime", (Literal("%Y-%m-%d"),))

    def test_str_lengths(self) -> None:
        node = parse_dsl('col("X").str.lengths()')
        assert node == MethodCall(ColRef("X"), "str.lengths", ())

    def test_str_extract(self) -> None:
        # Use escaped backslash inside the DSL string literal.
        node = parse_dsl('col("X").str.extract("(\\\\d+)", 1)')
        assert node == MethodCall(ColRef("X"), "str.extract", (Literal(r"(\d+)"), Literal(1)))


# ---------------------------------------------------------------------------
# Datetime methods
# ---------------------------------------------------------------------------


class TestDatetimeMethods:
    def test_dt_year(self) -> None:
        node = parse_dsl('col("X").dt.year()')
        assert node == MethodCall(ColRef("X"), "dt.year", ())

    def test_dt_month(self) -> None:
        node = parse_dsl('col("X").dt.month()')
        assert node == MethodCall(ColRef("X"), "dt.month", ())

    def test_dt_day(self) -> None:
        node = parse_dsl('col("X").dt.day()')
        assert node == MethodCall(ColRef("X"), "dt.day", ())

    def test_dt_hour(self) -> None:
        node = parse_dsl('col("X").dt.hour()')
        assert node == MethodCall(ColRef("X"), "dt.hour", ())

    def test_dt_minute(self) -> None:
        node = parse_dsl('col("X").dt.minute()')
        assert node == MethodCall(ColRef("X"), "dt.minute", ())

    def test_dt_second(self) -> None:
        node = parse_dsl('col("X").dt.second()')
        assert node == MethodCall(ColRef("X"), "dt.second", ())

    def test_dt_strftime(self) -> None:
        node = parse_dsl('col("X").dt.strftime("%Y")')
        assert node == MethodCall(ColRef("X"), "dt.strftime", (Literal("%Y"),))

    def test_dt_timestamp(self) -> None:
        node = parse_dsl('col("X").dt.timestamp()')
        assert node == MethodCall(ColRef("X"), "dt.timestamp", ())


# ---------------------------------------------------------------------------
# Method chaining
# ---------------------------------------------------------------------------


class TestMethodChaining:
    def test_strip_then_lower(self) -> None:
        node = parse_dsl('col("X").str.strip().str.lower()')
        inner = MethodCall(ColRef("X"), "str.strip", ())
        outer = MethodCall(inner, "str.lower", ())
        assert node == outer

    def test_abs_then_round(self) -> None:
        node = parse_dsl('col("X").abs().round(2)')
        inner = MethodCall(ColRef("X"), "abs", ())
        outer = MethodCall(inner, "round", (Literal(2),))
        assert node == outer

    def test_fill_null_then_cast(self) -> None:
        node = parse_dsl('col("X").fill_null(0).cast("float64")')
        inner = MethodCall(ColRef("X"), "fill_null", (Literal(0),))
        outer = MethodCall(inner, "cast", (Literal("float64"),))
        assert node == outer


# ---------------------------------------------------------------------------
# When / otherwise
# ---------------------------------------------------------------------------


class TestWhenChain:
    def test_simple_when_otherwise(self) -> None:
        node = parse_dsl('when(col("X") == "a", 1).otherwise(0)')
        assert node == WhenChain(
            whens=(WhenClause(BinaryOp("==", ColRef("X"), Literal("a")), Literal(1)),),
            otherwise=Literal(0),
        )

    def test_double_when_otherwise(self) -> None:
        node = parse_dsl('when(col("X") == "a", 1).when(col("X") == "b", 2).otherwise(3)')
        assert node == WhenChain(
            whens=(
                WhenClause(BinaryOp("==", ColRef("X"), Literal("a")), Literal(1)),
                WhenClause(BinaryOp("==", ColRef("X"), Literal("b")), Literal(2)),
            ),
            otherwise=Literal(3),
        )

    def test_when_with_complex_condition(self) -> None:
        node = parse_dsl('when(col("X") > 0 & col("Y") < 10, "ok").otherwise("bad")')
        expected_cond = BinaryOp(
            "&",
            BinaryOp(">", ColRef("X"), Literal(0)),
            BinaryOp("<", ColRef("Y"), Literal(10)),
        )
        assert node == WhenChain(
            whens=(WhenClause(expected_cond, Literal("ok")),),
            otherwise=Literal("bad"),
        )

    def test_when_value_is_expression(self) -> None:
        node = parse_dsl('when(col("X") > 0, col("X") * 2).otherwise(0)')
        assert node == WhenChain(
            whens=(
                WhenClause(
                    BinaryOp(">", ColRef("X"), Literal(0)),
                    BinaryOp("*", ColRef("X"), Literal(2)),
                ),
            ),
            otherwise=Literal(0),
        )


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


class TestSyntaxErrors:
    def test_empty_expression(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("   ")

    def test_unclosed_paren(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("X"')

    def test_extra_close_paren(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("X"))')

    def test_col_missing_arg(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("col()")

    def test_col_non_string_arg(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("col(123)")

    def test_unknown_identifier(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("foobar")

    def test_unknown_method(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("X").nonexistent()')

    def test_unknown_str_submethod(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("X").str.hack()')

    def test_when_missing_otherwise(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('when(col("X") == 1, "y")')

    def test_when_missing_comma(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('when(col("X") == 1 "y").otherwise("n")')

    def test_binary_op_missing_right_operand(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("X") +')

    def test_error_contains_position(self) -> None:
        try:
            parse_dsl('col("X").badmethod()')
        except DSLSyntaxError as exc:
            assert exc.position >= 0
        else:
            pytest.fail("Expected DSLSyntaxError")

    def test_error_contains_expression(self) -> None:
        expr = 'col("X").badmethod()'
        try:
            parse_dsl(expr)
        except DSLSyntaxError as exc:
            assert exc.expression == expr
        else:
            pytest.fail("Expected DSLSyntaxError")

    def test_invalid_character(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("col(\"X\") @ 1")
