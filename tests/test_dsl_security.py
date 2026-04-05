"""Security tests for the DSL parser and compiler.

Verifies that the DSL is a closed language — it cannot call arbitrary Python,
access the file system, or produce anything other than a polars.Expr.
"""

import polars as pl
import pytest

from schemashift.dsl import parse_and_compile
from schemashift.dsl.parser import parse_dsl
from schemashift.errors import DSLSyntaxError

# ---------------------------------------------------------------------------
# Non-allowlisted methods are rejected
# ---------------------------------------------------------------------------


class TestMethodAllowlist:
    def test_system_method_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x").system("rm -rf /")')

    def test_eval_method_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x").eval("__import__(\'os\')")')

    def test_exec_method_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x").exec("import os")')

    def test_arbitrary_method_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x").to_pandas()')

    def test_private_method_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x")._private()')

    def test_dunder_method_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x").__class__()')

    def test_unknown_str_submethod_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x").str.format("pattern")')

    def test_unknown_dt_submethod_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x").dt.tzinfo()')

    def test_unknown_namespace_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x").os.system("ls")')

    def test_list_methods_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x").list.get(0)')


# ---------------------------------------------------------------------------
# Python-builtins and injection patterns are rejected
# ---------------------------------------------------------------------------


class TestInjectionPrevention:
    def test_import_pattern_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('__import__("os")')

    def test_dunder_ident_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("__builtins__")

    def test_open_builtin_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("open")

    def test_exec_ident_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("exec")

    def test_eval_ident_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("eval")

    def test_semicolon_injection_rejected(self) -> None:
        """Semicolon is not a valid token."""
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x"); import os')

    def test_backtick_injection_rejected(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("`ls`")

    def test_hash_injection_rejected(self) -> None:
        """Hash/comment character is not a valid token."""
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("x") # + col("y")')


# ---------------------------------------------------------------------------
# Only polars.Expr is produced
# ---------------------------------------------------------------------------


class TestOutputType:
    def test_col_ref_produces_expr(self) -> None:
        result = parse_and_compile('col("x")')
        assert isinstance(result, pl.Expr)

    def test_arithmetic_produces_expr(self) -> None:
        result = parse_and_compile('col("x") + 1')
        assert isinstance(result, pl.Expr)

    def test_string_method_produces_expr(self) -> None:
        result = parse_and_compile('col("x").str.lower()')
        assert isinstance(result, pl.Expr)

    def test_when_chain_produces_expr(self) -> None:
        result = parse_and_compile('when(col("x") == 1, "a").otherwise("b")')
        assert isinstance(result, pl.Expr)

    def test_literal_produces_expr(self) -> None:
        result = parse_and_compile('"hello"')
        assert isinstance(result, pl.Expr)

    def test_complex_expression_produces_expr(self) -> None:
        expr = 'when(col("Score") >= 90, "A").when(col("Score") >= 80, "B").otherwise("C")'
        result = parse_and_compile(expr)
        assert isinstance(result, pl.Expr)


# ---------------------------------------------------------------------------
# Robustness — long and garbage inputs
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_very_long_valid_expression(self) -> None:
        """A 1000+ character expression should parse without stack overflow."""
        # Build: col("X") + 1 + 1 + 1 ... (many additions)
        # Each " + 1" is 4 chars; col("X") is 8 chars.
        # Need ~250 ones for > 1000 chars: 8 + 249 * 4 = 1004.
        parts = ['col("X")'] + ["1"] * 249
        expression = " + ".join(parts)
        assert len(expression) > 1000
        result = parse_and_compile(expression)
        assert isinstance(result, pl.Expr)
        # Verify it executes correctly
        df = pl.DataFrame({"X": [0]})
        out = df.select(result.alias("out"))["out"].to_list()
        assert out == [249]

    def test_garbage_string_raises_dsl_syntax_error(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("$@#!")

    def test_sql_injection_like_string_raises_dsl_syntax_error(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("'; DROP TABLE users; --")

    def test_python_code_raises_dsl_syntax_error(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("import os; os.system('ls')")

    def test_only_operator_raises_dsl_syntax_error(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl("+")

    def test_only_comma_raises_dsl_syntax_error(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl(",")

    def test_unicode_identifier_raises_dsl_syntax_error(self) -> None:
        with pytest.raises(DSLSyntaxError):
            parse_dsl('col("α"β")')

    def test_random_numbers_only_expression_parses(self) -> None:
        """A plain number literal is valid DSL."""
        result = parse_and_compile("999")
        assert isinstance(result, pl.Expr)

    def test_deeply_nested_parens_parse(self) -> None:
        """Deeply nested parentheses should parse correctly."""
        expression = "(" * 50 + "1" + ")" * 50
        result = parse_and_compile(expression)
        assert isinstance(result, pl.Expr)

    def test_error_type_is_always_dsl_syntax_error(self) -> None:
        """Random garbage never raises unexpected exception types."""
        garbage_inputs = [
            "!!!",
            "???",
            "<<>>",
            "col(col(col()))",
            "when()",
            ".str.lower()",
            "1 + + + 1",
        ]
        for inp in garbage_inputs:
            with pytest.raises(DSLSyntaxError):  # never a bare Exception or TypeError
                parse_dsl(inp)


# ---------------------------------------------------------------------------
# Cast type validation
# ---------------------------------------------------------------------------


class TestCastValidation:
    def test_invalid_cast_type_rejected(self) -> None:
        """Unknown cast target type raises DSLSyntaxError (caught during compile)."""
        from schemashift.errors import DSLSyntaxError as DslSE

        with pytest.raises(DslSE):
            parse_and_compile('col("x").cast("pickle")')

    def test_valid_cast_types_accepted(self) -> None:
        valid_types = [
            "str",
            "utf8",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "bool",
            "date",
            "datetime",
        ]
        for t in valid_types:
            result = parse_and_compile(f'col("x").cast("{t}")')
            assert isinstance(result, pl.Expr), f"Expected Expr for cast({t!r})"


# ---------------------------------------------------------------------------
# not keyword
# ---------------------------------------------------------------------------


class TestNotKeyword:
    def test_not_produces_safe_expr(self) -> None:
        result = parse_and_compile('not col("x").is_null()')
        assert isinstance(result, pl.Expr)

    def test_double_not_produces_safe_expr(self) -> None:
        result = parse_and_compile('not not col("x").is_null()')
        assert isinstance(result, pl.Expr)
