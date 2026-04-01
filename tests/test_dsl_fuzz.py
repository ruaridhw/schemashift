"""Targeted DSL fuzz tests for robustness."""

from __future__ import annotations

import pytest

from schemashift.dsl.parser import parse_dsl
from schemashift.errors import DSLSyntaxError

FUZZ_INPUTS = [
    # Empty and whitespace
    "",
    " ",
    "\t",
    "\n",
    "   ",
    # Partial expressions
    "col(",
    'col("',
    'col("")',
    "col()",
    "when(",
    "when(col(",
    ".str.lower()",
    ".dt.year()",
    # Operators without operands
    "+",
    "-",
    "*",
    "/",
    "==",
    "!=",
    '+col("x")',
    'col("x")+',
    'col("x") /',
    # Unmatched brackets
    '((col("x"))',
    '(col("x")',
    # Python injection attempts
    "__import__('os')",
    "eval('1')",
    "exec('import os')",
    "open('/etc/passwd')",
    "__builtins__",
    "globals()",
    "lambda x: x",
    "import os",
    "1 if True else 2",
    # Very long expressions
    'col("x") ' + '+ col("y")' * 50,
    'col("' + "a" * 200 + '")',
    # Unicode
    'col("héllo")',
    'col("价格")',
    # Nested parens
    '((((col("x"))))',
    'col("x") * (col("y") + (col("z") / 2))',
    # Method chain edge cases
    'col("x").str.lower().str.upper()',
    'col("x").round(2).abs()',
    # Invalid methods
    'col("x").nonexistent()',
    'col("x").str.nonexistent()',
    'col("x").__class__',
    'col("x").eval()',
    # Number edge cases
    "0",
    "1.5",
    "-1",
    "1e10",
    'col("x") + 1e100',
    # Comparison chaining
    'col("x") == col("y") == col("z")',
    # when without otherwise
    'when(col("x") == 1, "a")',
    # String edge cases
    'col("x with spaces")',
    'col("x\\"escaped")',
    # Null/bool literals
    'col("x").fill_null(null)',
    'col("x").fill_null(true)',
    'col("x").fill_null(false)',
]


@pytest.mark.parametrize("expression", FUZZ_INPUTS)
def test_fuzz_input_never_crashes(expression: str) -> None:
    """Parser must raise DSLSyntaxError or return AST — never raise other exceptions."""
    try:
        parse_dsl(expression)
    except DSLSyntaxError:
        pass  # Expected
    except Exception as exc:
        pytest.fail(
            f"Parser raised {type(exc).__name__}: {exc!r} for input {expression!r}"
        )
