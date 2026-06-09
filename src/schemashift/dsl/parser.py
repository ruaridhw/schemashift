"""Lark-based parser for the schemashift DSL.

Wraps the Lark LALR parser and translates all parse failures to
:class:`schemashift.errors.DSLSyntaxError`.
"""

from __future__ import annotations

from pathlib import Path

import lark

from schemashift.errors import DSLSyntaxError

_GRAMMAR = (Path(__file__).parent / "grammar.lark").read_text()

_PARSER = lark.Lark(
    _GRAMMAR,
    parser="lalr",
    propagate_positions=True,
)

# Examples used by match_examples() to produce human-readable error hints.
_ERROR_EXAMPLES: dict[str, list[str]] = {
    "expression must not be empty": [""],
    "missing .otherwise() clause": ['when(col("x") == 1, "y")'],
    "unclosed parenthesis": ['col("x"'],
    "incomplete method call": ['col("x").'],
}


def parse(expr: str) -> lark.Tree:
    """Parse *expr* into a Lark parse tree.

    Raises :class:`schemashift.errors.DSLSyntaxError` on any syntax error.
    """
    if not expr or not expr.strip():
        raise DSLSyntaxError(
            "Expression must not be empty",
            expression=expr,
            position=0,
        )
    try:
        return _PARSER.parse(expr)
    except lark.exceptions.UnexpectedInput as exc:
        hint = exc.match_examples(_PARSER.parse, _ERROR_EXAMPLES)
        msg = hint or _format_error(exc, expr)
        raise DSLSyntaxError(
            msg,
            expression=expr,
            position=max(0, getattr(exc, "column", 1) - 1),
        ) from exc


def _format_error(exc: lark.exceptions.UnexpectedInput, expr: str) -> str:
    """Produce a compact error message from a raw Lark exception."""
    col = getattr(exc, "column", None)
    if col is not None:
        snippet = expr[max(0, col - 1) : col + 10]
        return f"Unexpected syntax near {snippet!r} (column {col})"
    return f"Syntax error in expression: {expr!r}"
