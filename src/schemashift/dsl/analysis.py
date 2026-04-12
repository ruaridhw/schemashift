"""DSL analysis helpers — extract column references from a DSL expression string."""

from __future__ import annotations

import ast as stdlib_ast

import lark

from .parser import parse


def collect_col_refs(expr: str) -> set[str]:
    """Return the set of source column names referenced in *expr*.

    Parses *expr* and walks the resulting parse tree to collect every
    ``col("name")`` reference without executing the expression.

    Raises :class:`schemashift.errors.DSLSyntaxError` on parse failure.
    """
    tree = parse(expr)
    visitor = _ColRefVisitor()
    visitor.visit(tree)
    return visitor.cols


class _ColRefVisitor(lark.Visitor):
    def __init__(self) -> None:
        self.cols: set[str] = set()

    def col_ref(self, tree: lark.Tree) -> None:
        name = stdlib_ast.literal_eval(str(tree.children[0]))
        self.cols.add(name)
