"""DSL parser and compiler for safe column expressions."""

import polars as pl

from .compiler import compile_dsl
from .parser import parse_dsl

__all__ = ["parse_dsl", "compile_dsl", "parse_and_compile"]


def parse_and_compile(expression: str) -> pl.Expr:
    """Parse a DSL expression string and compile it to a :class:`polars.Expr`."""
    return compile_dsl(parse_dsl(expression))
