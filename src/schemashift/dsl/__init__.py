"""DSL parser and compiler for safe column expressions.

Expression Reference
--------------------
  col("Column Name")                        # reference a column
  col("X") + col("Y")                       # arithmetic: +, -, *, /, %
  col("X") / 1000                           # divide by constant
  col("Name").str.strip()                   # strip whitespace
  col("Name").str.lower()                   # lowercase
  col("Name").str.to_uppercase()            # uppercase
  col("Name").str.slice(0, 3)               # first 3 chars
  col("Name").str.replace("old", "new")                  # replace substring (literal)
  col("Name").str.replace_regex("\\d+", "NUM")           # replace via regex
  coalesce(col("A"), col("B"), "fallback")               # first non-null value
  col("Name").str.contains("x")            # boolean contains
  col("Name").str.starts_with("x")         # boolean
  col("Name").str.ends_with("x")           # boolean
  col("Date").str.to_datetime("%Y-%m-%d")  # parse datetime
  col("Name").str.lengths()                # string length
  col("dt").dt.year()                      # extract year
  col("dt").dt.month()                     # extract month
  col("dt").dt.day()                       # extract day
  col("dt").dt.strftime("%Y-%m-%d")        # format datetime
  col("x").round(2)                        # round
  col("x").abs()                           # absolute value
  col("x").cast("float64")                 # cast: str, int32, int64, float32, float64, bool, datetime, date
  col("x").fill_null(0)                    # fill nulls
  col("x").is_null()                       # boolean null check
  when(col("T") == "A", "Result A").otherwise("Other")                        # conditional
  when(col("T") == "A", "A").when(col("T") == "B", "B").otherwise("C")        # chained
  lookup(col("Country"), "country_to_iso2")                                   # built-in table
  custom_lookup(col("Status"), {"A": "Active", "B": "Inactive"})              # user-defined mapping
  custom_lookup(col("Country"), {"Türkiye": "TR"}, "country_to_iso2")         # extend built-in table
"""

import polars as pl

from .analysis import collect_col_refs
from .compiler import compile_dsl
from .parser import parse_dsl

__all__ = ["parse_dsl", "compile_dsl", "parse_and_compile", "collect_col_refs"]


def parse_and_compile(expression: str) -> pl.Expr:
    """Parse a DSL expression string and compile it to a :class:`polars.Expr`."""
    return compile_dsl(parse_dsl(expression))
