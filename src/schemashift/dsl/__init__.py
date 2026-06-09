"""DSL parser and compiler for safe column expressions.

Expression Reference
--------------------
  col("Column Name")                        # reference a column
  col("X") + col("Y")                       # arithmetic: +, -, *, /, %
  col("X") / 1000                           # divide by constant
  col("Name").str.strip()                   # strip whitespace (both sides)
  col("Name").str.lstrip()                  # strip whitespace (left)
  col("Name").str.rstrip()                  # strip whitespace (right)
  col("Name").str.lower()                   # lowercase
  col("Name").str.to_uppercase()            # uppercase
  col("Name").str.slice(0, 3)               # first 3 chars
  col("Name").str.replace("old", "new")                    # replace first occurrence (literal)
  col("Name").str.replace_all("old", "new")                # replace all occurrences (literal)
  col("Name").str.replace_regex("\\d+", "NUM")             # replace via regex
  coalesce(col("A"), col("B"), "fallback")                 # first non-null value
  col("Name").str.contains("x")            # boolean contains
  col("Name").str.starts_with("x")         # boolean
  col("Name").str.ends_with("x")           # boolean
  col("Date").str.to_datetime("%Y-%m-%d")  # parse datetime
  col("Name").str.lengths()                # string length
  col("Name").str.count_matches("a")       # count regex matches
  col("Name").str.zfill(5)                 # zero-pad string
  col("dt").dt.year()                      # extract year
  col("dt").dt.month()                     # extract month
  col("dt").dt.day()                       # extract day
  col("dt").dt.quarter()                   # extract quarter (1-4)
  col("dt").dt.week()                      # ISO week number
  col("dt").dt.weekday()                   # day of week (1=Mon, 7=Sun)
  col("dt").dt.ordinal_day()               # day of year (1-365)
  col("dt").dt.strftime("%Y-%m-%d")        # format datetime
  col("dt").dt.truncate("1d")              # truncate to period
  col("dt").dt.millisecond()               # millisecond component
  col("dt").dt.microsecond()               # microsecond component
  col("dt").dt.total_seconds()             # duration to total seconds
  col("x").round(2)                        # round
  col("x").floor()                         # floor
  col("x").ceil()                          # ceiling
  col("x").abs()                           # absolute value
  col("x").sqrt()                          # square root
  col("x").pow(2)                          # raise to power
  col("x").clip(0, 100)                    # clamp to range
  col("x").cast("float64")                 # cast: str, int32, int64, float32, float64, bool, datetime, date
  col("x").fill_null(0)                    # fill nulls
  col("x").is_null()                       # boolean null check
  col("x").is_not_null()                   # boolean not-null check
  when(col("T") == "A", "Result A").otherwise("Other")                        # conditional
  when(col("T") == "A", "A").when(col("T") == "B", "B").otherwise("C")        # chained
  lookup(col("Country"), "country_to_iso2")                                   # built-in table
  custom_lookup(col("Status"), {"A": "Active", "B": "Inactive"})              # user-defined mapping
  custom_lookup(col("Country"), {"Türkiye": "TR"}, "country_to_iso2")         # extend built-in table
"""

import polars as pl
from lark.visitors import VisitError

from .analysis import collect_col_refs
from .compiler import DSLTransformer
from .parser import parse

__all__ = ["collect_col_refs", "parse_and_compile"]


def parse_and_compile(expression: str) -> pl.Expr:
    """Parse a DSL expression string and compile it to a :class:`polars.Expr`."""
    try:
        return DSLTransformer().transform(parse(expression))
    except VisitError as exc:
        raise exc.orig_exc from exc
