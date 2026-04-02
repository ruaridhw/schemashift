# Expression DSL

Column mappings support a safe, closed expression language that compiles directly to native Polars expressions.

The DSL is **not** Python. There is no `eval()`, no imports, no arbitrary method calls — only the operations explicitly listed here. Anything outside this allowlist raises `DSLSyntaxError` at config-load time, before any data is touched.

## Syntax overview

```
col("ColumnName")                        # reference a column
col("X") / 1000                          # arithmetic: + - * /
col("X").method()                        # method chain
when(condition, value).otherwise(value)  # conditional
coalesce(col("A"), col("B"), "fallback") # first non-null
col("x").cast("float64")                 # type cast
```

## Column references

```
col("Name")          # exact column name (case-sensitive)
col("Sales (USD)")   # spaces and special chars are fine inside quotes
```

## Arithmetic

```
col("t_in").cast("int64") * 1000000
col("yield_pct") * col("wafer_count")
col("step_sequence") + col("offset") - 10
```

Supports `+`, `-`, `*`, `/`.

## String operations

All string methods are accessed via `.str.`:

| Expression | Description |
|------------|-------------|
| `col("x").str.strip()` | Strip leading/trailing whitespace |
| `col("x").str.lstrip()` | Strip leading whitespace |
| `col("x").str.rstrip()` | Strip trailing whitespace |
| `col("x").str.lower()` | Lowercase |
| `col("x").str.upper()` | Uppercase |
| `col("x").str.to_datetime("fmt")` | Parse string to datetime |
| `col("x").str.replace("old", "new")` | Literal string replace |
| `col("x").str.replace_regex("pattern", "replacement")` | Regex replace |
| `col("x").str.slice(start, length)` | Substring by position |
| `col("x").str.split("delim")` | Split into list |

Methods can be chained:

```
col("Name").str.strip().str.lower()
```

## Date/time operations

Date methods are accessed via `.dt.`:

| Expression | Description |
|------------|-------------|
| `col("dt").dt.year()` | Extract year |
| `col("dt").dt.month()` | Extract month |
| `col("dt").dt.day()` | Extract day |
| `col("dt").dt.hour()` | Extract hour |
| `col("dt").dt.minute()` | Extract minute |
| `col("dt").dt.second()` | Extract second |
| `col("dt").dt.date()` | Extract date part |
| `col("dt").dt.strftime("fmt")` | Format as string |

## Arithmetic on expressions

```
col("t_in").cast("int64") * 1_000_000    # Unix epoch seconds → Polars Datetime (microseconds)
```

## Direct column methods

These are called directly on `col(...)` without a namespace:

| Expression | Description |
|------------|-------------|
| `col("x").cast("float64")` | Cast to a Polars dtype |
| `col("x").is_null()` | Boolean null mask |
| `col("x").is_not_null()` | Boolean not-null mask |
| `col("x").fill_null("val")` | Replace nulls with a literal |
| `col("x").abs()` | Absolute value |

### Accepted cast targets

`str`, `int32`, `int64`, `float32`, `float64`, `bool`, `date`, `datetime`, `duration`

## Conditionals

```
when(col("Type") == "solar", "Solar").otherwise("Other")
```

Multi-branch:

```
when(col("T") == "A", "Alpha")
  .when(col("T") == "B", "Beta")
  .otherwise("Unknown")
```

`when(condition, value)` takes a boolean expression and the value to assign when true. `.otherwise(value)` is required to close the chain.

### Comparison operators

`==`, `!=`, `<`, `<=`, `>`, `>=`

## Coalesce

Returns the first non-null value across multiple expressions or literals:

```
coalesce(col("A"), col("B"), "fallback")
```

## Combining expressions

```
col("t_in").cast("int64") * 1_000_000    # epoch seconds → microseconds, then cast to datetime

col("HOLD_STATUS") != "NONE"             # boolean from string sentinel

when(col("t_out") == 0, null).otherwise(col("t_out").cast("int64") * 1_000_000)
```

## Error handling

A bad expression raises `DSLSyntaxError` at parse time (before any data is read):

```python
from schemashift.errors import DSLSyntaxError

try:
    ss.validate_config(config)
except DSLSyntaxError as e:
    print(e.expression)   # the offending expression string
    print(e.position)     # character position of the error
```

A valid expression that fails at evaluation raises `DSLRuntimeError`.
