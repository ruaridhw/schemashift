# Research: Dataframely Integration into SchemaShift

Date: 2026-04-11

## Part 1: Current SchemaShift Architecture

### Validation Approach (TargetSchema)

**File:** `src/schemashift/target_schema.py`

`TargetSchema` is a pydantic model containing:
- `name`: Schema identifier
- `description`: Human-readable description
- `columns`: List of `TargetColumn` objects (each with `name`, `type`, `required`, `description`)

**Two validation methods:**

1. **`validate_lazy(lf: pl.LazyFrame)`** — Structural validation only (column names + dtypes). No data collection. Checks required columns present and dtype compatibility via `isinstance()`. Collects ALL errors into a single `SchemaValidationError`.

2. **`validate_eager(df: pl.DataFrame)`** — Full validation: columns, dtypes, AND null checks. Calls `validate_lazy()` first, then counts nulls per required column. Still aggregated at column level — reports *how many* nulls, not *which rows*.

### Transform Pipeline

**File:** `src/schemashift/transform.py`

`transform(path, config, n_rows=None) -> pl.DataFrame` flow:
1. Read file → LazyFrame via `read_file(path, config.reader)`
2. Build expressions: each `ColumnMapping` → `pl.Expr` via `_mapping_to_expr()`
3. Apply: `.select(expressions)` if `drop_unmapped=True`, else `.with_columns(expressions)`
4. Optionally `.limit(n_rows)`
5. `.collect()` → DataFrame

**Single mapping transformation:**
```
ColumnMapping → pl.Expr:
  1. Source: one of source→pl.col(), expr→parse_and_compile(), constant→pl.lit()
  2. Optional dtype cast
  3. Optional fill_null
  4. Alias to mapping.target
```

### Models

**ColumnMapping:**
- `target`, `source | expr | constant` (exactly one), `dtype`, `fillna`

**ReaderConfig:**
- `skip_rows`, `sheet_name`, `separator`, `encoding`

**FormatConfig:**
- `name`, `description`, `version`, `target_schema`, `reader`, `columns`, `drop_unmapped`
- `source_columns()` extracts all referenced source column names

### Error Hierarchy

All inherit from `SchemaShiftError`:

| Error | Granularity |
|---|---|
| `SchemaValidationError` | Dataframe-level (aggregated column errors) |
| `ConfigValidationError` | Field-level |
| `DSLSyntaxError` | Expression-level (has `.expression`, `.position`) |
| `DSLRuntimeError` | Column + expression-level |
| `FormatDetectionError` / `AmbiguousFormatError` | Detection-level |
| `ReaderError`, `UnsupportedFileError` | File-level |
| `LLMGenerationError` | Attempt-level (`.attempts` list) |
| `ReviewRejectedError` | Config-level |

### Key Gap

Validation is **NOT row-level**. `SchemaValidationError` aggregates failures:
- Which columns are missing or wrong dtype
- How many nulls per required column
- But NOT which rows have nulls, NOT other data quality issues, NOT row-level detail

### DSL Compiler

**File:** `src/schemashift/dsl/compiler.py`

`parse_and_compile(expression: str) -> pl.Expr` uses Lark grammar → `DSLTransformer` → `pl.Expr`.

Supports: column refs, arithmetic, comparisons, logic, string/datetime/numeric methods, when/then/otherwise, coalesce, lookup tables. Closed language — no eval, no arbitrary method calls.

---

## Part 2: Dataframely Capabilities

### Schema System

Declarative, class-based schemas extending `Schema`:

```python
class MySchema(dy.Schema):
    user_id = dy.Int64(primary_key=True)
    name = dy.String(min_length=1, max_length=255)
    created_at = dy.Datetime(timezone="UTC")
    score = dy.Float64(min=0, max=100, nullable=False)
```

- Supports inheritance (subclass columns append to parent)
- Automatic error detection at definition time
- Primary key constraints with uniqueness validation
- Both eager and lazy DataFrame evaluation

### Three Validation Methods

1. **`schema.validate(df)`** — Raises `ValidationError` on failure; supports optional type casting
2. **`schema.is_valid(df)`** — Returns boolean
3. **`schema.filter(df)`** — Returns `(valid_rows_df, FailureInfo)` for row-level separation

### Serialisation

**JSON serialisation built-in:**
- `schema.serialise()` → JSON string (custom encoder handles Polars expressions via base64, Decimal/DateTime as ISO, tuples with type markers)
- `deserialize_schema(json_string)` → dynamically reconstructs schema without prior class knowledge
- `DeserializationError` for unknown type markers

**Parquet metadata integration:**
- `read_parquet_metadata_schema(path)` extracts schema from Parquet metadata
- `write_parquet()`, `sink_parquet()`, `read_parquet()`, `scan_parquet()`
- Delta Lake: `write_delta()`, `read_delta()`, `scan_delta()`

**No native YAML support** — JSON only via serialize/deserialize.

### Row-Level Validation (Rust Plugin)

**File:** `src/polars_plugin/validation_error.rs`

The Rust plugin provides granular row-level validation:
- `RuleFailure` struct captures per-rule failures with rule identifier and row count
- Schema-level vs column-level errors (column rules use `|` delimiter)
- Three validation functions: `all_rules_horizontal()` (row-wise bool), `all_rules()` (scalar), `all_rules_required()` (formatted error reporting)
- Locale-specific number formatting in error messages

### FailureInfo: The Row-Level Detail System

`FailureInfo` (returned by `schema.filter()`) provides:
- `.invalid()` — original invalid rows without rule columns
- `.details()` — invalid rows with rule results as valid/invalid/unknown enum
- `.counts()` — failure count per individual rule
- `.cooccurrence_counts()` — co-occurring rule failures mapped to frequency
- `.write_parquet(path)` / `.write_delta(path)` — persist failures

### Column Types (19 types)

- **Numeric**: `Integer`, `Int8-Int64`, `UInt8-UInt64`, `Float`, `Float32`, `Float64`, `Decimal`
- **Temporal**: `Date`, `Datetime`, `Time`, `Duration`
- **Categorical**: `Categorical`, `Enum`
- **Text/Binary**: `String`, `Binary`
- **Containers**: `Array`, `List`, `Struct`
- **Generic**: `Bool`, `Object`, `Any`, `Column`

### Column Constraints → Polars Expressions

All constraints translate to Polars expressions under the hood:
- String: `min_length`, `max_length`, `regex`
- Numeric: `min`, `max`, `min_exclusive`, `max_exclusive`, `is_in`
- Datetime: `min`/`max` bounds, `resolution`, `time_zone`, `time_unit`
- Nullability: `expr.is_not_null()` by default unless `nullable=True`

### Custom Rules

```python
class MySchema(dy.Schema):
    score = dy.Float64()

    @dy.rule()
    def score_is_positive(self):
        return pl.col("score") > 0

    @dy.rule(group_by="primary_key")
    def grouped_check(self):
        return ...
```

### Key APIs

```python
Schema.validate(df, cast=False)    # Raises on failure
Schema.is_valid(df)                # Returns bool
Schema.filter(df)                  # Returns (valid_df, FailureInfo)
Schema.serialise()                 # JSON string
Schema.columns()                   # Metadata
Schema.column_names()              # List[str]
deserialize_schema(json)           # Reconstruct from JSON
@rule                              # Custom validation decorator
```

---

## Summary: Key Differences

| Aspect | SchemaShift (current) | Dataframely |
|---|---|---|
| Schema definition | Pydantic model (`TargetSchema`) | Class-based (`dy.Schema`) |
| Validation granularity | Column-level (aggregated) | Row-level + column-level |
| Row-level failure details | None | Full (`FailureInfo`) |
| Constraint types | Name + dtype + nullability only | Rich (min/max, regex, length, custom rules) |
| Serialisation | JSON/YAML (FormatConfig) | JSON (schema.serialise()) |
| Custom rules | None | `@rule` decorator with Polars exprs |
| Transformation | Full DSL → Polars expr pipeline | Not in scope |
| Format detection | Registry + LLM | Not in scope |
