# Dataframely Integration Design

## Context

schemashift's current validation (`TargetSchema`) provides only column-level feedback: missing columns, dtype mismatches, and aggregate null counts. When a transform config produces bad data, users cannot see *which rows* broke or *why* without manual debugging. The goal is to integrate [dataframely](https://github.com/Quantco/dataframely) (`dy`) to get rich, row-level validation details for every failure — schema constraint violations and transform-induced errors alike — returned all at once in a single result object.

This is a breaking change. No migration path or backwards compatibility is needed.

## Design

### 1. SchemaConfig (replaces TargetSchema)

**New file: `src/schemashift/schema.py`**

`SchemaConfig` is a Pydantic model loadable from YAML/JSON that maps to dataframely column types and their constraints. It replaces both `TargetSchema` and `TargetColumn`.

```yaml
# Example schema.yaml
name: invoice_output
description: Canonical invoice format
columns:
  invoice_id:
    type: int64
    nullable: false
    primary_key: true
  vendor_name:
    type: string
    min_length: 1
    max_length: 255
  amount:
    type: float64
    min: 0
    nullable: false
  currency:
    type: string
    is_in: [USD, EUR, GBP]
  invoice_date:
    type: date
    min: "2020-01-01"
  notes:
    type: string
    nullable: true
```

**Pydantic models:**

- `ColumnConstraints` — per-column config with fields: `type` (DType, required), `nullable` (bool, default False), `primary_key` (bool, default False), `min`, `max`, `min_exclusive`, `max_exclusive`, `min_length`, `max_length`, `regex`, `is_in`, `description`.
- `SchemaConfig` — top-level with `name`, `description`, `columns: dict[str, ColumnConstraints]`.

**Factory function:**

`build_dy_schema(config: SchemaConfig) -> type[dy.Schema]` dynamically constructs a `dy.Schema` subclass using `type()` metaclass construction. Maps each column entry to the corresponding `dy.Int64(...)`, `dy.String(min_length=..., ...)`, etc.

**Type mapping:** `DType` string (e.g., `"int64"`) maps to both a Polars dtype and a dy column class. New lookup table in `schema.py`:

```python
DY_COLUMN_MAP: dict[str, type[dy.Column]] = {
    "int8": dy.Int8, "int16": dy.Int16, "int32": dy.Int32, "int64": dy.Int64,
    "uint8": dy.UInt8, "uint16": dy.UInt16, "uint32": dy.UInt32, "uint64": dy.UInt64,
    "float32": dy.Float32, "float64": dy.Float64,
    "str": dy.String, "string": dy.String, "utf8": dy.String,
    "bool": dy.Bool,
    "date": dy.Date, "datetime": dy.Datetime, "time": dy.Time, "duration": dy.Duration,
    "categorical": dy.Categorical, "binary": dy.Binary,
}
```

**Escape hatch:** Users who need custom `@dy.rule()` methods write a `dy.Schema` class directly and pass it in. The API accepts `SchemaConfig | type[dy.Schema]` anywhere a schema is needed. `resolve_schema()` normalises both to `type[dy.Schema]`.

### 2. TransformSpec (replaces FormatConfig)

**File: `src/schemashift/models.py`**

`FormatConfig` is renamed to `TransformSpec`. The `target_schema: str | None` field is replaced with a required `schema: SchemaConfig` field.

```python
class TransformSpec(BaseModel):
    name: str
    description: str = ""
    version: int = 1
    schema: SchemaConfig                    # Required, replaces target_schema
    reader: ReaderConfig = ReaderConfig()
    columns: list[ColumnMapping]
    drop_unmapped: bool = True
```

In YAML configs, the schema is defined inline within the transform spec:

```yaml
name: invoice_v2
schema:
  name: invoice_output
  columns:
    amount:
      type: float64
      min: 0
      nullable: false
    currency:
      type: string
      is_in: [USD, EUR, GBP]
columns:
  - target: amount
    source: Price
    dtype: float64
  - target: currency
    source: Currency Code
```

### 3. TransformResult and FailureInfo

**New file: `src/schemashift/result.py`**

#### FailureInfo

A unified failure container covering both transform-induced errors and schema constraint violations.

```python
@dataclass(frozen=True)
class FailureInfo:
    """Unified row-level failure details from transform + validation."""

    # From dy.Schema.filter() — row-level constraint violations
    schema_failures: dy.FailureInfo | None

    # DSL expressions that failed entirely — {column_name: error_message}
    expression_errors: dict[str, str]

    @property
    def invalid(self) -> pl.DataFrame | None:
        """Original invalid rows (without rule columns)."""

    @property
    def details(self) -> pl.DataFrame | None:
        """Invalid rows with per-rule valid/invalid/unknown status."""

    @property
    def counts(self) -> dict[str, int]:
        """Failure counts per rule/error. Expression errors keyed as 'expression_error:{col}'."""

    @property
    def has_failures(self) -> bool:
        """True if any expression errors or schema violations exist."""
```

#### TransformResult

```python
@dataclass(frozen=True)
class TransformResult:
    valid: pl.DataFrame      # Rows that passed all dy rules
    failures: FailureInfo    # All failure details (always present)

    @property
    def all_valid(self) -> bool:
        return not self.failures.has_failures
```

### 4. Transform Pipeline

**File: `src/schemashift/transform.py`**

#### New `transform()` signature

```python
def transform(
    path: Path,
    config: TransformSpec,
    schema: SchemaConfig | type[dy.Schema] | None = None,  # Override config.schema
    *,
    strict: bool = False,
    n_rows: int | None = None,
) -> TransformResult:
```

- Schema resolution order: explicit `schema` param > `config.schema`. Since `config.schema` is required on `TransformSpec`, a schema is always available. The `schema` param exists only to override (e.g., passing a `dy.Schema` class with custom rules).
- `strict=True` raises `SchemaValidationError` (with `.failures: FailureInfo`) if any failures exist.
- `strict=False` always returns `TransformResult`.

#### Pipeline flow

```
1. Read file → LazyFrame
2. For each ColumnMapping:
     try: build expression (lenient casts: strict=False)
     except: record in expression_errors, produce null column
3. Apply expressions → .collect() → DataFrame
4. Resolve schema: SchemaConfig → build_dy_schema() or use dy.Schema directly
5. dy_schema.filter(df) → (valid_df, dy.FailureInfo)
6. Build FailureInfo(schema_failures=dy_failures, expression_errors=errors)
7. Return TransformResult(valid=valid_df, failures=failure_info)
8. If strict and failures → raise SchemaValidationError(failures=failure_info)
```

Key change: casts in `_mapping_to_expr()` use `strict=False` so partial failures produce nulls instead of exceptions. The nulls then get caught by dy's validation as row-level failures.

### 5. Error Hierarchy

**File: `src/schemashift/errors.py`**

```
SchemaShiftError
├── ConfigValidationError
├── DSLSyntaxError                 # unchanged
├── DSLRuntimeError                # unchanged
├── SchemaValidationError          # updated: carries .failures: FailureInfo
├── FormatDetectionError           # unchanged
│   └── AmbiguousFormatError       # unchanged
├── ReviewRejectedError            # unchanged
├── LLMGenerationError             # unchanged
├── UnsupportedFileError           # unchanged
└── ReaderError                    # unchanged
```

`SchemaValidationError` gains `.failures: FailureInfo` so strict-mode callers can inspect row-level details in their except handler.

`TransformError` is removed — expression failures are captured in `FailureInfo.expression_errors` instead.

### 6. Orchestration

**File: `src/schemashift/orchestration.py`**

`smart_transform()` updated to return `TransformResult` instead of `pl.DataFrame`. The two-phase `validate_lazy()` + `validate_eager()` calls are removed — replaced by the single `dy.Schema.filter()` inside `transform()`.

```python
def smart_transform(
    path: Path,
    registry: Registry,
    schema: SchemaConfig | type[dy.Schema] | None = None,
    *,
    strict: bool = False,
    # ... same LLM/review params ...
) -> TransformResult:
```

### 7. Files Changed

| File | Action |
|---|---|
| `src/schemashift/schema.py` | NEW — SchemaConfig, ColumnConstraints, build_dy_schema(), resolve_schema() |
| `src/schemashift/result.py` | NEW — TransformResult, FailureInfo |
| `src/schemashift/models.py` | MODIFY — rename FormatConfig → TransformSpec, replace target_schema with schema: SchemaConfig |
| `src/schemashift/transform.py` | MODIFY — return TransformResult, lenient casts, dy.Schema.filter() integration |
| `src/schemashift/orchestration.py` | MODIFY — return TransformResult, remove validate_lazy/validate_eager calls |
| `src/schemashift/errors.py` | MODIFY — SchemaValidationError gains .failures |
| `src/schemashift/target_schema.py` | REMOVE |
| `src/schemashift/dtypes.py` | MODIFY — add DY_COLUMN_MAP or move to schema.py |
| `src/schemashift/llm.py` | MODIFY — update to use TransformSpec and SchemaConfig |
| `src/schemashift/detection.py` | MODIFY — update to use TransformSpec |
| `src/schemashift/registry.py` | MODIFY — update to use TransformSpec |
| `pyproject.toml` | MODIFY — add `dataframely` dependency |
| `tests/` | MODIFY — update all tests for new types and return values |

### 8. DSL

The DSL (`dsl/`) remains unchanged in scope — it handles column transformations only. The one behavioural change: casts produced by `_mapping_to_expr()` use `strict=False` to support lenient evaluation. Validation rules are expressed through `dy.Schema` constraints, not the DSL.

Future consideration: DSL could be extended to express validation predicates (compiled to `@dy.rule()` functions), but this is out of scope for this design.

## Verification

1. **Unit tests:** SchemaConfig round-trips through YAML → Pydantic → dy.Schema. Constraints (min, max, regex, is_in, nullable) map correctly.
2. **Transform tests:** A TransformSpec with intentional partial failures (bad casts, out-of-range values) returns TransformResult with correct valid/invalid row split and meaningful FailureInfo.
3. **Strict mode:** `strict=True` raises SchemaValidationError with FailureInfo attached.
4. **Expression error capture:** A TransformSpec referencing a nonexistent column produces TransformResult with expression_errors populated (not a raised exception).
5. **dy.Schema escape hatch:** Passing a custom dy.Schema class with `@rule` methods works end-to-end.
6. **Orchestration:** `smart_transform()` returns TransformResult with full failure details.
7. **Existing DSL tests:** All DSL parser/compiler tests pass unchanged.
