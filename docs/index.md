# schemashift

Enterprise software deployments often depend on loading canonical datasets from client source systems — but
those systems export whatever they want: third-party flat files, formats you've never seen before,
and arbitrary Excel workbooks. Wiring each one up by hand means bespoke pandas or Polars code for
every integration, every time.

**schemashift** solves this with a **declarative transform format** and a **safe expression DSL** designed
around three goals:

1. **Robustness** — strong types, schema validation, and end-to-end checks catch problems before data
   reaches your application.
2. **LLM-friendly syntax** — the DSL mirrors Polars expressions but is a closed language with no
   arbitrary code. An LLM can write a correct transform in one shot, with (ideally) far fewer tokens than
   generating pure Python, and the result is always safe to execute.
3. **Similarity-aware** — transforms are structured definitions, so similarity analysis against existing
   specs is straightforward, thereby saving time and tokens when an almost-familiar format reappears.

When a new format arrives with no matching transform, `smart_transform()` sends the file headers and your
dataset schema to your LLM, validates the generated transform end-to-end, and saves it to the registry
so the next run is instant.

---

## Thirty-second example

**1. Define what you want out:**

This is your canonical "result" format that you only need to define once per dataset.
In the future `type` may also be various `Enum`s that you define or other custom types.

```yaml
# examples/schemas/bank_statement.yaml
name: bank_statement
columns:
  transaction_id:
    type: string
    nullable: false
  posted_at:
    type: date
    nullable: false
  description:
    type: string
    nullable: false
  amount:
    type: number
    nullable: false
  currency:
    type: string
    nullable: false
  account_id:
    type: string
    nullable: false
  data_source:
    type: string
    nullable: false
```

**2. Write a transform for one source format:**

Riverbank provides debit and credit columns separately:

```json
{
  "name": "riverbank_statement",
  "description": "Riverbank current account statement export",
  "schema_name": "bank_statement",
  "columns": [
    { "target": "transaction_id", "source": "Ref" },
    { "target": "posted_at", "expr": "col('Date').str.to_datetime('%Y-%m-%d')", "dtype": "date" },
    { "target": "description", "source": "Details" },
    {
      "target": "amount",
      "expr": "col('Credit').cast('float64').fill_null(0) - col('Debit').cast('float64').fill_null(0)",
      "dtype": "number"
    },
    { "target": "currency", "constant": "GBP" },
    { "target": "account_id", "constant": "RIVER-001" },
    { "target": "data_source", "constant": "riverbank" }
  ]
}
```

The point is that JSON transforms are simple enough for an LLM to infer, write, and validate — and executing the result is a single tool call.

**3. Transform:**

```python
import schemashift as ss

registry = ss.FileSystemRegistry("./examples/transforms/")
schema = ss.DatasetSchema.from_yaml("examples/schemas/bank_statement.yaml")
result = ss.smart_transform("riverbank_statement.csv", registry=registry, dataset_schema=schema)
```

## When a new format arrives

If the transform is saved to the Registry, it gets instantly re-loaded.
Otherwise, if an LLM is provided, it will be used to generate the transform and save it for next time.

```python
from langchain_anthropic import ChatAnthropic

schema = ss.DatasetSchema.from_yaml("examples/schemas/bank_statement.yaml")
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

result = ss.smart_transform(
    "metro_credit_statement.tsv",
    registry=registry,
    dataset_schema=schema,
    llm=llm,
    auto_register=True,   # saves the transform so next run hits the registry
)
```

::::{grid} 1 1 3 3
:gutter: 3
:class-container: sd-mt-4

:::{grid-item-card} Transform known formats
:link: getting-started
:link-type: doc

Register a transform once per source. Call `transform()` to apply it — returns a
`TransformResult` with valid rows and failure details. Pass `n_rows=N` to preview without reading the full file.
:::

:::{grid-item-card} Auto-detect from a registry
:link: getting-started
:link-type: doc

Point schemashift at a directory of transforms and a file. The detector matches on column fingerprints
and picks the right transform — or raises `AmbiguousFormatError` when the match is ambiguous.
:::

:::{grid-item-card} LLM-assisted generation
:link: user-guide/llm-generation
:link-type: doc

Unknown format? `smart_transform()` sends the file headers and dataset schema to your LLM, validates
the generated transform end-to-end, and optionally saves it to the registry for next time.
:::

::::

## Install

```bash
# Core library
pip install schemashift

# With LLM config generation
pip install "schemashift[llm]"
```

---

```{toctree}
:maxdepth: 1
:caption: Getting started

getting-started
```

```{toctree}
:maxdepth: 2
:caption: User guide

user-guide/config-format
user-guide/dsl
user-guide/llm-generation
user-guide/cli
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```
