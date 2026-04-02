# schemashift

**Declarative file format transformer** — config-driven column mappings with a safe expression DSL.

Turn any tabular file (CSV, XLSX, Parquet, JSON) into a canonical schema using a single JSON config
per source format. When a new format arrives with no matching config, an LLM generates one automatically.

---

::::{grid} 1 1 3 3
:gutter: 3
:class-container: sd-mt-4

:::{grid-item-card} Transform known formats
:link: getting-started
:link-type: doc

Register a config once per source. Call `transform()` or `auto_transform()` to apply it — always
returning a lazy Polars frame, never loading more than you need.
:::

:::{grid-item-card} Auto-detect from a registry
:link: getting-started
:link-type: doc

Point schemashift at a directory of configs and a file. The detector matches on column fingerprints
and picks the right config — or raises `AmbiguousFormatError` when the match is ambiguous.
:::

:::{grid-item-card} LLM-assisted generation
:link: user-guide/llm-generation
:link-type: doc

Unknown format? `smart_transform()` sends the file headers and target schema to your LLM, validates
the generated config end-to-end, and optionally saves it to the registry for next time.
:::

::::

## Install

```bash
# Core library
pip install schemashift

# With LLM config generation
pip install "schemashift[llm]"
```

## Thirty-second example

**1. Define what you want out:**

```yaml
# schemas/lot_movement.yaml
name: lot_movement
columns:
  - name: lot_id
    type: str
    required: true
  - name: wafer_count
    type: int32
    required: true
  - name: operation
    type: str
    required: true
  - name: track_in_time
    type: datetime
    required: true
  - name: hold_flag
    type: bool
    required: true
  - name: data_source
    type: str
    required: true
```

**2. Write a config for one source format:**

```json
{
  "name": "camstar_mes",
  "columns": [
    { "target": "lot_id",        "source": "LOT_ID" },
    { "target": "wafer_count",   "source": "QTY", "dtype": "int32" },
    { "target": "operation",     "source": "CURRENT_OPER" },
    { "target": "track_in_time", "expr": "col(\"TRACKIN_DT\").str.to_datetime(\"%Y-%m-%d %H:%M:%S\")" },
    { "target": "hold_flag",     "expr": "col(\"HOLD_STATUS\") != \"NONE\"" },
    { "target": "data_source",   "constant": "camstar_mes" }
  ]
}
```

**3. Transform:**

```python
import schemashift as ss

registry = ss.FileSystemRegistry("./configs/")
df = ss.auto_transform("camstar_mes.csv", registry=registry).collect()
```

## When a new format arrives

```python
from langchain_anthropic import ChatAnthropic

schema = ss.TargetSchema.from_yaml("schemas/lot_movement.yaml")
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

df = ss.smart_transform(
    "fabx.tsv",
    registry=registry,
    target_schema=schema,
    llm=llm,
    auto_register=True,   # saves the config so next run hits the registry
).collect()
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
