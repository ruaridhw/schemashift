# schemashift

Enterprise software deployments often depend on loading canonical datasets from client source systems — but
those systems export whatever they want: third-party flat files, formats you've never seen before,
and arbitrary Excel workbooks. Wiring each one up by hand means bespoke pandas or Polars code for
every integration, every time.

**schemashift** solves this with a **declarative config format** and a **safe expression DSL** designed
around three goals:

1. **Robustness** — strong types, schema validation, and end-to-end checks catch problems before data
   reaches your application.
2. **LLM-friendly syntax** — the DSL mirrors Polars expressions but is a closed language with no
   arbitrary code. An LLM can write a correct transformation in one shot, with (ideally) far fewer tokens than
   generating pure Python, and the result is always safe to execute.
3. **Similarity-aware** — transformations are structured definitions, so similarity analysis against existing
   configs is straightforward, thereby saving time and tokens when an almost-familiar format reappears.

When a new format arrives with no matching config, `smart_transform()` sends the file headers and your
target schema to your LLM, validates the generated config end-to-end, and saves it to the registry
so the next run is instant.

---

## Thirty-second example

**1. Define what you want out:**

This is your canonical "result" format that you only need to define once per dataset.
In the future `type` may also be various `Enum`s that you define or other custom types.

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

This is the definition of how we get from a given input format to the result you defined above.

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

The point is that JSON configs are simple enough for an LLM to infer, write, and validate — and executing the result is a single tool call.

**3. Transform:**

```python
import schemashift as ss

registry = ss.FileSystemRegistry("./configs/")
df = ss.smart_transform("camstar_mes.csv", registry=registry)
```

## When a new format arrives

If the transformation is saved to the Registry, it gets instantly re-loaded.
Otherwise, if an LLM is provided, it will be used to generate the transformation and save it for next time.

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
)
```

::::{grid} 1 1 3 3
:gutter: 3
:class-container: sd-mt-4

:::{grid-item-card} Transform known formats
:link: getting-started
:link-type: doc

Register a config once per source. Call `transform()` to apply it — returns a
`polars.DataFrame`. Pass `n_rows=N` to preview without reading the full file.
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
