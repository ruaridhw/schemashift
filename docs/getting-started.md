# Getting started

## Installation

```bash
# Core library
pip install schemashift

# With LLM config generation
pip install "schemashift[llm]"
```

Requires Python 3.12+.

## Core concepts

schemashift has three objects you'll use in every pipeline:

**`TargetSchema`** — the shape you want the output to have (column names, types, required flags). Defined once in YAML, reused across all source configs.

**`FormatConfig`** — describes how to turn one specific source file into the target schema. Lives in a JSON file. Each column mapping uses exactly one of `source` (rename), `expr` (DSL expression), or `constant` (literal value).

**`Registry`** — a collection of `FormatConfig` objects. `FileSystemRegistry` reads JSON files from a directory. `DictRegistry` is for in-memory/testing use.

## Step-by-step setup

### 1. Define a target schema

Create a YAML file describing the canonical output you want every source to produce:

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
  - name: step_sequence
    type: int32
    required: false
  - name: tool_id
    type: str
    required: true
  - name: track_in_time
    type: datetime
    required: true
  - name: track_out_time
    type: datetime
    required: false
  - name: recipe
    type: str
    required: true
  - name: route
    type: str
    required: false
  - name: priority
    type: int32
    required: false
  - name: hold_flag
    type: bool
    required: true
  - name: data_source
    type: str
    required: true
```

Load it in Python:

```python
import schemashift as ss

schema = ss.TargetSchema.from_yaml("schemas/lot_movement.yaml")
```

### 2. Write a config for a source format

Each MES or ERP system exports lot data in a different shape. Here's a config for a Camstar MES CSV export:

```json
{
  "name": "camstar_mes",
  "target_schema": "lot_movement",
  "columns": [
    { "target": "lot_id",        "source": "LOT_ID" },
    { "target": "wafer_count",   "source": "QTY", "dtype": "int32" },
    { "target": "operation",     "source": "CURRENT_OPER" },
    { "target": "step_sequence", "source": "OPER_SEQ", "dtype": "int32" },
    { "target": "tool_id",       "source": "RESOURCE" },
    { "target": "track_in_time", "expr": "col(\"TRACKIN_DT\").str.to_datetime(\"%Y-%m-%d %H:%M:%S\")" },
    { "target": "track_out_time","expr": "col(\"TRACKOUT_DT\").str.to_datetime(\"%Y-%m-%d %H:%M:%S\")" },
    { "target": "recipe",        "source": "RECIPE_NAME" },
    { "target": "route",         "source": "FLOW" },
    { "target": "priority",      "source": "LOT_PRIORITY", "dtype": "int32" },
    { "target": "hold_flag",     "expr": "col(\"HOLD_STATUS\") != \"NONE\"" },
    { "target": "data_source",   "constant": "camstar_mes" }
  ]
}
```

Save this as `configs/camstar_mes.json`.

### 3. Transform a file

```python
registry = ss.FileSystemRegistry("./configs/")
config = registry.get("camstar_mes")

df = ss.transform("camstar_mes.csv", config)   # polars.DataFrame
```

Pass `n_rows=N` to preview the first N rows without reading the whole file.

### 4. Auto-detect the format

Once you have multiple configs registered (e.g. `camstar_mes`, `fabx_tsv`, `sap_erp`), let schemashift pick the right one based on column fingerprinting:

```python
df = ss.smart_transform("camstar_mes.csv", registry=registry)
```

The detector matches on the file's column names. If two configs both match, `AmbiguousFormatError` is raised — add more columns to one of the configs to disambiguate.

### 5. Validate the output

```python
schema.validate_eager(df)   # checks column names, dtypes, and required-column nulls
```

## Next steps

- {doc}`user-guide/config-format` — full reference for `FormatConfig` fields
- {doc}`user-guide/dsl` — expression DSL for column transformations
- {doc}`user-guide/llm-generation` — auto-generate configs for unknown formats
- {doc}`user-guide/cli` — use schemashift from the command line
