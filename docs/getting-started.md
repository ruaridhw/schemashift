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

**`DatasetSchema`** — the validated dataset contract: column names, types, nullability, and optional constraints. Defined once in YAML, reused across source transforms.

**`TransformSpec`** — describes how to turn one specific source file into the dataset schema. Lives in a JSON file under a `transforms/` directory. Each column mapping uses exactly one of `source` (rename), `expr` (DSL expression), or `constant` (literal value).

**`Registry`** — a collection of `TransformSpec` objects. `FileSystemRegistry` reads JSON transform files from a directory. `DictRegistry` is for in-memory/testing use.

## Step-by-step setup

### 1. Define a dataset schema

Create a YAML file describing the dataset each source should produce:

```yaml
# schemas/lot_movement.yaml
name: lot_movement
columns:
  lot_id:
    type: str
    nullable: false
  wafer_count:
    type: int32
    nullable: false
  operation:
    type: str
    nullable: false
  step_sequence:
    type: int32
    nullable: true
  tool_id:
    type: str
    nullable: false
  track_in_time:
    type: datetime
    nullable: false
  track_out_time:
    type: datetime
    nullable: true
  recipe:
    type: str
    nullable: false
  route:
    type: str
    nullable: true
  priority:
    type: int32
    nullable: true
  hold_flag:
    type: bool
    nullable: false
  data_source:
    type: str
    nullable: false
```

Load it in Python:

```python
import schemashift as ss

schema = ss.DatasetSchema.from_yaml("schemas/lot_movement.yaml")
```

### 2. Write a config for a source format

Each MES or ERP system exports lot data in a different shape. Here's a config for a Camstar MES CSV export:

```json
{
  "name": "camstar_mes",
  "schema_name": "lot_movement",
  "columns": [
    { "target": "lot_id",        "source": "LOT_ID" },
    { "target": "wafer_count",   "source": "QTY", "dtype": "int32" },
    { "target": "operation",     "source": "CURRENT_OPER" },
    { "target": "step_sequence", "source": "OPER_SEQ", "dtype": "int32" },
    { "target": "tool_id",       "source": "RESOURCE" },
    { "target": "track_in_time", "expr": "col('TRACKIN_DT').str.to_datetime('%Y-%m-%d %H:%M:%S')" },
    { "target": "track_out_time","expr": "col('TRACKOUT_DT').str.to_datetime('%Y-%m-%d %H:%M:%S')" },
    { "target": "recipe",        "source": "RECIPE_NAME" },
    { "target": "route",         "source": "FLOW" },
    { "target": "priority",      "source": "LOT_PRIORITY", "dtype": "int32" },
    { "target": "hold_flag",     "expr": "col('HOLD_STATUS') != 'NONE'" },
    { "target": "data_source",   "constant": "camstar_mes" }
  ]
}
```

Save this as `transforms/camstar_mes.json`.

### 3. Transform a file

```python
registry = ss.FileSystemRegistry("./transforms/")
config = registry.get("camstar_mes")

schema = ss.DatasetSchema.from_yaml("schemas/lot_movement.yaml")
result = ss.transform("camstar_mes.csv", config, dataset_schema=schema)
df = result.valid
```

Pass `n_rows=N` to preview the first N rows without reading the whole file.

### 4. Auto-detect the format

Once you have multiple transforms registered (e.g. `camstar_mes`, `fabx_tsv`, `sap_erp`), let schemashift pick the right one based on column fingerprinting:

```python
result = ss.smart_transform("camstar_mes.csv", registry=registry, dataset_schema=schema)
```

The detector matches on the file's column names. If two transforms both match, `AmbiguousFormatError` is raised — add more source columns to one of the transforms to disambiguate.

### 5. Validate the output

Validation runs inside `transform()` and `smart_transform()` when you pass a `DatasetSchema`. Valid rows are returned in `result.valid`; row-level failures are available through `result.failures`.

## Next steps

- {doc}`user-guide/config-format` — full reference for `TransformSpec` fields
- {doc}`user-guide/dsl` — expression DSL for column transformations
- {doc}`user-guide/llm-generation` — auto-generate configs for unknown formats
- {doc}`user-guide/cli` — use schemashift from the command line
