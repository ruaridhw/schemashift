# Config format reference

A `FormatConfig` is a JSON (or YAML) file that describes how to transform one specific source format into your target schema.

## Full structure

```json
{
  "name": "camstar_mes",
  "description": "Camstar MES lot movement export",
  "version": 1,
  "target_schema": "lot_movement",
  "reader": {
    "skip_rows": 0,
    "separator": ",",
    "encoding": "utf-8"
  },
  "columns": [
    { "target": "lot_id",        "source": "LOT_ID" },
    { "target": "wafer_count",   "source": "QTY", "dtype": "int32" },
    { "target": "track_in_time", "expr": "col(\"TRACKIN_DT\").str.to_datetime(\"%Y-%m-%d %H:%M:%S\")" },
    { "target": "hold_flag",     "expr": "col(\"HOLD_STATUS\") != \"NONE\"" },
    { "target": "data_source",   "constant": "camstar_mes" }
  ],
  "drop_unmapped": true
}
```

## Top-level fields

`name` _(required)_
: Unique identifier for this config within the registry. Used by `registry.get()` and for auto-detection.

`description`
: Optional human-readable description. Included in LLM-generated configs.

`version`
: Integer version number. Defaults to `1`.

`target_schema`
: Name of the `TargetSchema` this config produces. Used for validation.

`reader`
: Optional `ReaderConfig` controlling how the file is read. See [Reader options](#reader-options).

`columns` _(required)_
: List of `ColumnMapping` objects. See [Column mappings](#column-mappings).

`drop_unmapped`
: If `true`, columns not listed in `columns` are dropped from the output. Defaults to `true`.

## Column mappings

Each entry in `columns` must have a `target` field and **exactly one** of `source`, `expr`, or `constant`.

### Rename a column

```json
{ "target": "lot_id", "source": "LOT_ID" }
```

Renames the source column as-is. No type conversion.

### Apply a DSL expression

```json
{ "target": "track_in_time", "expr": "col(\"TRACKIN_DT\").str.to_datetime(\"%Y-%m-%d %H:%M:%S\")" }
```

Evaluates the expression and assigns the result to `target`. See {doc}`dsl` for the full expression reference.

### Set a constant

```json
{ "target": "data_source", "constant": "camstar_mes" }
```

Broadcasts a literal value to every row.

### Common optional fields

`dtype`
: Cast the result to this Polars dtype after the mapping. Accepted values: `str`, `int32`, `int64`, `float32`, `float64`, `bool`, `date`, `datetime`, `duration`.

`fillna`
: Fill nulls in the output column with this value after the mapping is applied.

## Reader options

The `reader` block controls low-level file reading.

`skip_rows`
: Number of rows to skip before the header. Default: `0`.

`sheet_name`
: For Excel files: sheet name (string) or 1-based sheet index (integer). Default: first sheet.

`separator`
: For CSV/TSV: field delimiter character. Default: `","`.

`encoding`
: File encoding. Default: `"utf-8"`.


## Supported file formats

| Format | Notes |
|--------|-------|
| `.csv` | Lazy scan via Polars |
| `.tsv` | CSV with `separator="\t"` |
| `.parquet` | Lazy scan via Polars |
| `.xlsx` / `.xls` | Via `fastexcel` (calamine engine) — read eagerly then lazy |
| `.json` | Read eagerly then lazy |

## Validation

Use `validate_config()` to check that all DSL expressions parse correctly without running against real data:

```python
import schemashift as ss

config = ss.FormatConfig.model_validate_json(open("my_config.json").read())
errors = ss.validate_config(config)
if errors:
    for err in errors:
        print(err)
```

Or from the CLI:

```bash
schemashift validate my_config.json
```
