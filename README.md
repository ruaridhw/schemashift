# schemashift

Declarative file format transformer — config-driven column mappings with a safe expression DSL.

Transform tabular files (CSV, XLSX, Parquet, JSON, TSV) into a canonical schema using a single JSON config per source format. When encountering an unknown format, an LLM generates the config automatically.

## Installation

```bash
# Core library
pip install schemashift

# With LLM config generation
pip install "schemashift[llm]"
```

## Quick start

### 1. Define a target schema

```yaml
# schemas/certificates.yaml
name: canonical_certificate
columns:
  - name: certificate_id
    type: str
    required: true
  - name: volume_mwh
    type: float64
    required: true
  - name: issue_date
    type: datetime
    required: true
  - name: technology
    type: str
    required: true
  - name: data_source
    type: str
    required: true
```

### 2. Write a config for a known source format

```json
{
  "name": "provider_x_certificates",
  "columns": [
    { "target": "certificate_id", "source": "Cert. ID" },
    { "target": "volume_mwh", "expr": "col(\"Volume (kWh)\") / 1000" },
    { "target": "issue_date", "expr": "col(\"Issue Date\").str.to_datetime(\"%Y-%m-%d\")" },
    { "target": "technology", "source": "Tech Type" },
    { "target": "data_source", "constant": "provider_x" }
  ]
}
```

### 3. Transform

```python
import schemashift as ss

registry = ss.FileSystemRegistry("./configs/")
result = ss.transform("data.csv", registry.get("provider_x_certificates"))
result.collect()  # returns a polars.DataFrame
```

Or auto-detect the format from the registry:

```python
result = ss.auto_transform("data.csv", registry=registry)
```

## LLM-assisted config generation

When a file arrives from a new source with no matching config, `smart_transform` generates one automatically:

```python
import schemashift as ss

schema = ss.TargetSchema.from_yaml("schemas/certificates.yaml")
registry = ss.FileSystemRegistry("./configs/")

# bring your own LangChain-compatible LLM:
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# or via Azure AI Foundry:
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
llm = AzureAIChatCompletionsModel(
    endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>",
    credential="<FOUNDRY_API_KEY>",
    model_name="claude-haiku-4-5",
)

result = ss.smart_transform(
    "unknown_source.csv",
    registry=registry,
    target_schema=schema,
    llm=llm,
    auto_register=True,   # saves the generated config for next time
)
```

With a human review step:

```python
def review(config, sample_df):
    print(config.model_dump_json(indent=2))
    print(sample_df)
    return config  # return None to reject

result = ss.smart_transform(..., review_fn=review)
```

## CLI

```bash
# Transform with an explicit config
schemashift transform data.csv --config provider_x.json --output result.csv

# Auto-detect format from registry
schemashift transform data.csv --registry ./configs/ --output result.csv

# Validate a config
schemashift validate provider_x.json

# Dry-run a config against sample data (first 10 rows)
schemashift dry-run provider_x.json --sample data.csv

# Generate a config for an unknown file (requires LLM credentials — see below)
schemashift generate data.csv --target-schema schemas/certificates.yaml --output new_config.json

# Generate with interactive review before saving
schemashift generate data.csv --registry ./configs/ --interactive

# List registered configs
schemashift list --registry ./configs/
```

### LLM credentials

The `generate` command auto-detects credentials from environment variables (a `.env` file in the working directory is loaded automatically):

| Priority | Variables | Provider |
|----------|-----------|----------|
| 1 | `FOUNDRY_API_KEY` + `FOUNDRY_ENDPOINT` (or `FOUNDRY_RESOURCE`) | Azure AI Foundry |
| 2 | `ANTHROPIC_API_KEY` | Anthropic |

**Azure AI Foundry** (e.g. an Azure AI project running Claude):

```bash
FOUNDRY_API_KEY=<key>
FOUNDRY_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
MODEL_NAME=claude-haiku-4-5   # optional, this is the default
```

If you only set `FOUNDRY_RESOURCE` (without `FOUNDRY_ENDPOINT`), the endpoint is inferred as
`https://<FOUNDRY_RESOURCE>.services.ai.azure.com/api/projects/<FOUNDRY_RESOURCE>`.

**Anthropic (direct)**:

```bash
ANTHROPIC_API_KEY=<key>
```

## Expression DSL

Column mappings support a safe, closed expression language that compiles to native Polars expressions:

```
col("Volume (kWh)") / 1000                          # arithmetic
col("Name").str.strip().str.lower()                 # string ops
col("Date").str.to_datetime("%Y-%m-%d")             # date parsing
col("dt").dt.year()                                 # date extraction
when(col("Type") == "solar", "Solar").otherwise("Other")   # conditionals
when(col("T") == "A", "A").when(col("T") == "B", "B").otherwise("C")
coalesce(col("A"), col("B"), "fallback")            # first non-null
col("x").cast("float64")                            # type casting
col("Code").str.replace_regex("\\d+", "NUM")        # regex replace
```

No `eval()`, no arbitrary Python — only the explicitly allowlisted operations above.

## Config reference

```json
{
  "name": "my_format",
  "description": "Optional description",
  "version": 1,
  "target_schema": "canonical_certificate",
  "reader": {
    "skip_rows": 0,
    "sheet_name": "Sheet1",
    "separator": ",",
    "encoding": "utf-8"
  },
  "columns": [
    { "target": "out_col", "source": "SourceCol" },
    { "target": "out_col", "expr": "col(\"X\") / 1000", "dtype": "float64" },
    { "target": "out_col", "constant": "fixed_value", "fillna": "unknown" }
  ],
  "drop_unmapped": true
}
```

Each column mapping requires exactly one of `source`, `expr`, or `constant`. The `dtype` field casts the result; `fillna` fills nulls after the mapping is applied.

## Supported file formats

| Format | Reader |
|--------|--------|
| CSV | `pl.scan_csv` (lazy) |
| TSV | `pl.scan_csv` with `separator="\t"` |
| Parquet | `pl.scan_parquet` (lazy) |
| XLSX / XLS | `pl.read_excel` via fastexcel (eager, then lazy) |
| JSON | `pl.read_json` (eager, then lazy) |
