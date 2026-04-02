# CLI reference

schemashift ships a `schemashift` CLI command.

## transform

Apply a config to a file and write the output.

```bash
schemashift transform <FILE> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | Path to a `FormatConfig` JSON file |
| `--registry DIR` | Path to a registry directory (auto-detect format) |
| `--output PATH` | Output file path (CSV, Parquet, or JSON by extension). Prints first 20 rows to stdout if omitted. |

Exactly one of `--config` or `--registry` must be provided.

**Examples:**

```bash
# Explicit config
schemashift transform camstar_mes.csv --config camstar_mes.json --output result.csv

# Auto-detect from registry
schemashift transform fabx.tsv --registry ./configs/ --output result.parquet
```

## validate

Check that a config file is valid — all DSL expressions parse, all required fields are present.

```bash
schemashift validate <CONFIG>
```

```bash
schemashift validate camstar_mes.json
# Config 'camstar_mes' is valid.
```

## dry-run

Apply a config to the first N rows of a file and print the result. Useful for checking a config before running against the full dataset.

```bash
schemashift dry-run <CONFIG> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--sample PATH` | File to run the dry-run against |
| `--rows N` | Number of rows to sample (default: 10) |

```bash
schemashift dry-run camstar_mes.json --sample camstar_mes.csv --rows 5
```

## generate

Generate a `FormatConfig` for an unknown file using an LLM.

Requires `pip install "schemashift[llm]"` and LLM credentials in the environment (or a `.env` file).

```bash
schemashift generate <FILE> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--target-schema PATH` | YAML `TargetSchema` to generate a config for (required) |
| `--registry DIR` | Save the generated config here |
| `--output PATH` | Save the generated config to this specific path |
| `--interactive` | Review the config before saving |
| `--retries N` | LLM retry attempts on validation failure (default: 2) |

```bash
# Generate and print
schemashift generate sap_erp.csv --target-schema schemas/lot_movement.yaml

# Generate and save to registry
schemashift generate sap_erp.csv \
    --registry ./configs/ \
    --target-schema schemas/lot_movement.yaml

# Generate with review step
schemashift generate sap_erp.csv \
    --registry ./configs/ \
    --target-schema schemas/lot_movement.yaml \
    --interactive
```

## list

List all configs registered in a directory.

```bash
schemashift list --registry ./configs/
```

```
camstar_mes   (v1)  — Camstar MES lot movement export
fabx_tsv      (v1)  — FabX custom WIP export (tab-delimited, Unix epoch timestamps)
sap_erp       (v1)  — SAP ERP lot data (semicolon-delimited, German headers)
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for direct Anthropic access |
| `FOUNDRY_API_KEY` | Azure AI Foundry API key |
| `FOUNDRY_ENDPOINT` | Full Azure AI Foundry endpoint URL |
| `FOUNDRY_RESOURCE` | Azure resource name (endpoint inferred if `FOUNDRY_ENDPOINT` not set) |
| `MODEL_NAME` | Model override for LLM generation (default: `claude-haiku-4-5`) |

A `.env` file in the current working directory is loaded automatically when the `generate` command runs.
