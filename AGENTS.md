# AGENTS.md

Instructions for coding agents working in this repository.

## Preferred tools

- Use `rg` for fast text and file search instead of slower defaults like `grep` or `find` when available.
- Use `rtk` when reading large files, listing directories, summarising command output, or running noisy commands that would otherwise waste context.
- Good defaults in this repo:
  - `rg "pattern" src tests`
  - `rg --files src tests docs`
  - `rtk read README.md`
  - `rtk tree src`
  - `rtk pytest`
  - `rtk ruff check src tests`

## Commands

```bash
rtk uv run pytest                              # all tests with coverage
rtk uv run pytest tests/test_dsl_parser.py -v  # single file
rtk uv run pytest -k "test_name"               # single test by name
rtk uv run ruff check src/                     # lint
rtk uv run black src/ tests/            # formatting check
pre-commit run                                 # lint + format (run after staging changes)
```

## Architecture

`schemashift` transforms tabular files into a canonical schema using declarative JSON/YAML configs.

Pipeline:

```text
File → Reader → LazyFrame → Transform Engine → LazyFrame
                    ↑
              Registry / Detector
                    ↑ (miss)
              LLM Generator
```

## Important concepts

### Config model

`FormatConfig` is the central object. Each `ColumnMapping` must define exactly one of:

- `source` — rename a column
- `expr` — DSL expression string compiled at transform time
- `constant` — literal value broadcast to all rows

`FormatConfig.source_columns()` extracts referenced source columns for format detection.

### DSL

The DSL is a closed language: string → AST → `polars.Expr`.

- `parser.py` is a hand-written recursive descent parser with explicit allowlists.
- `ast_nodes.py` defines frozen AST dataclasses.
- `compiler.py` lowers AST nodes into native Polars expressions.

When adding a DSL operation:

1. Add it to the allowlist in `parser.py`
2. Add compilation support in `compiler.py`
3. Add it to `_DSL_REFERENCE` in `llm.py`

Never use `eval()` or introduce arbitrary method dispatch.

### Transform engine

- `transform(path, config)` returns a `pl.LazyFrame`
- Avoid materialising data unless `dry_run()` is required or the caller explicitly collects
- `smart_transform()` handles detect-or-generate flow and optional review / auto-registration

### Registry

- `DictRegistry` is in-memory
- `FileSystemRegistry` stores one JSON file per config
- `FileSystemRegistry.load_schema()` looks for target schemas in a `schemas/` subdirectory

### LLM generation

`generate_config()` accepts a LangChain `BaseChatModel`, validates generated JSON, validates DSL syntax, and dry-runs before success.

### Target schema

`TargetSchema` is the source of truth for output shape:

- `validate_lazy()` checks names and dtypes without collecting
- `validate_eager()` additionally checks nulls in required columns

## Key constraints

- Keep format-specific logic in `readers.py`; the transform engine should only work with `pl.LazyFrame`
- Config validation and output-schema validation are different concerns
- Excel uses `fastexcel` / calamine semantics; integer sheet references map to `sheet_id` (1-based)
- Unix timestamp to Polars `Datetime` requires microseconds (`1_000_000` multiplier), not nanoseconds

## Errors

All project-specific errors inherit from `SchemaShiftError`.

Important ones:

- `DSLSyntaxError`
- `DSLRuntimeError`
- `AmbiguousFormatError`
- `LLMGenerationError`

For deeper project-specific guidance, see `CLAUDE.md`.
