# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tech Stack

- **Language:** Python 3.12+
- **Data processing:** Polars
- **Validation:** Pydantic v2
- **Config format:** YAML (PyYAML)
- **CLI framework:** Click
- **Excel support:** fastexcel
- **Optional:** langchain-core (for LLM features, install with `[llm]` extra)

## Commands

```bash
uv run pytest                        # all tests with coverage
uv run pytest tests/test_dsl_parser.py -v   # single file
uv run pytest -k "test_name"         # single test by name
uv run black .                       # lint
uv run ruff check src/               # lint
pre-commit run                       # lint + format (run after staging changes)
```

## Architecture

schemashift transforms tabular files into a canonical schema using declarative JSON/YAML configs. The pipeline is:

```
File → Reader → LazyFrame → Transform Engine → LazyFrame
                    ↑
              Registry / Detector
                    ↑ (miss)
              LLM Generator
```

### Config model (`models.py`)

`FormatConfig` is the central object. Each `ColumnMapping` must have **exactly one** of:
- `source` — rename a column
- `expr` — DSL expression string (compiled at transform time)
- `constant` — literal value broadcast to all rows

`FormatConfig.source_columns()` extracts referenced source column names (used by the detector).

### DSL (`dsl/`)

Three-layer pipeline: string → AST → `polars.Expr`.

- `parser.py` — hand-written recursive descent. Allowlist of methods in `_STR_METHODS`, `_DT_METHODS`, `_DIRECT_METHODS`. Add new ops here first.
- `ast_nodes.py` — frozen dataclasses (`ColRef`, `BinaryOp`, `MethodCall`, `WhenChain`, `Coalesce`, etc.)
- `compiler.py` — `compile_dsl()` uses `match/case` to dispatch each node type to a `polars.Expr`. `_CAST_TYPES` maps dtype strings to Polars types.
- `_lookups.py` — `lookup()` and `custom_lookup()` table-driven mapping functions.

To add a new DSL operation: add to the allowlist in `parser.py`, add a case in `compiler.py`, update the DSL reference in `dsl/__init__.py` docstring.

The DSL is a **closed language** — no eval, no arbitrary method calls. The parser raises `DSLSyntaxError` for anything not explicitly allowlisted.

### Transform engine (`transform.py`)

`transform(path, config)` → `pl.LazyFrame`. Never materialises data unless `dry_run()` or the caller calls `.collect()`.

`smart_transform()` is the full detect-or-generate flow: registry hit → apply directly; miss → call `llm.generate_config()` → optional `review_fn` callback → optional `auto_register`.

### Detection (`detection.py`)

Fingerprints an input file's columns against registered configs. Called by `smart_transform()` on a registry miss. Uses `FormatConfig.source_columns()` to build the match signature.

### Registry (`registry.py`)

`DictRegistry` (in-memory) and `FileSystemRegistry` (JSON files per config in a directory). `FileSystemRegistry.load_schema()` looks for a `TargetSchema` in a `schemas/` subdirectory — convention used by the CLI.

### LLM generation (`llm.py`)

`generate_config()` accepts a LangChain `BaseChatModel` only (no `llm_fn`). Retry loop (default 2 retries): extract JSON → `FormatConfig.model_validate` → `validate_config` (DSL parse check) → `dry_run`. Each failed attempt is logged at `WARNING` and stored in `LLMGenerationError.attempts`.

### Target schema (`target_schema.py`)

`TargetSchema` is the source of truth for output shape. `validate_lazy()` checks column names/dtypes against the `LazyFrame` schema without collecting. `validate_eager()` also checks for nulls in required columns.

### Error hierarchy

All errors inherit from `SchemaShiftError`. Key ones: `DSLSyntaxError` (parse-time, has `.expression` and `.position`), `DSLRuntimeError` (evaluation-time), `AmbiguousFormatError` (has `.candidates`), `LLMGenerationError` (has `.attempts`).

### Shared dtypes (`dtypes.py`)

Single source of truth for the string-alias → Polars dtype map, shared between the DSL compiler and the target schema validator.

## Key design constraints

- The transform engine only ever sees a `pl.LazyFrame` — format-specific logic stays in `readers.py`.
- `TargetSchema` validates the *output* of a config, not the config itself. A syntactically valid config can still produce schema-failing output.
- Excel files use `fastexcel` (calamine engine). Integer `sheet_name` values are passed as `sheet_id` (1-based) not `sheet_name`.
- Unix timestamp → Polars `Datetime` requires multiplying by `1_000_000` (microseconds), not nanoseconds.
