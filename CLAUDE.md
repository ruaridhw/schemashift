# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
rtk uv run black src/ tests/             # formatting check
pre-commit run                                 # lint + format (run after staging changes)
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
- `compiler.py` — `compile_dsl()` uses `match/case` to dispatch each node type to a `polars.Expr`. dtype string→Polars type mapping lives in `dtypes.py` as `DTYPE_MAP`.
- `analysis.py` — `collect_col_refs()` walks an AST to extract all referenced column names. Used by `FormatConfig.source_columns()`.
- `_lookups.py` — auto-discovers JSON files in `dsl/tables/` and exposes them as the `TABLES` dict (used by `lookup()` / `custom_lookup()` DSL ops).

To add a new DSL operation: add to the allowlist in `parser.py`, add a case in `compiler.py`, add to `_DSL_REFERENCE` in `llm.py`.

The DSL is a **closed language** — no eval, no arbitrary method calls. The parser raises `DSLSyntaxError` for anything not explicitly allowlisted.

### Transform engine (`transform.py`)

`transform(path, config)` → `pl.LazyFrame`. Never materialises data unless `dry_run()` or the caller calls `.collect()`. Also exposes `validate_config()` (DSL parse check) and `dry_run()`.

### Orchestration (`orchestration.py`)

Higher-level flows built on top of the core transform engine.

`auto_transform()` detects the format from the registry and applies it directly. `smart_transform()` is the full detect-or-generate flow: registry hit → apply directly; miss → call `llm.generate_config()` → optional `review_fn` callback → optional `auto_register`.

### Format detection (`detection.py`)

`detect_format(file_columns, registry)` scores registered configs against the file's column set and returns the best match (or `None`). Configs are ranked by specificity: `len(required_cols) / len(file_cols)`. Raises `AmbiguousFormatError` when multiple configs tie above the minimum score threshold.

### Registry (`registry.py`)

`DictRegistry` (in-memory) and `FileSystemRegistry` (JSON files per config in a directory). `FileSystemRegistry.load_schema()` looks for a `TargetSchema` in a `schemas/` subdirectory — convention used by the CLI.

### LLM generation (`llm.py`)

`generate_config()` accepts any `LLMBackend` implementation; plain LangChain `BaseChatModel` instances are auto-wrapped in `LangChainLLMBackend`. Retry loop (default 2 retries): extract JSON → `FormatConfig.model_validate` → `validate_config` (DSL parse check) → `dry_run`. Each failed attempt is logged at `WARNING` and stored in `LLMGenerationError.attempts`.

### Target schema (`target_schema.py`)

`TargetSchema` is the source of truth for output shape. `validate_lazy()` checks column names/dtypes against the `LazyFrame` schema without collecting. `validate_eager()` also checks for nulls in required columns.

### Error hierarchy

All errors inherit from `SchemaShiftError`. Full list:

- `DSLSyntaxError` — parse-time, has `.expression` and `.position`
- `DSLRuntimeError` — evaluation-time
- `FormatDetectionError` — base for detection failures; `AmbiguousFormatError` (has `.candidates`) is a subclass
- `ReviewRejectedError` — raised when a `review_fn` rejects a generated config
- `LLMGenerationError` — has `.attempts` (list of failed attempt dicts)
- `ConfigValidationError` — invalid config structure
- `SchemaValidationError` — output shape doesn't match `TargetSchema`
- `UnsupportedFileError` — unrecognised file extension
- `ReaderError` — file read failure

## Key design constraints

- The transform engine only ever sees a `pl.LazyFrame` — format-specific logic stays in `readers.py`.
- `TargetSchema` validates the *output* of a config, not the config itself. A syntactically valid config can still produce schema-failing output.
- Excel files use `fastexcel` (calamine engine). Integer `sheet_name` values are passed as `sheet_id` (1-based) not `sheet_name`.
- Unix timestamp → Polars `Datetime` requires multiplying by `1_000_000` (microseconds), not nanoseconds.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **schemashift** (1192 symbols, 4074 relationships, 99 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/schemashift/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/schemashift/context` | Codebase overview, check index freshness |
| `gitnexus://repo/schemashift/clusters` | All functional areas |
| `gitnexus://repo/schemashift/processes` | All execution flows |
| `gitnexus://repo/schemashift/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyse to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyse without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
