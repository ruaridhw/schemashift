"""CLI entry-point for schemashift using Click."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from schemashift.target_schema import TargetSchema

import click
import polars as pl

from schemashift.errors import (
    AmbiguousFormatError,
    ConfigValidationError,
    FormatDetectionError,
)
from schemashift.models import FormatConfig
from schemashift.registry import FileSystemRegistry
from schemashift.transform import auto_transform, validate_config
from schemashift.transform import dry_run as _dry_run
from schemashift.transform import transform as _transform


@click.group()
def cli() -> None:
    """schemashift — Declarative file format transformer."""


@cli.command()
@click.argument("file")
@click.option("--config", "-c", help="Path to config JSON file.")
@click.option("--registry", "-r", help="Path to registry directory.")
@click.option("--output", "-o", help="Output file path (.csv, .parquet, or .json).")
def transform(file: str, config: str | None, registry: str | None, output: str | None) -> None:
    """Transform a file using a config or auto-detect from registry."""
    try:
        if config is not None:
            fmt_config = _load_format_config(config)
            lf = _transform(file, fmt_config)
        elif registry is not None:
            reg = FileSystemRegistry(registry)
            lf = auto_transform(file, reg)
        else:
            raise click.UsageError("Provide either --config or --registry.")

        df = lf.collect()

        if output is not None:
            _write_output(df, output)
            click.echo(f"Written {len(df)} rows to '{output}'.")
        else:
            click.echo(str(df.head(20)))

    except (FormatDetectionError, AmbiguousFormatError, ConfigValidationError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Unexpected error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("config_path")
def validate(config_path: str) -> None:
    """Validate a format config file."""
    try:
        fmt_config = _load_format_config(config_path)
    except (ConfigValidationError, Exception) as exc:
        click.echo(f"Config load error: {exc}", err=True)
        sys.exit(1)

    errors = validate_config(fmt_config)
    if errors:
        click.echo("Validation failed:")
        for err in errors:
            click.echo(f"  - {err}")
        sys.exit(1)
    else:
        click.echo(f"Config '{fmt_config.name}' is valid.")


@cli.command(name="dry-run")
@click.argument("config_path")
@click.option("--sample", "-s", required=True, help="Sample data file.")
@click.option("--rows", "-n", default=10, show_default=True, help="Number of rows to process.")
def dry_run(config_path: str, sample: str, rows: int) -> None:
    """Dry-run a config against sample data."""
    try:
        fmt_config = _load_format_config(config_path)
        df = _dry_run(fmt_config, sample, n_rows=rows)
        click.echo(str(df))
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("file")
@click.option(
    "--target-schema",
    "-t",
    help="Path to target schema YAML. If omitted, looks in --registry/schemas/",
)
@click.option("--output", "-o", help="Output path for generated config JSON.")
@click.option("--registry", "-r", help="Registry directory (auto-register if provided).")
@click.option("--name", "-n", help="Name for the generated config.")
@click.option("--rows", default=15, show_default=True, help="Sample rows for LLM.")
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    default=False,
    help="Interactively review generated config before saving.",
)
def generate(  # noqa: PLR0913
    file: str,
    target_schema: str | None,
    output: str | None,
    registry: str | None,
    name: str | None,
    rows: int,
    interactive: bool,
) -> None:
    """Generate a FormatConfig for an unknown file using an LLM.

    Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in environment,
    or configure your LLM via environment variables.
    """
    try:
        from schemashift.llm import generate_config

        schema = _resolve_schema(target_schema, registry)

        # Try to load LangChain LLM from environment
        llm = _load_default_llm()

        reg = FileSystemRegistry(registry) if registry else None

        config = generate_config(
            path=file,
            target_schema=schema,
            llm=llm,
            format_name=name,
            sample_rows=rows,
        )

        if interactive:
            click.echo("\nGenerated config:")
            click.echo(config.model_dump_json(indent=2))

            from schemashift.transform import dry_run

            sample = dry_run(config, file, n_rows=5)
            click.echo("\nSample output (first 5 rows):")
            click.echo(str(sample))

            if not click.confirm("\nAccept this config?", default=False):
                click.echo("Config rejected. Aborting.", err=True)
                sys.exit(1)

        if reg is not None:
            reg.register(config)
            click.echo(f"Registered config '{config.name}' in '{registry}'.")

        config_json = config.model_dump_json(indent=2)
        if output:
            Path(output).write_text(config_json, encoding="utf-8")
            click.echo(f"Config written to '{output}'.")
        else:
            click.echo(config_json)

    except (click.UsageError, click.ClickException):
        raise
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command(name="list")
@click.option("--registry", "-r", required=True, help="Registry directory.")
def list_configs(registry: str) -> None:
    """List all registered configs."""
    try:
        reg = FileSystemRegistry(registry)
        configs = reg.list_configs()
        if not configs:
            click.echo("No configs registered.")
            return
        for cfg in configs:
            click.echo(f"{cfg.name}  (v{cfg.version})  — {cfg.description}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_schema(
    target_schema_path: str | None, registry_path: str | None
) -> TargetSchema:
    """Resolve a TargetSchema from an explicit path or a registry schemas/ dir.

    Resolution order:
    1. If *target_schema_path* is given, load it directly.
    2. If *registry_path* is given, glob for ``*.yaml``/``*.yml`` in
       ``{registry_path}/schemas/``.  Exactly one match is required.
    3. Otherwise raise :class:`click.UsageError`.
    """
    from schemashift.target_schema import TargetSchema

    if target_schema_path is not None:
        return TargetSchema.from_yaml(target_schema_path)

    if registry_path is not None:
        schemas_dir = Path(registry_path) / "schemas"
        if schemas_dir.exists():
            yamls = list(schemas_dir.glob("*.yaml")) + list(schemas_dir.glob("*.yml"))
            if len(yamls) == 1:
                return TargetSchema.from_yaml(yamls[0])
            elif len(yamls) > 1:
                names = [y.name for y in yamls]
                raise click.UsageError(
                    f"Multiple schemas found in '{schemas_dir}': {names}. "
                    f"Use --target-schema to specify one."
                )

    raise click.UsageError(
        "Provide --target-schema or --registry with a schemas/ subdirectory."
    )


def _load_default_llm() -> Any:
    """Load a LangChain LLM from environment variables.

    Prefers Anthropic (claude-haiku-4-5) when ANTHROPIC_API_KEY is set,
    falls back to OpenAI (gpt-4o-mini) when OPENAI_API_KEY is set.
    """
    import os

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise click.ClickException(
                "langchain-anthropic is not installed. Run: uv add 'schemashift[llm]'"
            )
        return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise click.ClickException(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            )
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    raise click.ClickException(
        "No LLM API key found. Set ANTHROPIC_API_KEY (recommended) or OPENAI_API_KEY."
    )


def _load_format_config(path: str) -> FormatConfig:
    """Load a FormatConfig from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FormatConfig.model_validate(data)


def _write_output(df: pl.DataFrame, output: str) -> None:
    """Write a DataFrame to the given path based on its file extension."""
    out_path = Path(output)
    ext = out_path.suffix.lower()
    if ext == ".csv":
        df.write_csv(output)
    elif ext == ".parquet":
        df.write_parquet(output)
    elif ext == ".json":
        df.write_json(output)
    else:
        raise click.UsageError(
            f"Unsupported output format '{ext}'. Use .csv, .parquet, or .json."
        )
