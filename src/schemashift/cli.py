"""CLI entry-point for schemashift using Click."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
