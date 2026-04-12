"""Accessors for the bundled JSON Schema for TransformSpec."""

import importlib.resources as ilr
import json
from pathlib import Path


def get_schema_path() -> Path:
    """Return the path to the bundled format_config.json schema file."""
    return Path(str(ilr.files("schemashift").joinpath("schema").joinpath("format_config.json")))


def get_schema() -> dict:
    """Return the bundled JSON Schema for TransformSpec (includes $schema key)."""
    return json.loads(get_schema_path().read_text(encoding="utf-8"))
