"""Tests for JSON Schema generation and the bundled schema file."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from schemashift.cli import cli
from schemashift.models import TransformSpec
from schemashift.schema import get_schema, get_schema_path


class TestSchemaCommand:
    """Tests for the ``schemashift schema`` CLI command."""

    def test_exits_zero_and_outputs_valid_json(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["schema"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["title"] == "TransformSpec"

    def test_schema_contains_dollar_schema_key(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["schema"])
        data = json.loads(result.output)
        assert data["$schema"] == "http://json-schema.org/draft-07/schema#"

    def test_output_option_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "schema.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["schema", "--output", str(out)])
        assert result.exit_code == 0
        assert "Schema written to" in result.output
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["title"] == "TransformSpec"

    def test_dtype_enum_contains_int32(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["schema"])
        data = json.loads(result.output)
        dtype_any_of = data["$defs"]["ColumnMapping"]["properties"]["dtype"]["anyOf"]
        enum_values = next(entry["enum"] for entry in dtype_any_of if "enum" in entry)
        assert "int32" in enum_values


class TestBundledSchema:
    """Tests for the bundled schema file and accessors."""

    def test_get_schema_path_exists(self) -> None:
        path = get_schema_path()
        assert path.exists()
        assert path.name == "format_config.json"

    def test_get_schema_returns_dict_with_title(self) -> None:
        schema = get_schema()
        assert schema["title"] == "TransformSpec"
        assert "$schema" in schema

    def test_bundled_schema_is_fresh(self) -> None:
        """The bundled schema must match the live model. Regenerate with:

        uv run schemashift schema -o src/schemashift/schema/format_config.json
        """
        bundled = get_schema()
        bundled_without_meta = {k: v for k, v in bundled.items() if k != "$schema"}
        live = TransformSpec.model_json_schema()
        assert bundled_without_meta == live, (
            "Bundled schema is out of date. Regenerate with:\n"
            "  uv run schemashift schema -o src/schemashift/schema/format_config.json"
        )
