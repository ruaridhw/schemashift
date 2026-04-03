"""Integration tests that call a real LLM.

These tests are excluded from the default pytest run via the ``llm`` marker.
Run explicitly with::

    pytest -m llm

Requires ``ANTHROPIC_API_KEY`` (or ``FOUNDRY_API_KEY`` + ``FOUNDRY_RESOURCE``)
in the environment.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from schemashift.cli import cli
from schemashift.llm import generate_config, load_default_llm
from schemashift.target_schema import TargetSchema

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def llm():
    return load_default_llm()


@pytest.fixture
def lot_movement_schema():
    return TargetSchema.from_yaml(FIXTURES / "configs" / "lot_movement_schema.yaml")


@pytest.fixture
def camstar_csv():
    return str(FIXTURES / "csv" / "camstar_sample.csv")


class TestGenerateConfigIntegration:
    """End-to-end generation with a real LLM."""

    def test_generates_valid_config(self, camstar_csv, lot_movement_schema, llm):
        config = generate_config(
            path=camstar_csv,
            target_schema=lot_movement_schema,
            llm=llm,
            n_sample_rows=5,
        )
        assert config.name
        assert len(config.columns) > 0

        # Config should actually transform the file without error
        from schemashift.transform import dry_run

        df = dry_run(config, camstar_csv, n_rows=3)
        assert len(df) == 3

    def test_prompt_influences_output(self, camstar_csv, lot_movement_schema, llm):
        """The prompt arg should be included as context for the LLM."""
        config = generate_config(
            path=camstar_csv,
            target_schema=lot_movement_schema,
            llm=llm,
            n_sample_rows=5,
            user_prompt=(
                "The QTY column represents wafer count per lot. "
                "HOLD_STATUS of 'NONE' means the lot is NOT on hold (false)."
            ),
        )
        assert config.name
        assert len(config.columns) > 0

        from schemashift.transform import dry_run

        df = dry_run(config, camstar_csv, n_rows=3)
        assert len(df) == 3


class TestCLIGenerateIntegration:
    """End-to-end CLI generate with a real LLM."""

    def test_cli_generate(self, camstar_csv, lot_movement_schema, tmp_path):
        out = tmp_path / "config.json"
        schema_path = FIXTURES / "configs" / "lot_movement_schema.yaml"
        result = CliRunner().invoke(
            cli,
            ["generate", camstar_csv, "-t", str(schema_path), "-o", str(out), "--rows", "5"],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_cli_generate_with_prompt(self, camstar_csv, lot_movement_schema, tmp_path):
        out = tmp_path / "config.json"
        schema_path = FIXTURES / "configs" / "lot_movement_schema.yaml"
        result = CliRunner().invoke(
            cli,
            [
                "generate",
                camstar_csv,
                "-t",
                str(schema_path),
                "-o",
                str(out),
                "--rows",
                "5",
                "--prompt",
                "QTY is wafer count. HOLD_STATUS 'NONE' means not on hold.",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
