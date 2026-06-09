"""Public ontology contract tests for schemashift."""

from __future__ import annotations

import json

import polars as pl
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

import schemashift as ss
import schemashift.validation as validation
from schemashift.cli import cli
from schemashift.models import ColumnMapping, TransformSpec
from schemashift.transform import transform
from schemashift.validation import ColumnConstraints


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _dataset_schema() -> validation.DatasetSchema:
    return validation.DatasetSchema(
        name="records",
        columns={
            "id": ColumnConstraints(type="int64", nullable=False),
        },
    )


class TestPublicOntology:
    def test_dataset_schema_is_public_name(self) -> None:
        assert validation.DatasetSchema.__name__ == "DatasetSchema"
        assert ss.DatasetSchema is validation.DatasetSchema

    @pytest.mark.parametrize(
        "legacy_name",
        ["SchemaConfig", "TargetSchema"],
    )
    def test_legacy_schema_class_names_are_not_exported(self, legacy_name: str) -> None:
        assert not hasattr(validation, legacy_name)
        assert not hasattr(ss, legacy_name)


class TestDatasetSchemaField:
    def test_transform_spec_uses_dataset_schema_field(self) -> None:
        spec = TransformSpec(
            name="records",
            dataset_schema=_dataset_schema(),
            columns=[ColumnMapping(target="id", source="raw_id")],
        )

        assert spec.dataset_schema is not None
        assert spec.model_dump()["dataset_schema"]["name"] == "records"

    @pytest.mark.parametrize("legacy_field", ["target_schema", "schema", "output_schema"])
    def test_legacy_schema_fields_are_rejected(self, legacy_field: str) -> None:
        with pytest.raises(ValidationError):
            TransformSpec.model_validate(
                {
                    "name": "records",
                    legacy_field: _dataset_schema().model_dump(),
                    "columns": [{"target": "id", "source": "raw_id"}],
                }
            )


class TestDatasetSchemaRuntimeParameter:
    def test_transform_accepts_dataset_schema_parameter(self, tmp_path) -> None:
        csv = tmp_path / "records.csv"
        pl.DataFrame({"raw_id": [1, 2]}).write_csv(csv)
        spec = TransformSpec(name="records", columns=[ColumnMapping(target="id", source="raw_id")])

        result = transform(csv, spec, dataset_schema=_dataset_schema())

        assert result.all_valid
        assert result.valid["id"].to_list() == [1, 2]


class TestDatasetSchemaCli:
    def test_generate_uses_dataset_schema_option(self, runner: CliRunner, tmp_path, monkeypatch) -> None:
        schema_path = tmp_path / "dataset.yaml"
        schema_path.write_text(
            """
name: records
columns:
  id:
    type: int64
    nullable: false
""".strip(),
            encoding="utf-8",
        )
        csv = tmp_path / "records.csv"
        pl.DataFrame({"raw_id": [1]}).write_csv(csv)
        generated = TransformSpec(name="records_csv", columns=[ColumnMapping(target="id", source="raw_id")])

        monkeypatch.setattr("schemashift.cli._load_default_llm", lambda: object())
        monkeypatch.setattr("schemashift.cli.generate_config", lambda *args, **kwargs: generated)

        result = runner.invoke(cli, ["generate", str(csv), "--dataset-schema", str(schema_path)])

        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["name"] == "records_csv"

    def test_generate_rejects_target_schema_option(self, runner: CliRunner, tmp_path) -> None:
        csv = tmp_path / "records.csv"
        pl.DataFrame({"raw_id": [1]}).write_csv(csv)

        result = runner.invoke(cli, ["generate", str(csv), "--target-schema", "legacy.yaml"])

        assert result.exit_code != 0
        assert "No such option: --target-schema" in result.output
