"""Tests for schemashift.validation — SchemaConfig and dy.Schema factory."""

from pathlib import Path

import dataframely as dy
import polars as pl
import pytest
import yaml

from schemashift.validation import (
    ColumnConstraints,
    SchemaConfig,
    build_dy_schema,
    resolve_schema,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_schema_config() -> SchemaConfig:
    return SchemaConfig(
        name="test_schema",
        description="A simple test schema",
        columns={
            "id": ColumnConstraints(type="int64", nullable=False, primary_key=True),
            "name": ColumnConstraints(type="string", nullable=False, min_length=1),
            "amount": ColumnConstraints(type="float64", nullable=False, min=0),
            "category": ColumnConstraints(type="string", nullable=True),
        },
    )


@pytest.fixture
def simple_schema_yaml(tmp_path: Path, simple_schema_config: SchemaConfig) -> Path:
    data = simple_schema_config.model_dump()
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


# ---------------------------------------------------------------------------
# SchemaConfig construction and YAML round-trip
# ---------------------------------------------------------------------------


class TestSchemaConfigConstruction:
    def test_basic_fields(self, simple_schema_config: SchemaConfig) -> None:
        assert simple_schema_config.name == "test_schema"
        assert simple_schema_config.description == "A simple test schema"
        assert len(simple_schema_config.columns) == 4

    def test_column_constraints(self, simple_schema_config: SchemaConfig) -> None:
        id_col = simple_schema_config.columns["id"]
        assert id_col.type == "int64"
        assert id_col.nullable is False
        assert id_col.primary_key is True

    def test_string_constraints(self, simple_schema_config: SchemaConfig) -> None:
        name_col = simple_schema_config.columns["name"]
        assert name_col.type == "string"
        assert name_col.min_length == 1

    def test_numeric_constraints(self, simple_schema_config: SchemaConfig) -> None:
        amount_col = simple_schema_config.columns["amount"]
        assert amount_col.type == "float64"
        assert amount_col.min == 0


class TestSchemaConfigYamlRoundTrip:
    def test_from_yaml(self, simple_schema_yaml: Path) -> None:
        loaded = SchemaConfig.from_yaml(simple_schema_yaml)
        assert loaded.name == "test_schema"
        assert len(loaded.columns) == 4

    def test_yaml_round_trip_preserves_constraints(self, simple_schema_yaml: Path) -> None:
        loaded = SchemaConfig.from_yaml(simple_schema_yaml)
        assert loaded.columns["amount"].min == 0
        assert loaded.columns["name"].min_length == 1
        assert loaded.columns["id"].primary_key is True

    def test_yaml_round_trip_preserves_nullable(self, simple_schema_yaml: Path) -> None:
        loaded = SchemaConfig.from_yaml(simple_schema_yaml)
        assert loaded.columns["id"].nullable is False
        assert loaded.columns["category"].nullable is True


class TestColumnConstraintsValidation:
    def test_string_with_numeric_constraint_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid for string type"):
            ColumnConstraints(type="string", min=0)

    def test_numeric_with_string_constraint_raises(self) -> None:
        with pytest.raises(ValueError, match="only valid for string types"):
            ColumnConstraints(type="int64", min_length=1)

    def test_float_with_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="only valid for string types"):
            ColumnConstraints(type="float64", regex=r"\d+")


# ---------------------------------------------------------------------------
# build_dy_schema
# ---------------------------------------------------------------------------


class TestBuildDySchema:
    def test_produces_dy_schema_subclass(self, simple_schema_config: SchemaConfig) -> None:
        schema = build_dy_schema(simple_schema_config)
        assert issubclass(schema, dy.Schema)

    def test_schema_name_matches(self, simple_schema_config: SchemaConfig) -> None:
        schema = build_dy_schema(simple_schema_config)
        assert schema.__name__ == "test_schema"

    def test_column_names(self, simple_schema_config: SchemaConfig) -> None:
        schema = build_dy_schema(simple_schema_config)
        assert set(schema.column_names()) == {"id", "name", "amount", "category"}

    def test_validates_good_data(self, simple_schema_config: SchemaConfig) -> None:
        schema = build_dy_schema(simple_schema_config)
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Carol"],
                "amount": [10.0, 20.0, 30.0],
                "category": ["food", "transport", None],
            }
        )
        result = schema.filter(df)
        assert result.result.height == 3

    def test_catches_null_in_required_column(self, simple_schema_config: SchemaConfig) -> None:
        schema = build_dy_schema(simple_schema_config)
        df = pl.DataFrame(
            {
                "id": pl.Series([1, None, 3], dtype=pl.Int64),
                "name": pl.Series(["Alice", "Bob", "Carol"], dtype=pl.Utf8),
                "amount": pl.Series([10.0, 20.0, 30.0], dtype=pl.Float64),
                "category": pl.Series([None, None, None], dtype=pl.Utf8),
            }
        )
        result = schema.filter(df)
        assert result.result.height == 2
        assert result.failure is not None
        assert "id|nullability" in result.failure.counts()

    def test_catches_min_violation(self, simple_schema_config: SchemaConfig) -> None:
        schema = build_dy_schema(simple_schema_config)
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Carol"],
                "amount": [10.0, -5.0, 30.0],
                "category": pl.Series([None, None, None], dtype=pl.Utf8),
            }
        )
        result = schema.filter(df)
        assert result.result.height == 2
        assert "amount|min" in result.failure.counts()

    def test_catches_min_length_violation(self, simple_schema_config: SchemaConfig) -> None:
        schema = build_dy_schema(simple_schema_config)
        df = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["Alice", ""],
                "amount": [10.0, 20.0],
                "category": pl.Series([None, None], dtype=pl.Utf8),
            }
        )
        result = schema.filter(df)
        assert result.result.height == 1
        assert "name|min_length" in result.failure.counts()

    def test_is_in_integer(self) -> None:
        config = SchemaConfig(
            name="s",
            columns={"status": ColumnConstraints(type="int64", is_in=[1, 2, 3])},
        )
        schema = build_dy_schema(config)
        df = pl.DataFrame({"status": [1, 4, 2, 5]})
        result = schema.filter(df)
        assert result.result.height == 2

    def test_is_in_string_via_check(self) -> None:
        config = SchemaConfig(
            name="s",
            columns={"currency": ColumnConstraints(type="string", is_in=["USD", "EUR", "GBP"])},
        )
        schema = build_dy_schema(config)
        df = pl.DataFrame({"currency": ["USD", "XXX", "EUR", "JPY"]})
        result = schema.filter(df)
        assert result.result.height == 2
        assert result.failure.counts()["currency|check__is_in"] == 2

    def test_nullable_column_allows_null(self) -> None:
        config = SchemaConfig(
            name="s",
            columns={"notes": ColumnConstraints(type="string", nullable=True)},
        )
        schema = build_dy_schema(config)
        df = pl.DataFrame({"notes": ["hello", None, "world"]})
        result = schema.filter(df)
        assert result.result.height == 3


# ---------------------------------------------------------------------------
# resolve_schema
# ---------------------------------------------------------------------------


class TestResolveSchema:
    def test_resolves_schema_config(self, simple_schema_config: SchemaConfig) -> None:
        schema = resolve_schema(simple_schema_config)
        assert issubclass(schema, dy.Schema)

    def test_resolves_dy_schema_class(self) -> None:
        class MySchema(dy.Schema):
            x = dy.Int64()

        result = resolve_schema(MySchema)
        assert result is MySchema

    def test_rejects_invalid_type(self) -> None:
        with pytest.raises(TypeError, match=r"Expected SchemaConfig or dy\.Schema"):
            resolve_schema("not a schema")  # type: ignore[arg-type]
