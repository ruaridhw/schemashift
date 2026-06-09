"""End-to-end tests for the dataframely integration.

Verifies:
- Partial transform failures (lenient casts) surface in FailureInfo
- Strict mode raises SchemaValidationError with FailureInfo attached
- Expression errors are captured, not raised
- dy.Schema escape hatch works end-to-end
- TransformResult contract is correct
"""

from pathlib import Path

import dataframely as dy
import polars as pl
import pytest

from schemashift.errors import SchemaValidationError
from schemashift.models import ColumnMapping, TransformSpec
from schemashift.result import TransformResult
from schemashift.transform import transform
from schemashift.validation import ColumnConstraints, SchemaConfig


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "data.csv"
    pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "", None],
            "amount": [100.0, -5.0, 50.0, None],
            "currency": ["USD", "XXX", "EUR", "GBP"],
        }
    ).write_csv(str(p))
    return p


@pytest.fixture
def schema_config() -> SchemaConfig:
    return SchemaConfig(
        name="test_output",
        columns={
            "id": ColumnConstraints(type="int64", nullable=False),
            "name": ColumnConstraints(type="string", nullable=False, min_length=1),
            "amount": ColumnConstraints(type="float64", nullable=False, min=0),
            "currency": ColumnConstraints(type="string", is_in=["USD", "EUR", "GBP"]),
        },
    )


@pytest.fixture
def transform_spec() -> TransformSpec:
    return TransformSpec(
        name="test_spec",
        columns=[
            ColumnMapping(target="id", source="id"),
            ColumnMapping(target="name", source="name"),
            ColumnMapping(target="amount", source="amount"),
            ColumnMapping(target="currency", source="currency"),
        ],
    )


class TestPartialFailures:
    """Rows with bad data should appear in failures, not raise exceptions."""

    def test_valid_rows_returned_in_valid(
        self, sample_csv: Path, transform_spec: TransformSpec, schema_config: SchemaConfig
    ) -> None:
        result = transform(sample_csv, transform_spec, schema=schema_config)
        # Only row 0 (id=1, name=Alice, amount=100, currency=USD) should be fully valid
        assert result.valid.height == 1
        assert result.valid["id"].to_list() == [1]

    def test_invalid_rows_in_failure_info(
        self, sample_csv: Path, transform_spec: TransformSpec, schema_config: SchemaConfig
    ) -> None:
        result = transform(sample_csv, transform_spec, schema=schema_config)
        assert result.failures.has_failures
        assert result.failures.invalid is not None
        assert result.failures.invalid.height == 3

    def test_failure_counts_per_rule(
        self, sample_csv: Path, transform_spec: TransformSpec, schema_config: SchemaConfig
    ) -> None:
        result = transform(sample_csv, transform_spec, schema=schema_config)
        counts = result.failures.counts
        # Row 1: amount=-5 violates min=0, currency=XXX violates is_in
        # Row 2: name="" violates min_length=1
        # Row 3: name=null, amount=null violate nullable=false
        assert counts["amount|min"] >= 1
        assert counts["currency|check__is_in"] >= 1
        assert counts["name|min_length"] >= 1

    def test_all_valid_is_false(
        self, sample_csv: Path, transform_spec: TransformSpec, schema_config: SchemaConfig
    ) -> None:
        result = transform(sample_csv, transform_spec, schema=schema_config)
        assert result.all_valid is False

    def test_returns_transform_result(
        self, sample_csv: Path, transform_spec: TransformSpec, schema_config: SchemaConfig
    ) -> None:
        result = transform(sample_csv, transform_spec, schema=schema_config)
        assert isinstance(result, TransformResult)


class TestCleanData:
    """When all rows are valid, failures should have no issues."""

    def test_all_valid_data(self, tmp_path: Path) -> None:
        csv = tmp_path / "clean.csv"
        pl.DataFrame(
            {
                "value": [10, 20, 30],
                "label": ["a", "b", "c"],
            }
        ).write_csv(str(csv))

        config = TransformSpec(
            name="clean",
            columns=[
                ColumnMapping(target="value", source="value"),
                ColumnMapping(target="label", source="label"),
            ],
        )
        schema = SchemaConfig(
            name="s",
            columns={
                "value": ColumnConstraints(type="int64", nullable=False),
                "label": ColumnConstraints(type="string", nullable=False),
            },
        )
        result = transform(csv, config, schema=schema)
        assert result.all_valid
        assert result.valid.height == 3
        assert not result.failures.has_failures


class TestStrictMode:
    """strict=True should raise SchemaValidationError with FailureInfo."""

    def test_raises_on_failures(
        self, sample_csv: Path, transform_spec: TransformSpec, schema_config: SchemaConfig
    ) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            transform(sample_csv, transform_spec, schema=schema_config, strict=True)
        assert exc_info.value.failures is not None
        assert exc_info.value.failures.has_failures

    def test_strict_no_failures_does_not_raise(self, tmp_path: Path) -> None:
        csv = tmp_path / "ok.csv"
        pl.DataFrame({"x": [1, 2, 3]}).write_csv(str(csv))
        config = TransformSpec(
            name="ok",
            columns=[ColumnMapping(target="x", source="x")],
        )
        schema = SchemaConfig(
            name="s",
            columns={"x": ColumnConstraints(type="int64", nullable=False)},
        )
        result = transform(csv, config, schema=schema, strict=True)
        assert result.all_valid


class TestExpressionErrors:
    """Expression failures should be captured in expression_errors, not raised."""

    def test_bad_column_ref_raises_at_collect(self, tmp_path: Path) -> None:
        """Referencing a nonexistent column is a config bug — it raises at collect time."""
        csv = tmp_path / "data.csv"
        pl.DataFrame({"a": [1, 2]}).write_csv(str(csv))
        config = TransformSpec(
            name="bad",
            columns=[
                ColumnMapping(target="a", source="a"),
                ColumnMapping(target="b", expr='col("nonexistent") + 1'),
            ],
        )
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            transform(csv, config)

    def test_bad_syntax_captured(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        pl.DataFrame({"a": [1, 2]}).write_csv(str(csv))
        config = TransformSpec(
            name="syntax",
            columns=[ColumnMapping(target="x", expr='col("a"')],
        )
        result = transform(csv, config)
        assert "x" in result.failures.expression_errors


class TestDySchemaEscapeHatch:
    """Users can pass a dy.Schema class directly for custom rules."""

    def test_custom_dy_schema_with_rule(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        pl.DataFrame({"score": [10, -5, 20, 0]}).write_csv(str(csv))

        class ScoreSchema(dy.Schema):
            score = dy.Int64(nullable=False, min=0)

            @dy.rule()
            def score_nonzero(cls):
                return pl.col("score") > 0

        config = TransformSpec(
            name="score",
            columns=[ColumnMapping(target="score", source="score")],
        )
        result = transform(csv, config, schema=ScoreSchema)
        # score=10 and score=20 pass (>0 and >=0)
        # score=-5 fails min=0
        # score=0 fails score_nonzero rule
        assert result.valid.height == 2
        assert result.valid["score"].to_list() == [10, 20]
        assert result.failures.has_failures


class TestLenientCasts:
    """Casts that fail for some rows should produce nulls, not exceptions."""

    def test_bad_cast_produces_nulls(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        pl.DataFrame({"val": ["10", "abc", "30"]}).write_csv(str(csv))
        config = TransformSpec(
            name="cast",
            columns=[ColumnMapping(target="val", source="val", dtype="int64")],
        )
        schema = SchemaConfig(
            name="s",
            columns={"val": ColumnConstraints(type="int64", nullable=False)},
        )
        result = transform(csv, config, schema=schema)
        # Row "abc" should fail to cast and become null, then fail validation
        assert result.valid.height == 2
        assert result.failures.has_failures

    def test_no_schema_still_returns_transform_result(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        pl.DataFrame({"a": [1, 2]}).write_csv(str(csv))
        config = TransformSpec(
            name="no_schema",
            columns=[ColumnMapping(target="a", source="a")],
        )
        result = transform(csv, config)
        assert isinstance(result, TransformResult)
        assert result.all_valid
        assert result.valid.height == 2
