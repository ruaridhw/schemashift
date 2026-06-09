"""Tests for smart_transform() — the full detect-or-generate flow."""

import polars as pl
import pytest
from conftest import make_tool_calling_llm

from schemashift.errors import FormatDetectionError, ReviewRejectedError
from schemashift.models import ColumnMapping, TransformSpec
from schemashift.orchestration import smart_transform
from schemashift.registry import DictRegistry
from schemashift.validation import ColumnConstraints, SchemaConfig


@pytest.fixture
def sample_csv(tmp_path):
    p = tmp_path / "data.csv"
    pl.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Score": [90, 85],
            "Grade": ["A", "B"],
        }
    ).write_csv(str(p))
    return str(p)


@pytest.fixture
def schema():
    return SchemaConfig(
        name="students",
        columns={
            "student_name": ColumnConstraints(type="str", nullable=False, description="Name"),
            "score": ColumnConstraints(type="float64", nullable=False, description="Score"),
            "grade": ColumnConstraints(type="str", nullable=False, description="Grade"),
        },
    )


@pytest.fixture
def matching_config():
    return TransformSpec(
        name="student_format",
        columns=[
            ColumnMapping(target="student_name", source="Name"),
            ColumnMapping(target="score", source="Score", dtype="float64"),
            ColumnMapping(target="grade", source="Grade"),
        ],
    )


def _valid_config() -> dict:
    return {
        "name": "gen",
        "columns": [
            {"target": "student_name", "source": "Name"},
            {"target": "score", "source": "Score", "dtype": "float64"},
            {"target": "grade", "source": "Grade"},
        ],
    }


class TestRegistryHit:
    def test_uses_registry_when_match(self, sample_csv, schema, matching_config):
        reg = DictRegistry()
        reg.register(matching_config)
        result = smart_transform(sample_csv, registry=reg, target_schema=schema)
        assert set(result.valid.columns) == {"student_name", "score", "grade"}
        assert len(result.valid) == 2

    def test_works_without_target_schema(self, sample_csv, matching_config):
        reg = DictRegistry()
        reg.register(matching_config)
        result = smart_transform(sample_csv, registry=reg)
        assert "student_name" in result.valid.columns


class TestLLMGeneration:
    def test_generates_when_no_match(self, sample_csv, schema):
        reg = DictRegistry()
        result = smart_transform(
            sample_csv, registry=reg, target_schema=schema, llm=make_tool_calling_llm(_valid_config())
        )
        assert set(result.valid.columns) == {"student_name", "score", "grade"}

    def test_auto_registers(self, sample_csv, schema):
        reg = DictRegistry()
        smart_transform(
            sample_csv,
            registry=reg,
            target_schema=schema,
            llm=make_tool_calling_llm(_valid_config()),
            auto_register=True,
        )
        assert reg.get("gen") is not None

    def test_raises_without_llm(self, sample_csv, schema):
        with pytest.raises(FormatDetectionError, match="no LLM"):
            smart_transform(sample_csv, registry=DictRegistry(), target_schema=schema)

    def test_raises_without_schema(self, sample_csv):
        with pytest.raises(ValueError, match="target_schema"):
            smart_transform(sample_csv, registry=DictRegistry(), llm=make_tool_calling_llm(_valid_config()))


class TestReviewFn:
    def test_review_fn_modifies_config(self, sample_csv, schema):
        reg = DictRegistry()

        def review(cfg, df_sample):
            return TransformSpec(name="reviewed", columns=cfg.columns)

        result = smart_transform(
            sample_csv,
            registry=reg,
            target_schema=schema,
            llm=make_tool_calling_llm(_valid_config()),
            review_fn=review,
            auto_register=True,
        )
        assert reg.get("reviewed") is not None
        assert len(result.valid) == 2

    def test_review_fn_rejection(self, sample_csv, schema):
        with pytest.raises(ReviewRejectedError, match="rejected"):
            smart_transform(
                sample_csv,
                registry=DictRegistry(),
                target_schema=schema,
                llm=make_tool_calling_llm(_valid_config()),
                review_fn=lambda cfg, df: None,
            )
