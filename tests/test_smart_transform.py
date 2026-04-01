"""Tests for smart_transform() — the full detect-or-generate flow."""

from __future__ import annotations

import json

import polars as pl
import pytest

from schemashift.errors import FormatDetectionError
from schemashift.models import ColumnMapping, FormatConfig
from schemashift.registry import DictRegistry
from schemashift.target_schema import TargetColumn, TargetSchema
from schemashift.transform import smart_transform


class MockLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, prompt: str) -> object:
        class Msg:
            pass

        msg = Msg()
        msg.content = self._response
        return msg


@pytest.fixture
def sample_csv(tmp_path):
    p = tmp_path / "data.csv"
    pl.DataFrame({
        "Name": ["Alice", "Bob"],
        "Score": [90, 85],
        "Grade": ["A", "B"],
    }).write_csv(str(p))
    return str(p)


@pytest.fixture
def schema():
    return TargetSchema(name="students", columns=[
        TargetColumn(name="student_name", type="str", required=True, description="Name"),
        TargetColumn(name="score", type="float64", required=True, description="Score"),
        TargetColumn(name="grade", type="str", required=True, description="Grade"),
    ])


@pytest.fixture
def matching_config():
    return FormatConfig(name="student_format", columns=[
        ColumnMapping(target="student_name", source="Name"),
        ColumnMapping(target="score", source="Score", dtype="float64"),
        ColumnMapping(target="grade", source="Grade"),
    ])


def _valid_llm_response() -> str:
    return json.dumps({"name": "gen", "columns": [
        {"target": "student_name", "source": "Name"},
        {"target": "score", "source": "Score", "dtype": "float64"},
        {"target": "grade", "source": "Grade"},
    ]})


class TestRegistryHit:
    def test_uses_registry_when_match(self, sample_csv, schema, matching_config):
        reg = DictRegistry()
        reg.register(matching_config)
        lf = smart_transform(sample_csv, registry=reg, target_schema=schema)
        df = lf.collect()
        assert set(df.columns) == {"student_name", "score", "grade"}
        assert len(df) == 2

    def test_works_without_target_schema(self, sample_csv, matching_config):
        reg = DictRegistry()
        reg.register(matching_config)
        lf = smart_transform(sample_csv, registry=reg)
        assert "student_name" in lf.collect().columns


class TestLLMGeneration:
    def test_generates_when_no_match(self, sample_csv, schema):
        reg = DictRegistry()
        lf = smart_transform(
            sample_csv, registry=reg, target_schema=schema,
            llm=MockLLM(_valid_llm_response()),
        )
        assert set(lf.collect().columns) == {"student_name", "score", "grade"}

    def test_auto_registers(self, sample_csv, schema):
        reg = DictRegistry()
        smart_transform(
            sample_csv, registry=reg, target_schema=schema,
            llm=MockLLM(_valid_llm_response()), auto_register=True,
        )
        assert reg.get("gen") is not None

    def test_raises_without_llm(self, sample_csv, schema):
        with pytest.raises(FormatDetectionError, match="no LLM"):
            smart_transform(sample_csv, registry=DictRegistry(), target_schema=schema)

    def test_raises_without_schema(self, sample_csv):
        with pytest.raises(ValueError, match="target_schema"):
            smart_transform(
                sample_csv, registry=DictRegistry(),
                llm=MockLLM(_valid_llm_response()),
            )


class TestReviewFn:
    def test_review_fn_modifies_config(self, sample_csv, schema):
        reg = DictRegistry()

        def review(cfg, df_sample):
            return FormatConfig(name="reviewed", columns=cfg.columns)

        lf = smart_transform(
            sample_csv, registry=reg, target_schema=schema,
            llm=MockLLM(_valid_llm_response()),
            review_fn=review, auto_register=True,
        )
        assert reg.get("reviewed") is not None
        assert len(lf.collect()) == 2

    def test_review_fn_rejection(self, sample_csv, schema):
        with pytest.raises(FormatDetectionError, match="rejected"):
            smart_transform(
                sample_csv, registry=DictRegistry(), target_schema=schema,
                llm=MockLLM(_valid_llm_response()),
                review_fn=lambda cfg, df: None,
            )
