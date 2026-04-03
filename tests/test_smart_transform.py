"""Tests for smart_transform() — the full detect-or-generate flow."""

import polars as pl
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from schemashift.errors import FormatDetectionError, ReviewRejectedError
from schemashift.models import ColumnMapping, FormatConfig
from schemashift.registry import DictRegistry
from schemashift.target_schema import TargetColumn, TargetSchema
from schemashift.transform import smart_transform


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Minimal fake LLM that supports tool-calling for use with create_agent."""

    def bind_tools(self, tools, **kwargs):
        return self


def _make_llm(config_args: dict) -> FakeToolCallingModel:
    """Return a fake LLM that submits *config_args* via submit_format_config then finishes."""
    return FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "submit_format_config", "args": config_args}],
            ),
            AIMessage(content="Done."),
        ]
    )


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
    return TargetSchema(
        name="students",
        columns=[
            TargetColumn(name="student_name", type="str", required=True, description="Name"),
            TargetColumn(name="score", type="float64", required=True, description="Score"),
            TargetColumn(name="grade", type="str", required=True, description="Grade"),
        ],
    )


@pytest.fixture
def matching_config():
    return FormatConfig(
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
        lf = smart_transform(sample_csv, registry=reg, target_schema=schema, llm=_make_llm(_valid_config()))
        assert set(lf.collect().columns) == {"student_name", "score", "grade"}

    def test_auto_registers(self, sample_csv, schema):
        reg = DictRegistry()
        smart_transform(
            sample_csv,
            registry=reg,
            target_schema=schema,
            llm=_make_llm(_valid_config()),
            auto_register=True,
        )
        assert reg.get("gen") is not None

    def test_raises_without_llm(self, sample_csv, schema):
        with pytest.raises(FormatDetectionError, match="no LLM"):
            smart_transform(sample_csv, registry=DictRegistry(), target_schema=schema)

    def test_raises_without_schema(self, sample_csv):
        with pytest.raises(ValueError, match="target_schema"):
            smart_transform(sample_csv, registry=DictRegistry(), llm=_make_llm(_valid_config()))


class TestReviewFn:
    def test_review_fn_modifies_config(self, sample_csv, schema):
        reg = DictRegistry()

        def review(cfg, df_sample):
            return FormatConfig(name="reviewed", columns=cfg.columns)

        lf = smart_transform(
            sample_csv,
            registry=reg,
            target_schema=schema,
            llm=_make_llm(_valid_config()),
            review_fn=review,
            auto_register=True,
        )
        assert reg.get("reviewed") is not None
        assert len(lf.collect()) == 2

    def test_review_fn_rejection(self, sample_csv, schema):
        with pytest.raises(ReviewRejectedError, match="rejected"):
            smart_transform(
                sample_csv,
                registry=DictRegistry(),
                target_schema=schema,
                llm=_make_llm(_valid_config()),
                review_fn=lambda cfg, df: None,
            )
