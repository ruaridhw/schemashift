"""Tests for schemashift.llm — LLM config generation engine."""

import logging

import polars as pl
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from schemashift.errors import LLMGenerationError
from schemashift.llm import _extract_json, build_prompt, generate_config
from schemashift.models import ColumnMapping, FormatConfig
from schemashift.target_schema import TargetColumn, TargetSchema


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Minimal fake LLM that supports tool-calling for use with create_agent."""

    def bind_tools(self, tools, **kwargs):
        return self


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_raw_json(self):
        result = _extract_json('{"name": "test", "columns": []}')
        assert result == '{"name": "test", "columns": []}'

    def test_code_block(self):
        text = 'Here:\n```json\n{"name": "t"}\n```'
        assert '"name"' in _extract_json(text)

    def test_json_with_surrounding_text(self):
        text = 'Sure! {"name": "t", "columns": []} done.'
        assert '"name"' in _extract_json(text)

    def test_no_json_raises(self):
        with pytest.raises(LLMGenerationError):
            _extract_json("no json here at all")


class TestBuildPrompt:
    @pytest.fixture
    def schema(self):
        return TargetSchema(
            name="test",
            columns=[
                TargetColumn(name="id", type="str", required=True, description="ID"),
                TargetColumn(name="value", type="float64", required=True, description="Value"),
            ],
        )

    def test_contains_target_columns(self, schema):
        df = pl.DataFrame({"col_a": ["x"], "col_b": [1.0]})
        prompt = build_prompt(df, schema, ["col_a", "col_b"])
        assert "id" in prompt
        assert "value" in prompt

    def test_contains_sample_data(self, schema):
        df = pl.DataFrame({"price": [42.5]})
        prompt = build_prompt(df, schema, ["price"])
        assert "price" in prompt
        assert "42" in prompt

    def test_contains_dsl_reference(self, schema):
        df = pl.DataFrame({"x": [1]})
        prompt = build_prompt(df, schema, ["x"])
        assert "col(" in prompt
        assert "str.to_datetime" in prompt

    def test_includes_example_configs(self, schema):
        df = pl.DataFrame({"x": [1]})
        ex = FormatConfig(name="ex", columns=[ColumnMapping(target="id", source="x")])
        prompt = build_prompt(df, schema, ["x"], example_configs=[ex])
        assert '"ex"' in prompt

    def test_format_name_in_prompt(self, schema):
        df = pl.DataFrame({"x": [1]})
        prompt = build_prompt(df, schema, ["x"], format_name="my_fmt")
        assert "my_fmt" in prompt

    def test_examples_note_includes_schema_name(self, schema):
        df = pl.DataFrame({"x": [1]})
        ex = FormatConfig(name="ex", columns=[ColumnMapping(target="id", source="x")])
        prompt = build_prompt(df, schema, ["x"], example_configs=[ex])
        assert "test" in prompt
        assert "same" in prompt

    def test_instructs_tool_use(self, schema):
        df = pl.DataFrame({"x": [1]})
        prompt = build_prompt(df, schema, ["x"])
        assert "submit_format_config" in prompt


class TestGenerateConfig:
    @pytest.fixture
    def csv_file(self, tmp_path):
        p = tmp_path / "data.csv"
        pl.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]}).write_csv(str(p))
        return str(p)

    @pytest.fixture
    def schema(self):
        return TargetSchema(
            name="students",
            columns=[
                TargetColumn(name="student", type="str", required=True, description="Student name"),
                TargetColumn(name="grade", type="float64", required=True, description="Grade"),
            ],
        )

    def _valid_args(self, name: str = "fmt") -> dict:
        return {
            "name": name,
            "columns": [
                {"target": "student", "source": "Name"},
                {"target": "grade", "source": "Score"},
            ],
        }

    def _bad_dsl_args(self) -> dict:
        return {
            "name": "f",
            "columns": [
                {"target": "student", "expr": "INVALID!!!"},
                {"target": "grade", "source": "Score"},
            ],
        }

    def _llm(self, *tool_call_args: dict | None) -> FakeToolCallingModel:
        """Build a fake LLM that makes tool calls in sequence then finishes."""
        responses: list[AIMessage] = []
        for i, args in enumerate(tool_call_args):
            if args is not None:
                responses.append(
                    AIMessage(
                        content="",
                        tool_calls=[{"id": f"tc{i}", "name": "submit_format_config", "args": args}],
                    )
                )
        responses.append(AIMessage(content="Done."))
        return FakeToolCallingModel(responses=responses)

    def test_successful_generation(self, csv_file, schema):
        config = generate_config(csv_file, schema, llm=self._llm(self._valid_args()))
        assert config.name == "fmt"
        assert len(config.columns) == 2

    def test_uses_path_stem_as_name(self, csv_file, schema):
        args = {
            "columns": [
                {"target": "student", "source": "Name"},
                {"target": "grade", "source": "Score"},
            ]
        }
        config = generate_config(csv_file, schema, llm=self._llm(args))
        assert config.name == "data"

    def test_retries_on_bad_dsl(self, csv_file, schema):
        config = generate_config(
            csv_file, schema, llm=self._llm(self._bad_dsl_args(), self._valid_args()), max_retries=2
        )
        assert config.name == "fmt"

    def test_raises_when_no_valid_config_submitted(self, csv_file, schema):
        llm = FakeToolCallingModel(responses=[AIMessage(content="I cannot help.")])
        with pytest.raises(LLMGenerationError):
            generate_config(csv_file, schema, llm=llm, max_retries=1)

    def test_error_includes_attempts(self, csv_file, schema):
        with pytest.raises(LLMGenerationError) as exc_info:
            generate_config(
                csv_file,
                schema,
                llm=self._llm(self._bad_dsl_args(), self._bad_dsl_args()),
                max_retries=1,
            )
        err = exc_info.value
        assert len(err.attempts) == 2
        assert all("response" in a and "error" in a for a in err.attempts)

    def test_warning_logged_on_validation_failure(self, csv_file, schema, caplog):
        with caplog.at_level(logging.WARNING, logger="schemashift.llm"):
            generate_config(
                csv_file,
                schema,
                llm=self._llm(self._bad_dsl_args(), self._valid_args()),
                max_retries=2,
            )
        assert any("attempt" in r.message.lower() for r in caplog.records)
