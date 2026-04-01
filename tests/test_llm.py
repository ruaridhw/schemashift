"""Tests for schemashift.llm — LLM config generation engine."""

import json

import polars as pl
import pytest

from schemashift.errors import LLMGenerationError
from schemashift.llm import _extract_json, build_prompt, generate_config
from schemashift.models import ColumnMapping, FormatConfig
from schemashift.target_schema import TargetColumn, TargetSchema


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

    def _valid_response(self, name="fmt"):
        return json.dumps(
            {
                "name": name,
                "columns": [
                    {"target": "student", "source": "Name"},
                    {"target": "grade", "source": "Score"},
                ],
            }
        )

    def test_successful_generation(self, csv_file, schema):
        config = generate_config(csv_file, schema, llm_fn=lambda p: self._valid_response())
        assert config.name == "fmt"
        assert len(config.columns) == 2

    def test_uses_path_stem_as_name(self, csv_file, schema):
        def respond(p):
            return json.dumps(
                {
                    "columns": [
                        {"target": "student", "source": "Name"},
                        {"target": "grade", "source": "Score"},
                    ]
                }
            )

        config = generate_config(csv_file, schema, llm_fn=respond)
        assert config.name == "data"  # stem of data.csv

    def test_retries_on_bad_json(self, csv_file, schema):
        calls = []

        def respond(p):
            calls.append(p)
            if len(calls) == 1:
                return "not json"
            return self._valid_response()

        config = generate_config(csv_file, schema, llm_fn=respond, max_retries=2)
        assert len(calls) == 2

    def test_retries_on_bad_dsl(self, csv_file, schema):
        calls = []

        def respond(p):
            calls.append(p)
            if len(calls) == 1:
                return json.dumps(
                    {
                        "name": "f",
                        "columns": [
                            {"target": "student", "expr": "INVALID!!!"},
                            {"target": "grade", "source": "Score"},
                        ],
                    }
                )
            return self._valid_response()

        config = generate_config(csv_file, schema, llm_fn=respond, max_retries=2)
        assert len(calls) == 2

    def test_raises_after_exhausted_retries(self, csv_file, schema):
        with pytest.raises(LLMGenerationError):
            generate_config(csv_file, schema, llm_fn=lambda p: "bad", max_retries=1)

    def test_raises_without_llm(self, csv_file, schema):
        with pytest.raises(ValueError, match="llm"):
            generate_config(csv_file, schema)

    def test_langchain_style_llm(self, csv_file, schema):
        class FakeLLM:
            def invoke(self, prompt):
                class Msg:
                    content = json.dumps(
                        {
                            "name": "lc",
                            "columns": [
                                {"target": "student", "source": "Name"},
                                {"target": "grade", "source": "Score"},
                            ],
                        }
                    )

                return Msg()

        config = generate_config(csv_file, schema, llm=FakeLLM())
        assert config.name == "lc"
