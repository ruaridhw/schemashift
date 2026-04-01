"""Property-based tests for schemashift using Hypothesis."""

from __future__ import annotations

import os
import tempfile

import polars as pl
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from schemashift.dsl import parse_and_compile
from schemashift.dsl.parser import parse_dsl
from schemashift.errors import DSLSyntaxError
from schemashift.models import ColumnMapping, FormatConfig
from schemashift.target_schema import TargetSchema
from schemashift.transform import transform

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_lower_ident_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_",
    min_size=1,
    max_size=15,
)

_col_name_short_st = st.text(
    alphabet="abcde_",
    min_size=1,
    max_size=8,
)


# ---------------------------------------------------------------------------
# 1. FormatConfig JSON round-trip
# ---------------------------------------------------------------------------


class TestFormatConfigJsonRoundtrip:
    @given(
        targets=st.lists(
            _lower_ident_st,
            min_size=1,
            max_size=10,
            unique=True,
        )
    )
    @settings(max_examples=50)
    def test_source_only_config_roundtrips_via_json(self, targets: list[str]) -> None:
        columns = [ColumnMapping(target=t, source=f"src_{t}") for t in targets]
        config = FormatConfig(name="test", columns=columns)

        json_str = config.model_dump_json()
        restored = FormatConfig.model_validate_json(json_str)

        assert config.name == restored.name
        assert len(config.columns) == len(restored.columns)
        for orig, rest in zip(config.columns, restored.columns, strict=True):
            assert orig.target == rest.target
            assert orig.source == rest.source

    @given(
        targets=st.lists(
            _lower_ident_st,
            min_size=1,
            max_size=10,
            unique=True,
        )
    )
    @settings(max_examples=50)
    def test_constant_only_config_roundtrips_via_json(self, targets: list[str]) -> None:
        columns = [ColumnMapping(target=t, constant="value") for t in targets]
        config = FormatConfig(name="test_const", columns=columns)

        json_str = config.model_dump_json()
        restored = FormatConfig.model_validate_json(json_str)

        assert config.name == restored.name
        assert len(config.columns) == len(restored.columns)
        for orig, rest in zip(config.columns, restored.columns, strict=True):
            assert orig.target == rest.target
            assert orig.constant == rest.constant


# ---------------------------------------------------------------------------
# 2. Source-only config preserves source columns → target columns
# ---------------------------------------------------------------------------


class TestSourceOnlyConfigProducesTargetColumns:
    @given(
        targets=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    @settings(max_examples=50)
    def test_output_has_exactly_target_columns(self, targets: list[str]) -> None:
        source_cols = {f"src_{t}": ["val"] for t in targets}
        df = pl.DataFrame(source_cols)

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "data.csv")
        df.write_csv(path)

        columns = [ColumnMapping(target=t, source=f"src_{t}") for t in targets]
        config = FormatConfig(name="test", columns=columns)

        result = transform(path, config).collect()
        assert set(result.columns) == set(targets)

    @given(
        targets=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    @settings(max_examples=50)
    def test_output_row_count_matches_input(self, targets: list[str]) -> None:
        n_rows = 3
        source_cols = {f"src_{t}": ["a", "b", "c"] for t in targets}
        df = pl.DataFrame(source_cols)

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "data.csv")
        df.write_csv(path)

        columns = [ColumnMapping(target=t, source=f"src_{t}") for t in targets]
        config = FormatConfig(name="test", columns=columns)

        result = transform(path, config).collect()
        assert len(result) == n_rows


# ---------------------------------------------------------------------------
# 3. Valid DSL expressions parse and compile without crashing
# ---------------------------------------------------------------------------


class TestValidDslExpressions:
    @given(
        col_name=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz_",
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_col_ref_always_parses(self, col_name: str) -> None:
        expr = f'col("{col_name}")'
        result = parse_and_compile(expr)
        assert isinstance(result, pl.Expr)

    @given(
        left=_col_name_short_st,
        right=_col_name_short_st,
        op=st.sampled_from(["+", "-", "*"]),
    )
    @settings(max_examples=50)
    def test_binary_arithmetic_always_parses(self, left: str, right: str, op: str) -> None:
        expr = f'col("{left}") {op} col("{right}")'
        result = parse_and_compile(expr)
        assert isinstance(result, pl.Expr)

    @given(col_name=_col_name_short_st)
    @settings(max_examples=50)
    def test_col_div_constant_parses(self, col_name: str) -> None:
        for const in [1, 100, 1000, 0.5]:
            expr = f'col("{col_name}") / {const}'
            result = parse_and_compile(expr)
            assert isinstance(result, pl.Expr)


# ---------------------------------------------------------------------------
# 4. DSL parser never crashes with unexpected exceptions
# ---------------------------------------------------------------------------


class TestDslParserNeverCrashes:
    @given(text=st.text(max_size=50))
    @settings(max_examples=200)
    def test_dsl_parser_raises_dsl_syntax_error_or_returns_ast(self, text: str) -> None:
        """Parser must either return an ASTNode or raise DSLSyntaxError — never crash."""
        try:
            parse_dsl(text)
        except DSLSyntaxError:
            pass  # Expected for invalid input
        except Exception as exc:
            pytest.fail(
                f"Parser raised unexpected {type(exc).__name__}: {exc!r} for input {text!r}"
            )


# ---------------------------------------------------------------------------
# 5. Constant mappings produce correct row counts
# ---------------------------------------------------------------------------


class TestConstantMappingRowCounts:
    @given(
        n_rows=st.integers(min_value=1, max_value=100),
        const_value=st.one_of(
            st.integers(),
            st.text(min_size=0, max_size=20),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50)
    def test_constant_mapping_fills_all_rows(self, n_rows: int, const_value: object) -> None:
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "data.csv")
        pl.DataFrame({"x": range(n_rows)}).write_csv(path)

        # Include a source mapping alongside the constant so that Polars
        # select() can broadcast the literal to the correct row count.
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="id", source="x"),
                ColumnMapping(target="out", constant=str(const_value)),
            ],
        )
        result = transform(path, config).collect()
        assert len(result) == n_rows
        assert "out" in result.columns

    @given(
        n_rows=st.integers(min_value=1, max_value=50),
        n_cols=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30)
    def test_multiple_constant_columns_all_filled(self, n_rows: int, n_cols: int) -> None:
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "data.csv")
        pl.DataFrame({"x": range(n_rows)}).write_csv(path)

        # Anchor the row count with a source mapping so Polars can broadcast
        # all constant literals to the correct length.
        columns = [ColumnMapping(target="id", source="x")] + [
            ColumnMapping(target=f"col_{i}", constant="fixed") for i in range(n_cols)
        ]
        config = FormatConfig(name="t", columns=columns)
        result = transform(path, config).collect()

        assert len(result) == n_rows
        for i in range(n_cols):
            assert f"col_{i}" in result.columns


# ---------------------------------------------------------------------------
# 6. TargetSchema YAML round-trip
# ---------------------------------------------------------------------------


class TestTargetSchemaYamlRoundtrip:
    @given(
        col_names=st.lists(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz_",
                min_size=1,
                max_size=12,
            ),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    @settings(max_examples=50)
    def test_schema_roundtrips_via_yaml(self, col_names: list[str]) -> None:
        schema_data = {
            "name": "test_schema",
            "description": "property test",
            "columns": [
                {
                    "name": n,
                    "type": "str",
                    "required": True,
                    "description": f"column {n}",
                }
                for n in col_names
            ],
        }

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "schema.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(schema_data, f)

        schema = TargetSchema.from_yaml(path)
        assert schema.name == "test_schema"
        assert [c.name for c in schema.columns] == col_names

    @given(
        col_names=st.lists(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz_",
                min_size=1,
                max_size=12,
            ),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    @settings(max_examples=30)
    def test_schema_required_columns_match(self, col_names: list[str]) -> None:
        schema_data = {
            "name": "req_schema",
            "columns": [
                {"name": n, "type": "str", "required": True} for n in col_names
            ],
        }

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "schema.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(schema_data, f)

        schema = TargetSchema.from_yaml(path)
        assert schema.required_columns() == col_names
