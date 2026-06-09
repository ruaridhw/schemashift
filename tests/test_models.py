"""Tests for schemashift.models."""

import json

import pytest
from pydantic import ValidationError

from schemashift.errors import ConfigValidationError
from schemashift.models import ColumnMapping, FormatConfig, ReaderConfig


class TestColumnMappingValidation:
    def test_source_only_is_valid(self):
        col = ColumnMapping(target="out", source="in")
        assert col.source == "in"
        assert col.expr is None
        assert not col.has_constant()

    def test_expr_only_is_valid(self):
        col = ColumnMapping(target="out", expr='col("price") * 1.2')
        assert col.expr == 'col("price") * 1.2'
        assert col.source is None
        assert not col.has_constant()

    def test_constant_only_is_valid(self):
        col = ColumnMapping(target="flag", constant=True)
        assert col.constant is True
        assert col.source is None
        assert col.expr is None

    def test_constant_zero_is_valid(self):
        col = ColumnMapping(target="val", constant=0)
        assert col.constant == 0

    def test_constant_none_string_requires_source_or_expr(self):
        # constant=None means it is not set; must provide source or expr
        with pytest.raises(ConfigValidationError, match="exactly one"):
            ColumnMapping(target="out")

    def test_none_set_raises_config_error(self):
        with pytest.raises(ConfigValidationError, match="exactly one"):
            ColumnMapping(target="out")

    def test_source_and_expr_raises_config_error(self):
        with pytest.raises(ConfigValidationError, match="exactly one"):
            ColumnMapping(target="out", source="in", expr='col("in")')

    def test_source_and_constant_raises_config_error(self):
        with pytest.raises(ConfigValidationError, match="exactly one"):
            ColumnMapping(target="out", source="in", constant=42)

    def test_expr_and_constant_raises_config_error(self):
        with pytest.raises(ConfigValidationError, match="exactly one"):
            ColumnMapping(target="out", expr='col("x")', constant=42)

    def test_all_three_raises_config_error(self):
        with pytest.raises(ConfigValidationError, match="exactly one"):
            ColumnMapping(target="out", source="in", expr='col("in")', constant=0)

    def test_constant_none_is_valid(self) -> None:
        col = ColumnMapping(target="flag", constant=None)
        assert col.has_constant() is True
        assert col.constant is None

    def test_constant_none_round_trips_json(self) -> None:
        config = FormatConfig(
            name="test",
            columns=[ColumnMapping(target="flag", constant=None)],
        )
        dumped = json.loads(config.model_dump_json())
        reloaded = FormatConfig.model_validate(dumped)
        assert reloaded.columns[0].constant is None
        assert reloaded.columns[0].has_constant()


class TestColumnMappingDtypeValidation:
    @pytest.mark.parametrize(
        "dtype",
        [
            "str",
            "utf8",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "bool",
            "date",
            "datetime",
            "time",
            "duration",
            "binary",
            "categorical",
            "null",
        ],
    )
    def test_valid_dtype_accepted(self, dtype):
        col = ColumnMapping(target="out", source="in", dtype=dtype)
        assert col.dtype == dtype

    def test_invalid_dtype_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ColumnMapping(target="out", source="in", dtype="bigint")

    def test_none_dtype_allowed(self):
        col = ColumnMapping(target="out", source="in", dtype=None)
        assert col.dtype is None

    def test_fillna_stored(self):
        col = ColumnMapping(target="out", source="in", fillna=0)
        assert col.fillna == 0


class TestReaderConfig:
    def test_defaults(self):
        cfg = ReaderConfig()
        assert cfg.skip_rows == 0
        assert cfg.sheet_name is None
        assert cfg.separator is None
        assert cfg.encoding == "utf-8"

    def test_custom_values(self):
        cfg = ReaderConfig(skip_rows=2, sheet_name="Data", separator="|", encoding="latin-1")
        assert cfg.skip_rows == 2
        assert cfg.sheet_name == "Data"
        assert cfg.separator == "|"
        assert cfg.encoding == "latin-1"

    def test_sheet_name_as_int(self):
        cfg = ReaderConfig(sheet_name=1)
        assert cfg.sheet_name == 1


class TestFormatConfigValidation:
    def _make_config(self, columns=None, **kwargs):
        if columns is None:
            columns = [
                ColumnMapping(target="a", source="col_a"),
                ColumnMapping(target="b", expr='col("col_b") * 2'),
            ]
        return FormatConfig(name="test", columns=columns, **kwargs)

    def test_valid_config_created(self):
        cfg = self._make_config()
        assert cfg.name == "test"
        assert cfg.version == 1
        assert cfg.drop_unmapped is True

    def test_default_reader_config(self):
        cfg = self._make_config()
        assert isinstance(cfg.reader, ReaderConfig)

    def test_duplicate_targets_raise_config_error(self):
        cols = [
            ColumnMapping(target="dup", source="a"),
            ColumnMapping(target="dup", source="b"),
        ]
        with pytest.raises(ConfigValidationError, match="duplicate target"):
            FormatConfig(name="bad", columns=cols)

    def test_unique_targets_pass(self):
        cols = [
            ColumnMapping(target="x", source="a"),
            ColumnMapping(target="y", source="b"),
        ]
        cfg = FormatConfig(name="ok", columns=cols)
        assert len(cfg.columns) == 2


class TestFormatConfigSourceColumns:
    def test_source_fields_extracted(self):
        cols = [
            ColumnMapping(target="out_a", source="raw_a"),
            ColumnMapping(target="out_b", source="raw_b"),
        ]
        cfg = FormatConfig(name="f", columns=cols)
        assert cfg.source_columns() == {"raw_a", "raw_b"}

    def test_col_references_in_expr_extracted(self):
        cols = [
            ColumnMapping(target="total", expr='col("price") * col("qty")'),
        ]
        cfg = FormatConfig(name="f", columns=cols)
        assert cfg.source_columns() == {"price", "qty"}

    def test_mix_of_source_and_expr(self):
        cols = [
            ColumnMapping(target="name", source="full_name"),
            ColumnMapping(target="tax", expr='col("amount") * 0.2'),
            ColumnMapping(target="flag", constant=True),
        ]
        cfg = FormatConfig(name="f", columns=cols)
        assert cfg.source_columns() == {"full_name", "amount"}

    def test_constant_columns_not_included(self):
        cols = [ColumnMapping(target="status", constant="active")]
        cfg = FormatConfig(name="f", columns=cols)
        assert cfg.source_columns() == set()

    def test_single_quoted_col_refs_extracted(self):
        cfg = FormatConfig(
            name="f",
            columns=[ColumnMapping(target="out", expr="col('price') * col('qty')")],
        )
        assert cfg.source_columns() == {"price", "qty"}

    def test_nested_expr_col_refs_extracted(self):
        cfg = FormatConfig(
            name="f",
            columns=[ColumnMapping(target="out", expr='coalesce(col("a"), col("b"))')],
        )
        assert cfg.source_columns() == {"a", "b"}


class TestFormatConfigJsonRoundTrip:
    def test_round_trip_via_model_dump_and_validate(self):
        original = FormatConfig(
            name="invoice",
            description="Invoice format",
            version=2,
            reader=ReaderConfig(skip_rows=1, separator=";"),
            columns=[
                ColumnMapping(target="date", source="invoice_date", dtype="date"),
                ColumnMapping(target="total", expr='col("net") + col("tax")', fillna=0.0),
                ColumnMapping(target="status", constant="pending"),
            ],
            drop_unmapped=False,
        )
        data = original.model_dump()
        restored = FormatConfig.model_validate(data)

        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.version == original.version
        assert restored.drop_unmapped == original.drop_unmapped
        assert restored.reader.skip_rows == 1
        assert restored.reader.separator == ";"
        assert len(restored.columns) == 3
        assert restored.columns[0].target == "date"
        assert restored.columns[0].source == "invoice_date"
        assert restored.columns[0].dtype == "date"
        assert restored.columns[1].expr == 'col("net") + col("tax")'
        assert restored.columns[1].fillna == pytest.approx(0.0)
        assert restored.columns[2].constant == "pending"


# ---------------------------------------------------------------------------
# FormatConfig.target_schema
# ---------------------------------------------------------------------------


class TestFormatConfigTargetSchema:
    def _minimal_columns(self) -> list[dict]:
        return [{"target": "out", "source": "in"}]

    def test_target_schema_accepted(self) -> None:
        cfg = FormatConfig(
            name="test",
            target_schema="lot_movement",
            columns=self._minimal_columns(),
        )
        assert cfg.target_schema == "lot_movement"

    def test_target_schema_defaults_to_none(self) -> None:
        cfg = FormatConfig(name="test", columns=self._minimal_columns())
        assert cfg.target_schema is None

    def test_target_schema_round_trips_json(self) -> None:
        original = FormatConfig(
            name="test",
            target_schema="lot_movement",
            columns=self._minimal_columns(),
        )
        data = original.model_dump()
        restored = FormatConfig.model_validate(data)
        assert restored.target_schema == "lot_movement"

    def test_target_schema_none_round_trips_json(self) -> None:
        original = FormatConfig(name="test", columns=self._minimal_columns())
        data = original.model_dump()
        restored = FormatConfig.model_validate(data)
        assert restored.target_schema is None
