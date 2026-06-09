"""Integration tests using the PRD worked examples."""

import json
from pathlib import Path

import polars as pl
import pytest

from schemashift.detection import detect_format
from schemashift.models import TransformSpec
from schemashift.registry import DictRegistry, FileSystemRegistry
from schemashift.transform import transform, validate_config
from schemashift.validation import SchemaConfig

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples"


class TestCamstarLotMovement:
    """End-to-end test for Camstar MES CSV format."""

    @pytest.fixture
    def config(self) -> TransformSpec:
        path = EXAMPLES / "configs" / "camstar_lot_movement.json"
        return TransformSpec.model_validate(json.loads(path.read_text()))

    @pytest.fixture
    def schema(self) -> SchemaConfig:
        return SchemaConfig.from_yaml(FIXTURES / "configs" / "lot_movement_schema.yaml")

    @pytest.fixture
    def csv_path(self) -> str:
        return str(FIXTURES / "csv" / "camstar_sample.csv")

    def test_config_loads_and_validates(self, config: TransformSpec) -> None:
        errors = validate_config(config)
        assert errors == []

    def test_transform_produces_canonical_columns(
        self, config: TransformSpec, csv_path: str, schema: SchemaConfig
    ) -> None:
        result = transform(csv_path, config)
        canonical_cols = set(schema.columns.keys())
        assert canonical_cols == set(result.valid.columns)

    def test_lot_ids_correct(self, config: TransformSpec, csv_path: str) -> None:
        result = transform(csv_path, config)
        assert result.valid["lot_id"].to_list() == ["L001234", "L001235", "L001236"]

    def test_wafer_count_int32(self, config: TransformSpec, csv_path: str) -> None:
        result = transform(csv_path, config)
        assert result.valid["wafer_count"].dtype == pl.Int32
        assert result.valid["wafer_count"].to_list() == [25, 24, 25]

    def test_hold_flag_boolean(self, config: TransformSpec, csv_path: str) -> None:
        result = transform(csv_path, config)
        assert result.valid["hold_flag"].dtype == pl.Boolean
        assert result.valid["hold_flag"].to_list() == [False, True, False]

    def test_data_source_constant(self, config: TransformSpec, csv_path: str) -> None:
        result = transform(csv_path, config)
        assert result.valid["data_source"].n_unique() == 1
        assert result.valid["data_source"][0] == "camstar"

    def test_track_in_time_is_datetime(self, config: TransformSpec, csv_path: str) -> None:
        result = transform(csv_path, config)
        assert isinstance(result.valid["track_in_time"].dtype, pl.Datetime)


class TestFabxLotMovement:
    """End-to-end test for FabX TSV format with Unix timestamps."""

    @pytest.fixture
    def config(self) -> TransformSpec:
        path = EXAMPLES / "configs" / "fabx_lot_movement.json"
        return TransformSpec.model_validate(json.loads(path.read_text()))

    @pytest.fixture
    def tsv_path(self) -> str:
        return str(FIXTURES / "tsv" / "fabx_sample.tsv")

    def test_config_loads(self, config: TransformSpec) -> None:
        assert config.name == "fabx_lot_movement"
        assert config.reader.separator == "\t"

    def test_transform_produces_expected_columns(self, config: TransformSpec, tsv_path: str) -> None:
        result = transform(tsv_path, config)
        expected_cols = {
            "lot_id",
            "wafer_count",
            "operation",
            "step_sequence",
            "tool_id",
            "track_in_time",
            "track_out_time",
            "recipe",
            "route",
            "priority",
            "hold_flag",
            "data_source",
        }
        assert expected_cols == set(result.valid.columns)

    def test_lot_ids(self, config: TransformSpec, tsv_path: str) -> None:
        result = transform(tsv_path, config)
        assert result.valid["lot_id"].to_list() == ["L001234", "L001235", "L001236"]

    def test_hold_flag_from_integer(self, config: TransformSpec, tsv_path: str) -> None:
        result = transform(tsv_path, config)
        assert result.valid["hold_flag"].dtype == pl.Boolean
        assert result.valid["hold_flag"].to_list() == [False, True, False]

    def test_data_source_fabx(self, config: TransformSpec, tsv_path: str) -> None:
        result = transform(tsv_path, config)
        assert result.valid["data_source"][0] == "fabx"

    def test_priority_int32(self, config: TransformSpec, tsv_path: str) -> None:
        result = transform(tsv_path, config)
        assert result.valid["priority"].dtype == pl.Int32
        assert result.valid["priority"].to_list() == [1, 3, 2]

    def test_track_in_time_is_datetime(self, config: TransformSpec, tsv_path: str) -> None:
        result = transform(tsv_path, config)
        assert isinstance(result.valid["track_in_time"].dtype, pl.Datetime)


class TestAutoDetection:
    """Test format auto-detection from registry."""

    @pytest.fixture
    def registry(self) -> DictRegistry:
        r = DictRegistry()
        camstar_path = EXAMPLES / "configs" / "camstar_lot_movement.json"
        fabx_path = EXAMPLES / "configs" / "fabx_lot_movement.json"
        r.register(TransformSpec.model_validate(json.loads(camstar_path.read_text())))
        r.register(TransformSpec.model_validate(json.loads(fabx_path.read_text())))
        return r

    def test_detects_camstar_from_columns(self, registry: DictRegistry) -> None:
        camstar_columns = [
            "LOT_ID",
            "CARRIER_ID",
            "QTY",
            "CURRENT_OPER",
            "OPER_SEQ",
            "RESOURCE",
            "TRACKIN_DT",
            "TRACKOUT_DT",
            "RECIPE_NAME",
            "FLOW",
            "LOT_PRIORITY",
            "HOLD_STATUS",
        ]
        config = detect_format(camstar_columns, registry)
        assert config is not None
        assert config.name == "camstar_lot_movement"

    def test_detects_fabx_from_columns(self, registry: DictRegistry) -> None:
        fabx_columns = [
            "lot",
            "wfrs",
            "op_code",
            "seq",
            "tool",
            "recipe",
            "t_in",
            "t_out",
            "route",
            "prio",
            "hold",
        ]
        config = detect_format(fabx_columns, registry)
        assert config is not None
        assert config.name == "fabx_lot_movement"

    def test_no_match_returns_none(self, registry: DictRegistry) -> None:
        unknown_columns = ["foo", "bar", "baz"]
        config = detect_format(unknown_columns, registry)
        assert config is None

    def test_filesystem_registry_with_examples(self) -> None:
        examples_dir = EXAMPLES / "configs"
        reg = FileSystemRegistry(examples_dir)
        configs = reg.list_configs()
        names = {c.name for c in configs}
        assert "camstar_lot_movement" in names
        assert "fabx_lot_movement" in names
