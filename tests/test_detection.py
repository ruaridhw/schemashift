"""Tests for format auto-detection."""

from __future__ import annotations

import pytest

from schemashift.errors import AmbiguousFormatError
from schemashift.detection import detect_format
from schemashift.models import ColumnMapping, FormatConfig
from schemashift.registry import DictRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(name: str, source_cols: list[str]) -> FormatConfig:
    """Build a FormatConfig that references the given source column names."""
    columns = [ColumnMapping(target=f"out_{c}", source=c) for c in source_cols]
    return FormatConfig(name=name, columns=columns)


def _make_expr_config(name: str, expr_cols: list[str]) -> FormatConfig:
    """Build a FormatConfig that references columns via DSL expr."""
    columns = [
        ColumnMapping(target="result", expr=" + ".join(f'col("{c}")' for c in expr_cols))
    ]
    return FormatConfig(name=name, columns=columns)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_exact_match_returns_config(self) -> None:
        reg = DictRegistry()
        cfg = _make_config("sales", ["id", "amount", "name"])
        reg.register(cfg)

        result = detect_format(["id", "amount", "name"], reg)
        assert result is not None
        assert result.name == "sales"

    def test_no_match_returns_none(self) -> None:
        reg = DictRegistry()
        cfg = _make_config("sales", ["order_id", "total"])
        reg.register(cfg)

        result = detect_format(["id", "amount", "name"], reg)
        assert result is None

    def test_ambiguous_raises_error(self) -> None:
        reg = DictRegistry()
        reg.register(_make_config("format_a", ["id", "amount"]))
        reg.register(_make_config("format_b", ["id", "amount"]))

        with pytest.raises(AmbiguousFormatError) as exc_info:
            detect_format(["id", "amount", "extra"], reg)

        assert "format_a" in exc_info.value.candidates
        assert "format_b" in exc_info.value.candidates

    def test_partial_match_file_has_extra_columns(self) -> None:
        """File has more columns than the config needs — should still match."""
        reg = DictRegistry()
        cfg = _make_config("minimal", ["id"])
        reg.register(cfg)

        result = detect_format(["id", "name", "amount", "category", "active"], reg)
        assert result is not None
        assert result.name == "minimal"

    def test_empty_registry_returns_none(self) -> None:
        reg = DictRegistry()
        result = detect_format(["id", "name"], reg)
        assert result is None

    def test_empty_file_columns_returns_none(self) -> None:
        reg = DictRegistry()
        reg.register(_make_config("cfg", ["id"]))
        result = detect_format([], reg)
        assert result is None

    def test_detect_with_expr_columns(self) -> None:
        """Config using DSL expr with col() references should still be detected."""
        reg = DictRegistry()
        cfg = _make_expr_config("expr_cfg", ["price", "quantity"])
        reg.register(cfg)

        result = detect_format(["price", "quantity", "product"], reg)
        assert result is not None
        assert result.name == "expr_cfg"

    def test_detect_expr_config_missing_column_no_match(self) -> None:
        reg = DictRegistry()
        cfg = _make_expr_config("expr_cfg", ["price", "quantity"])
        reg.register(cfg)

        result = detect_format(["price", "product"], reg)
        assert result is None

    def test_ambiguous_error_lists_all_candidates(self) -> None:
        reg = DictRegistry()
        reg.register(_make_config("a", ["x"]))
        reg.register(_make_config("b", ["x"]))
        reg.register(_make_config("c", ["x"]))

        with pytest.raises(AmbiguousFormatError) as exc_info:
            detect_format(["x", "y"], reg)

        assert len(exc_info.value.candidates) == 3

    def test_one_of_two_configs_matches(self) -> None:
        reg = DictRegistry()
        reg.register(_make_config("matching", ["col_a", "col_b"]))
        reg.register(_make_config("non_matching", ["col_c", "col_d"]))

        result = detect_format(["col_a", "col_b", "extra"], reg)
        assert result is not None
        assert result.name == "matching"
