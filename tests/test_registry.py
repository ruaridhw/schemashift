"""Tests for DictRegistry and FileSystemRegistry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemashift.models import ColumnMapping, FormatConfig
from schemashift.registry import DictRegistry, FileSystemRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(name: str = "test_config") -> FormatConfig:
    return FormatConfig(
        name=name,
        description="A test config",
        columns=[
            ColumnMapping(target="output_col", source="input_col"),
        ],
    )


# ---------------------------------------------------------------------------
# DictRegistry
# ---------------------------------------------------------------------------


class TestDictRegistry:
    def test_register_and_get(self) -> None:
        reg = DictRegistry()
        cfg = _make_config("alpha")
        reg.register(cfg)
        result = reg.get("alpha")
        assert result is not None
        assert result.name == "alpha"

    def test_get_unknown_returns_none(self) -> None:
        reg = DictRegistry()
        assert reg.get("nonexistent") is None

    def test_list_configs_empty(self) -> None:
        reg = DictRegistry()
        assert reg.list_configs() == []

    def test_list_configs_multiple(self) -> None:
        reg = DictRegistry()
        cfg_a = _make_config("a")
        cfg_b = _make_config("b")
        reg.register(cfg_a)
        reg.register(cfg_b)
        names = {c.name for c in reg.list_configs()}
        assert names == {"a", "b"}

    def test_register_overwrites_existing(self) -> None:
        reg = DictRegistry()
        reg.register(FormatConfig(name="x", description="v1", columns=[ColumnMapping(target="t", source="s")]))
        reg.register(FormatConfig(name="x", description="v2", columns=[ColumnMapping(target="t", source="s")]))
        result = reg.get("x")
        assert result is not None
        assert result.description == "v2"

    def test_delete_existing_returns_true(self) -> None:
        reg = DictRegistry()
        reg.register(_make_config("to_delete"))
        assert reg.delete("to_delete") is True
        assert reg.get("to_delete") is None

    def test_delete_nonexistent_returns_false(self) -> None:
        reg = DictRegistry()
        assert reg.delete("ghost") is False

    def test_delete_removes_from_list(self) -> None:
        reg = DictRegistry()
        reg.register(_make_config("keep"))
        reg.register(_make_config("remove"))
        reg.delete("remove")
        names = [c.name for c in reg.list_configs()]
        assert names == ["keep"]


# ---------------------------------------------------------------------------
# FileSystemRegistry
# ---------------------------------------------------------------------------


class TestFileSystemRegistry:
    def test_register_writes_json_file(self, tmp_path: Path) -> None:
        reg = FileSystemRegistry(tmp_path)
        cfg = _make_config("fs_config")
        reg.register(cfg)
        expected_file = tmp_path / "fs_config.json"
        assert expected_file.exists()
        data = json.loads(expected_file.read_text())
        assert data["name"] == "fs_config"

    def test_get_reads_back_config(self, tmp_path: Path) -> None:
        reg = FileSystemRegistry(tmp_path)
        cfg = _make_config("roundtrip")
        reg.register(cfg)
        result = reg.get("roundtrip")
        assert result is not None
        assert result.name == "roundtrip"
        assert result.description == cfg.description

    def test_get_unknown_returns_none(self, tmp_path: Path) -> None:
        reg = FileSystemRegistry(tmp_path)
        assert reg.get("missing") is None

    def test_list_configs(self, tmp_path: Path) -> None:
        reg = FileSystemRegistry(tmp_path)
        reg.register(_make_config("cfg1"))
        reg.register(_make_config("cfg2"))
        configs = reg.list_configs()
        names = {c.name for c in configs}
        assert names == {"cfg1", "cfg2"}

    def test_list_configs_empty_dir(self, tmp_path: Path) -> None:
        reg = FileSystemRegistry(tmp_path)
        assert reg.list_configs() == []

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        reg = FileSystemRegistry(tmp_path)
        reg.register(_make_config("deleteme"))
        assert reg.delete("deleteme") is True
        assert not (tmp_path / "deleteme.json").exists()

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        reg = FileSystemRegistry(tmp_path)
        assert reg.delete("ghost") is False

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        reg = FileSystemRegistry(nested)
        assert nested.is_dir()
        reg.register(_make_config("nested_cfg"))
        assert reg.get("nested_cfg") is not None

    def test_register_overwrites_existing_file(self, tmp_path: Path) -> None:
        reg = FileSystemRegistry(tmp_path)
        reg.register(FormatConfig(name="upd", description="v1", columns=[ColumnMapping(target="t", source="s")]))
        reg.register(FormatConfig(name="upd", description="v2", columns=[ColumnMapping(target="t", source="s")]))
        result = reg.get("upd")
        assert result is not None
        assert result.description == "v2"
