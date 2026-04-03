"""Tests for Excel reading edge cases."""

import pytest

from schemashift.models import ColumnMapping, FormatConfig, ReaderConfig
from schemashift.transform import transform


def _create_xlsx(tmp_path, rows, sheet_name="Sheet1", header_rows_before=0):
    """Helper to create an XLSX file using openpyxl."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name

    # Add empty/metadata rows before real data if requested
    for _ in range(header_rows_before):
        ws.append(["", "", ""])  # junk rows

    # Write rows (first row is header)
    for row in rows:
        ws.append(row)

    path = tmp_path / "test.xlsx"
    wb.save(str(path))
    return str(path)


class TestExcelBasic:
    def test_simple_excel_read(self, tmp_path):
        path = _create_xlsx(
            tmp_path,
            [
                ["id", "name", "value"],
                [1, "Alice", 100],
                [2, "Bob", 200],
            ],
        )
        config = FormatConfig(
            name="t",
            reader=ReaderConfig(sheet_name="Sheet1"),
            columns=[
                ColumnMapping(target="user_id", source="id"),
                ColumnMapping(target="user_name", source="name"),
            ],
        )
        result = transform(path, config).collect()
        assert set(result.columns) == {"user_id", "user_name"}
        assert len(result) == 2
        assert result["user_name"].to_list() == ["Alice", "Bob"]

    def test_excel_with_skip_rows(self, tmp_path):
        """Excel with junk header rows before real data."""
        path = _create_xlsx(
            tmp_path,
            [
                ["Report Title", "", ""],  # skip
                ["Generated: 2025-01-01", "", ""],  # skip
                ["", "", ""],  # skip
                ["id", "name", "value"],  # real header (row index 3)
                [1, "Alice", 100],
                [2, "Bob", 200],
            ],
            header_rows_before=0,
        )

        config = FormatConfig(
            name="t",
            reader=ReaderConfig(sheet_name="Sheet1", skip_rows=3),
            columns=[
                ColumnMapping(target="user_id", source="id"),
                ColumnMapping(target="user_name", source="name"),
            ],
        )
        result = transform(path, config).collect()
        assert set(result.columns) == {"user_id", "user_name"}
        assert len(result) == 2

    def test_excel_named_sheet(self, tmp_path):
        """Read from a specific named sheet."""
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.title = "Metadata"
        ws1.append(["This is not the data sheet"])

        ws2 = wb.create_sheet("WIP_Detail")
        ws2.append(["lot_id", "wafer_count", "operation"])
        ws2.append(["L001", 25, "ETCH_M1"])
        ws2.append(["L002", 24, "LITHO_M2"])

        path = tmp_path / "multi_sheet.xlsx"
        wb.save(str(path))

        config = FormatConfig(
            name="t",
            reader=ReaderConfig(sheet_name="WIP_Detail"),
            columns=[
                ColumnMapping(target="lot", source="lot_id"),
                ColumnMapping(target="wfrs", source="wafer_count"),
            ],
        )
        result = transform(str(path), config).collect()
        assert set(result.columns) == {"lot", "wfrs"}
        assert len(result) == 2
        assert result["lot"].to_list() == ["L001", "L002"]

    def test_excel_string_sheet_index(self, tmp_path):
        """Sheet name as integer index."""
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"
        ws.append(["col_a", "col_b"])
        ws.append(["x", 1])
        path = tmp_path / "indexed.xlsx"
        wb.save(str(path))

        config = FormatConfig(
            name="t",
            reader=ReaderConfig(sheet_name=0),  # first sheet by index
            columns=[ColumnMapping(target="a", source="col_a")],
        )
        result = transform(str(path), config).collect()
        assert result["a"].to_list() == ["x"]


class TestExcelFromPRD:
    """Test the SmartFactory example from the PRD (Section Appendix A)."""

    def test_smartfactory_lot_movement(self, tmp_path):
        """Reproduce the SmartFactory XLSX format from PRD Appendix A."""
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "WIP_Detail"

        # 3 header/metadata rows (to be skipped)
        ws.append(["SmartFactory Export", "", "", "", "", "", "", "", "", "", ""])
        ws.append(["Site: FAB1", "", "", "", "", "", "", "", "", "", ""])
        ws.append(["", "", "", "", "", "", "", "", "", "", ""])
        # Real header
        ws.append(
            [
                "Lot Number",
                "Wfr Cnt",
                "Step Name",
                "Step#",
                "Equip ID",
                "Recipe",
                "Arrival Time",
                "Departure Time",
                "Route Name",
                "Pri",
                "Status",
            ]
        )
        # Data rows
        ws.append(
            [
                "W.L001234",
                25,
                "M1-Etch",
                45,
                "ETCH-07A",
                "M1_ETCH_V3",
                "03/15/2025 08:12 AM",
                "03/15/2025 09:45 AM",
                "LOGIC28",
                "HOT",
                "Run",
            ]
        )
        ws.append(
            [
                "W.L001235",
                24,
                "M2-Litho",
                52,
                "ASML-1980",
                "M2_LITHO_EUV",
                "03/15/2025 08:30 AM",
                None,
                "LOGIC28",
                "NORMAL",
                "Hold",
            ]
        )

        path = tmp_path / "smartfactory.xlsx"
        wb.save(str(path))

        config = FormatConfig(
            name="smartfactory_lot_movement",
            reader=ReaderConfig(skip_rows=3, sheet_name="WIP_Detail"),
            columns=[
                ColumnMapping(
                    target="lot_id",
                    expr='col("Lot Number").str.replace("W.", "")',
                ),
                ColumnMapping(target="wafer_count", source="Wfr Cnt", dtype="int32"),
                ColumnMapping(target="operation", source="Step Name"),
                ColumnMapping(target="tool_id", source="Equip ID"),
                ColumnMapping(target="recipe", source="Recipe"),
                ColumnMapping(target="route", source="Route Name"),
                ColumnMapping(
                    target="priority",
                    expr=('when(col("Pri") == "HOT", 1).when(col("Pri") == "NORMAL", 3).otherwise(5)'),
                ),
                ColumnMapping(target="hold_flag", expr='col("Status") == "Hold"'),
                ColumnMapping(target="data_source", constant="smartfactory"),
            ],
        )
        result = transform(str(path), config).collect()
        assert "lot_id" in result.columns
        assert "hold_flag" in result.columns
        assert "data_source" in result.columns
        assert result["lot_id"].to_list()[0] == "L001234"
        assert result["hold_flag"].to_list() == [False, True]
        assert result["data_source"].n_unique() == 1
        assert result["data_source"][0] == "smartfactory"
        assert result["priority"].to_list()[0] == 1
        assert result["priority"].to_list()[1] == 3
