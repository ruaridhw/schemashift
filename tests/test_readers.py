"""Tests for schemashift.readers."""

from pathlib import Path

import polars as pl
import pytest

from schemashift.errors import UnsupportedFileError
from schemashift.models import ReaderConfig
from schemashift.readers import read_file, read_header

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CSV_SAMPLE = FIXTURES_DIR / "csv" / "sample.csv"
TSV_SAMPLE = FIXTURES_DIR / "tsv" / "sample.tsv"
JSON_SAMPLE = FIXTURES_DIR / "json" / "sample.json"

EXPECTED_COLUMNS = ["id", "name", "amount", "category", "active"]
EXPECTED_ROW_COUNT = 5


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    df = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64),
            "name": pl.Series(["Alice", "Bob", "Carol", "Dave", "Eve"], dtype=pl.Utf8),
            "amount": pl.Series([100.50, 250.00, 75.25, 320.00, 45.99], dtype=pl.Float64),
            "category": pl.Series(["food", "transport", "food", "utilities", "food"], dtype=pl.Utf8),
            "active": pl.Series([True, False, True, True, False], dtype=pl.Boolean),
        }
    )
    path = tmp_path / "sample.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def xlsx_file(tmp_path: Path) -> Path:
    """Create a small Excel fixture programmatically (requires fastexcel/openpyxl)."""
    pytest.importorskip("openpyxl")
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["id", "name", "amount", "category", "active"])
    rows = [
        (1, "Alice", 100.50, "food", True),
        (2, "Bob", 250.00, "transport", False),
        (3, "Carol", 75.25, "food", True),
        (4, "Dave", 320.00, "utilities", True),
        (5, "Eve", 45.99, "food", False),
    ]
    for row in rows:
        ws.append(list(row))
    path = tmp_path / "sample.xlsx"
    wb.save(path)
    return path


@pytest.fixture
def csv_with_skip_rows(tmp_path: Path) -> Path:
    content = "# comment line\n# another comment\nid,name,value\n1,Alice,10\n2,Bob,20\n"
    path = tmp_path / "skipped.csv"
    path.write_text(content)
    return path


class TestReadFileCsv:
    def test_returns_lazy_frame(self):
        lf = read_file(CSV_SAMPLE)
        assert isinstance(lf, pl.LazyFrame)

    def test_correct_columns(self):
        lf = read_file(CSV_SAMPLE)
        assert lf.collect_schema().names() == EXPECTED_COLUMNS

    def test_correct_row_count(self):
        lf = read_file(CSV_SAMPLE)
        assert lf.collect().height == EXPECTED_ROW_COUNT

    def test_with_no_config(self):
        lf = read_file(CSV_SAMPLE, config=None)
        assert lf.collect().height == EXPECTED_ROW_COUNT

    def test_with_explicit_separator(self):
        cfg = ReaderConfig(separator=",")
        lf = read_file(CSV_SAMPLE, config=cfg)
        assert lf.collect_schema().names() == EXPECTED_COLUMNS

    def test_skip_rows_skips_header_rows(self, csv_with_skip_rows):
        cfg = ReaderConfig(skip_rows=2)
        lf = read_file(csv_with_skip_rows, config=cfg)
        assert "id" in lf.collect_schema().names()
        assert lf.collect().height == 2

    def test_accepts_path_string(self):
        lf = read_file(str(CSV_SAMPLE))
        assert isinstance(lf, pl.LazyFrame)


class TestReadFileTsv:
    def test_returns_lazy_frame(self):
        lf = read_file(TSV_SAMPLE)
        assert isinstance(lf, pl.LazyFrame)

    def test_correct_columns(self):
        lf = read_file(TSV_SAMPLE)
        assert lf.collect_schema().names() == EXPECTED_COLUMNS

    def test_correct_row_count(self):
        lf = read_file(TSV_SAMPLE)
        assert lf.collect().height == EXPECTED_ROW_COUNT

    def test_tsv_with_default_separator_resolves_to_tab(self):
        # Reading .tsv without config should parse correctly
        df = read_file(TSV_SAMPLE).collect()
        assert df["name"][0] == "Alice"


class TestReadFileJson:
    def test_returns_lazy_frame(self):
        lf = read_file(JSON_SAMPLE)
        assert isinstance(lf, pl.LazyFrame)

    def test_correct_columns(self):
        lf = read_file(JSON_SAMPLE)
        assert set(lf.collect_schema().names()) == set(EXPECTED_COLUMNS)

    def test_correct_row_count(self):
        lf = read_file(JSON_SAMPLE)
        assert lf.collect().height == EXPECTED_ROW_COUNT

    def test_numeric_values_correct(self):
        df = read_file(JSON_SAMPLE).collect()
        amounts = sorted(df["amount"].to_list())
        assert amounts == pytest.approx([45.99, 75.25, 100.50, 250.00, 320.00])


class TestReadFileParquet:
    def test_returns_lazy_frame(self, parquet_file):
        lf = read_file(parquet_file)
        assert isinstance(lf, pl.LazyFrame)

    def test_correct_columns(self, parquet_file):
        lf = read_file(parquet_file)
        assert lf.collect_schema().names() == EXPECTED_COLUMNS

    def test_correct_row_count(self, parquet_file):
        lf = read_file(parquet_file)
        assert lf.collect().height == EXPECTED_ROW_COUNT


class TestReadFileExcel:
    def test_returns_lazy_frame(self, xlsx_file):
        lf = read_file(xlsx_file)
        assert isinstance(lf, pl.LazyFrame)

    def test_correct_columns(self, xlsx_file):
        lf = read_file(xlsx_file)
        assert set(lf.collect_schema().names()) == set(EXPECTED_COLUMNS)

    def test_correct_row_count(self, xlsx_file):
        lf = read_file(xlsx_file)
        assert lf.collect().height == EXPECTED_ROW_COUNT

    def test_with_sheet_name(self, xlsx_file):
        cfg = ReaderConfig(sheet_name="Sheet1")
        lf = read_file(xlsx_file, config=cfg)
        assert lf.collect().height == EXPECTED_ROW_COUNT


class TestUnsupportedExtension:
    def test_unknown_extension_raises_unsupported_file_error(self, tmp_path):
        f = tmp_path / "data.feather"
        f.write_text("dummy")
        with pytest.raises(UnsupportedFileError, match=r"\.feather"):
            read_file(f)

    def test_no_extension_raises_unsupported_file_error(self, tmp_path):
        f = tmp_path / "datafile"
        f.write_text("dummy")
        with pytest.raises(UnsupportedFileError):
            read_file(f)


class TestReadHeader:
    def test_csv_returns_column_names(self):
        headers = read_header(CSV_SAMPLE)
        assert headers == EXPECTED_COLUMNS

    def test_tsv_returns_column_names(self):
        headers = read_header(TSV_SAMPLE)
        assert headers == EXPECTED_COLUMNS

    def test_json_returns_column_names(self):
        headers = read_header(JSON_SAMPLE)
        assert set(headers) == set(EXPECTED_COLUMNS)

    def test_parquet_returns_column_names(self, parquet_file):
        headers = read_header(parquet_file)
        assert headers == EXPECTED_COLUMNS

    def test_unsupported_raises(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("x")
        with pytest.raises(UnsupportedFileError):
            read_header(f)
