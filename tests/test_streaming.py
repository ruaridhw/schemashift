"""Tests for streaming and lazy evaluation on large files."""

import polars as pl
import pytest

from schemashift.models import ColumnMapping, FormatConfig
from schemashift.transform import transform


class TestEagerEvaluation:
    """Verify transforms return DataFrames."""

    def test_transform_returns_dataframe(self, tmp_path):
        csv = tmp_path / "data.csv"
        pl.DataFrame({"x": [1, 2, 3]}).write_csv(str(csv))
        config = FormatConfig(name="t", columns=[ColumnMapping(target="y", source="x")])
        result = transform(str(csv), config)
        assert isinstance(result, pl.DataFrame)

    def test_filter_after_transform(self, tmp_path):
        csv = tmp_path / "data.csv"
        pl.DataFrame({"val": range(100)}).write_csv(str(csv))
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="value", source="val"),
            ],
        )
        result = transform(str(csv), config).filter(pl.col("value") > 50)
        assert len(result) == 49  # 51..99 = 49 rows


class TestLargeCSV:
    """Tests on realistically large CSV files (100k+ rows)."""

    @pytest.fixture
    def large_csv(self, tmp_path):
        n = 100_000
        df = pl.DataFrame(
            {
                "id": range(n),
                "amount_cents": [i * 100 for i in range(n)],
                "category": ["A" if i % 2 == 0 else "B" for i in range(n)],
                "name": [f"item_{i}" for i in range(n)],
            }
        )
        path = tmp_path / "large.csv"
        df.write_csv(str(path))
        return str(path)

    def test_large_csv_source_mapping(self, large_csv):
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="item_id", source="id"),
                ColumnMapping(target="amount_dollars", expr='col("amount_cents") / 100'),
            ],
        )
        result = transform(large_csv, config)
        assert len(result) == 100_000
        assert set(result.columns) == {"item_id", "amount_dollars"}
        assert result["amount_dollars"][0] == pytest.approx(0.0)
        assert result["amount_dollars"][1] == pytest.approx(1.0)

    def test_large_csv_filter_after_transform(self, large_csv):
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="item_id", source="id"),
                ColumnMapping(target="cat", source="category"),
            ],
        )
        result = transform(large_csv, config).filter(pl.col("cat") == "A")
        assert len(result) == 50_000

    def test_large_csv_n_rows_limits_output(self, large_csv):
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="item_id", source="id"),
            ],
        )
        result = transform(large_csv, config, n_rows=10)
        assert len(result) == 10

    def test_large_csv_with_constant_column(self, large_csv):
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="item_id", source="id"),
                ColumnMapping(target="source", constant="big_file_test"),
            ],
        )
        result = transform(large_csv, config)
        assert result["source"].n_unique() == 1
        assert result["source"][0] == "big_file_test"


class TestLargeTSV:
    """Same as CSV tests but with TSV."""

    @pytest.fixture
    def large_tsv(self, tmp_path):
        n = 10_000
        df = pl.DataFrame(
            {
                "id": range(n),
                "value": [float(i) for i in range(n)],
            }
        )
        path = tmp_path / "data.tsv"
        df.write_csv(str(path), separator="\t")
        return str(path)

    def test_large_tsv_transform(self, large_tsv):
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="row_id", source="id"),
                ColumnMapping(target="val", source="value"),
            ],
        )
        result = transform(large_tsv, config)
        assert len(result) == 10_000
        assert set(result.columns) == {"row_id", "val"}


class TestLargeParquet:
    """Parquet streaming via scan_parquet."""

    @pytest.fixture
    def large_parquet(self, tmp_path):
        n = 100_000
        df = pl.DataFrame(
            {
                "a": range(n),
                "b": [float(i) * 1.5 for i in range(n)],
            }
        )
        path = tmp_path / "data.parquet"
        df.write_parquet(str(path))
        return str(path)

    def test_parquet_transform(self, large_parquet):
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="alpha", source="a"),
                ColumnMapping(target="beta", source="b"),
            ],
        )
        result = transform(large_parquet, config)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 100_000
        assert result["beta"][1] == pytest.approx(1.5)

    def test_parquet_write_csv(self, large_parquet, tmp_path):
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="alpha", source="a"),
            ],
        )
        out_path = tmp_path / "out.csv"
        transform(large_parquet, config).write_csv(str(out_path))
        result = pl.read_csv(str(out_path))
        assert len(result) == 100_000
        assert "alpha" in result.columns

    def test_parquet_write_parquet(self, large_parquet, tmp_path):
        config = FormatConfig(
            name="t",
            columns=[
                ColumnMapping(target="alpha", source="a"),
                ColumnMapping(target="beta", source="b"),
            ],
        )
        out_path = tmp_path / "out.parquet"
        transform(large_parquet, config).write_parquet(str(out_path))
        result = pl.read_parquet(str(out_path))
        assert len(result) == 100_000
