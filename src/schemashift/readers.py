"""File readers that return Polars LazyFrames."""

import codecs
from pathlib import Path
from typing import Any

import polars as pl

from .errors import ReaderError, UnsupportedFileError
from .models import ReaderConfig

_EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls"})


def read_file(path: Path, config: ReaderConfig | None = None) -> pl.LazyFrame:
    """Read a file into a LazyFrame based on its extension.

    Supported extensions: .csv, .tsv, .xlsx, .xls, .parquet, .json

    Args:
        path: Path to the file.
        config: Optional reader configuration (skip_rows, separator, encoding, etc.).

    Returns:
        A Polars LazyFrame.

    Raises:
        UnsupportedFileError: For unrecognised file extensions.
        ReaderError: For any I/O or parsing failures.
    """
    path = Path(path)
    cfg = config or ReaderConfig()
    ext = path.suffix.lower()

    try:
        if ext == ".csv":
            return _read_csv(path, cfg, default_sep=",")
        if ext == ".tsv":
            return _read_csv(path, cfg, default_sep="\t")
        if ext in _EXCEL_EXTENSIONS:
            return _read_excel(path, cfg)
        if ext == ".parquet":
            return pl.scan_parquet(path)
        if ext == ".json":
            return pl.read_json(path).lazy()
        raise UnsupportedFileError(
            f"Unsupported file extension '{ext}' for file: {path}. Supported: .csv, .tsv, .xlsx, .xls, .parquet, .json"
        )
    except (UnsupportedFileError, ReaderError):
        raise
    except Exception as exc:
        raise ReaderError(f"Failed to read file '{path}': {exc}") from exc


def read_header(path: Path, config: ReaderConfig | None = None) -> list[str]:
    """Read only the column names from a file (useful for format detection).

    Args:
        path: Path to the file.
        config: Optional reader configuration.

    Returns:
        List of column name strings.

    Raises:
        UnsupportedFileError: For unrecognised file extensions.
        ReaderError: For any I/O or parsing failures.
    """
    lf = read_file(path, config)
    return lf.collect_schema().names()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalise_csv_encoding(encoding: str) -> str:
    """Normalise encoding strings for Polars scan_csv.

    Polars scan_csv only accepts 'utf8' or 'utf8-lossy'.  Resolve the
    canonical codec name first (so 'utf-8', 'UTF-8', 'utf_8' all normalise
    to 'utf-8'), then map the utf-8 family to the 'utf8' token Polars expects.
    """
    try:
        canonical = codecs.lookup(encoding).name  # e.g. 'utf-8', 'latin-1'
    except LookupError:
        return encoding  # let Polars surface its own error
    _POLARS_MAP: dict[str, str] = {"utf-8": "utf8", "utf-8-sig": "utf8"}
    return _POLARS_MAP.get(canonical, canonical)


def _read_csv(path: Path, cfg: ReaderConfig, default_sep: str) -> pl.LazyFrame:
    sep = cfg.separator if cfg.separator is not None else default_sep
    return pl.scan_csv(
        path,
        separator=sep,
        skip_rows=cfg.skip_rows,
        encoding=_normalise_csv_encoding(cfg.encoding),  # ty: ignore[invalid-argument-type]
    )


def _read_excel(path: Path, cfg: ReaderConfig) -> pl.LazyFrame:
    kwargs: dict[str, Any] = {}

    # sheet_name accepts a string name; integers are treated as 0-based sheet
    # indices and mapped to the 1-based sheet_id parameter that pl.read_excel
    # expects.
    if isinstance(cfg.sheet_name, int):
        kwargs["sheet_id"] = cfg.sheet_name + 1
    elif cfg.sheet_name is not None:
        kwargs["sheet_name"] = cfg.sheet_name

    # The calamine (fastexcel) engine receives skip_rows via read_options as
    # header_row, which is the 0-based row index of the header row.
    if cfg.skip_rows:
        kwargs["read_options"] = {"header_row": cfg.skip_rows}

    try:
        df = pl.read_excel(path, **kwargs)
    except Exception as exc:
        raise ReaderError(f"Failed to read Excel file '{path}': {exc}") from exc

    return df.lazy()
