"""Built-in lookup table registry. JSON files in tables/ are auto-discovered."""

import json
from pathlib import Path

_TABLES_DIR = Path(__file__).parent / "tables"


def _load_tables() -> dict[str, dict[str, str]]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(_TABLES_DIR.glob("*.json"))}


TABLES: dict[str, dict[str, str]] = _load_tables()
