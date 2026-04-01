"""Auto-detect which registered FormatConfig matches a file's columns."""

from __future__ import annotations

from schemashift.errors import AmbiguousFormatError
from schemashift.models import FormatConfig
from schemashift.registry import Registry


def detect_format(
    file_columns: list[str],
    registry: Registry,
) -> FormatConfig | None:
    """Detect which registered config matches a file's columns.

    A config matches when the file's column set is a superset of the config's
    referenced source columns (i.e. every column the config needs is present).

    Args:
        file_columns: Column names found in the file being inspected.
        registry: Registry to search for candidate configs.

    Returns:
        The matching FormatConfig when exactly one config matches, or None when
        no configs match.

    Raises:
        AmbiguousFormatError: When two or more configs match.
    """
    column_set = set(file_columns)
    matches: list[FormatConfig] = []

    for config in registry.list_configs():
        required = config.source_columns()
        if required and required.issubset(column_set):
            matches.append(config)

    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]

    candidate_names = [c.name for c in matches]
    raise AmbiguousFormatError(
        f"Multiple configs match the provided columns: {candidate_names}. "
        "Add more discriminating columns to distinguish them.",
        candidates=candidate_names,
    )
