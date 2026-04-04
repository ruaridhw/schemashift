"""Auto-detect which registered FormatConfig matches a file's columns."""

from schemashift.errors import AmbiguousFormatError
from schemashift.models import FormatConfig
from schemashift.registry import Registry


def detect_format(
    file_columns: list[str],
    registry: Registry,
    min_score: float = 0.4,
) -> FormatConfig | None:
    """Detect which registered config matches a file's columns.

    A config matches when the file's column set is a superset of the config's
    referenced source columns (i.e. every column the config needs is present).
    Matching configs are ranked by specificity score: ``len(required) / len(file_columns)``.

    Args:
        file_columns: Column names found in the file being inspected.
        registry: Registry to search for candidate configs.
        min_score: Minimum specificity score required for a match.

    Returns:
        The matching FormatConfig when exactly one config matches, or None when
        no configs match.

    Raises:
        AmbiguousFormatError: When two or more configs match.
    """
    column_set = set(file_columns)
    scored_matches: list[tuple[float, FormatConfig]] = []
    subset_matches: list[FormatConfig] = []

    for config in registry.list_configs():
        required = config.source_columns()
        # Configs whose every mapping uses `constant` (no source columns) are
        # intentionally excluded from detection — they match any file, so they
        # would always cause AmbiguousFormatError when combined with real configs.
        # Such configs must be applied explicitly via --config rather than
        # auto-detected from the registry.
        if not required or not required.issubset(column_set):
            continue
        subset_matches.append(config)
        score = len(required) / len(column_set)
        if score >= min_score:
            scored_matches.append((score, config))

    if not scored_matches:
        if len(subset_matches) > 1:
            candidate_names = [c.name for c in subset_matches]
            raise AmbiguousFormatError(
                f"Multiple low-specificity configs match the provided columns: {candidate_names}. "
                "Add more discriminating columns to distinguish them.",
                candidates=candidate_names,
            )
        return None
    best_score = max(score for score, _ in scored_matches)
    matches = [config for score, config in scored_matches if score == best_score]
    if len(matches) == 1:
        return matches[0]

    candidate_names = [c.name for c in matches]
    raise AmbiguousFormatError(
        f"Multiple configs match the provided columns: {candidate_names}. "
        "Add more discriminating columns to distinguish them.",
        candidates=candidate_names,
    )
