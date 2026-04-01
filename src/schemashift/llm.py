"""LLM-assisted config generation for schemashift."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import ValidationError

from .errors import LLMGenerationError
from .models import FormatConfig
from .target_schema import TargetSchema
from .transform import dry_run, validate_config

_log = logging.getLogger(__name__)

_DSL_REFERENCE = """\
DSL Expression Reference:
  col("Column Name")                        # reference a column
  col("X") + col("Y")                       # arithmetic: +, -, *, /, %
  col("X") / 1000                           # divide by constant
  col("Name").str.strip()                   # strip whitespace
  col("Name").str.lower()                   # lowercase
  col("Name").str.to_uppercase()            # uppercase
  col("Name").str.slice(0, 3)              # first 3 chars
  col("Name").str.replace("old", "new")                  # replace substring (literal)
  col("Name").str.replace_regex("\\d+", "NUM")           # replace via regex
  coalesce(col("A"), col("B"), "fallback")               # first non-null value
  col("Name").str.contains("x")            # boolean contains
  col("Name").str.starts_with("x")         # boolean
  col("Name").str.ends_with("x")           # boolean
  col("Date").str.to_datetime("%Y-%m-%d")  # parse datetime
  col("Name").str.lengths()                # string length
  col("dt").dt.year()                      # extract year
  col("dt").dt.month()                     # extract month
  col("dt").dt.day()                       # extract day
  col("dt").dt.strftime("%Y-%m-%d")        # format datetime
  col("x").round(2)                        # round
  col("x").abs()                           # absolute value
  col("x").cast("float64")   # cast: str, int32, int64, float32, float64, bool, datetime, date
  col("x").fill_null(0)                    # fill nulls
  col("x").is_null()                       # boolean null check
  when(col("T") == "A", "Result A").otherwise("Other")                        # conditional
  when(col("T") == "A", "A").when(col("T") == "B", "B").otherwise("C")       # chained
"""

_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n\s*```", re.DOTALL)
_OUTERMOST_BRACES_RE = re.compile(r"\{.*\}", re.DOTALL)
_RETRY_SUFFIX = "\n\nPlease fix and return only valid JSON."


def _extract_json(text: str) -> str:
    """Extract a JSON object string from LLM response text.

    Tries three strategies in order:
    1. A ```json...``` fenced code block.
    2. The outermost ``{...}`` span in the text.
    3. Raises :class:`~schemashift.errors.LLMGenerationError`.

    Args:
        text: Raw text returned by the LLM.

    Returns:
        The extracted JSON string (not yet parsed).

    Raises:
        LLMGenerationError: When no JSON object can be located.
    """
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _OUTERMOST_BRACES_RE.search(text)
    if m:
        return m.group(0).strip()

    raise LLMGenerationError("No JSON found in LLM response")


def build_prompt(
    sample_df: pl.DataFrame,
    target_schema: TargetSchema,
    file_columns: list[str],
    example_configs: list[FormatConfig] | None = None,
    format_name: str = "unknown_format",
) -> str:
    """Build an LLM prompt requesting a FormatConfig JSON for a given file.

    Args:
        sample_df: A small sample of the source data.
        target_schema: The desired output schema.
        file_columns: Column names present in the source file.
        example_configs: Optional list of existing FormatConfigs to show as examples.
        format_name: Suggested name for the new format config.

    Returns:
        A prompt string ready to send to an LLM.
    """
    parts: list[str] = []

    parts.append(
        "You are a data engineering assistant. Generate a FormatConfig JSON that maps "
        f"columns from a source file named '{format_name}' to the target schema below."
    )

    # Target schema
    parts.append("\n## Target Schema")
    for col in target_schema.columns:
        required_label = "required" if col.required else "optional"
        desc = f" — {col.description}" if col.description else ""
        parts.append(f"  - {col.name} ({col.type}, {required_label}){desc}")

    # DSL reference
    parts.append(f"\n## {_DSL_REFERENCE}")

    # Sample data as markdown table
    parts.append("## Sample Data")
    headers = sample_df.columns
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    parts.append(header_row)
    parts.append(separator_row)
    for row in sample_df.rows():
        parts.append("| " + " | ".join(str(v) for v in row) + " |")

    # Source columns with dtypes
    parts.append("\n## Source File Columns")
    for col_name, dtype in sample_df.schema.items():
        parts.append(f"  - {col_name} ({dtype})")

    # Example configs
    if example_configs:
        parts.append(
            f"\n## Example Configs\n"
            f"The following are example configs that map to the same "
            f"'{target_schema.name}' target schema:"
        )
        for ex in example_configs:
            parts.append(ex.model_dump_json(indent=2))

    # Output instructions
    parts.append(
        "\n## Output Instructions\n"
        "Return ONLY a valid JSON object matching the FormatConfig schema. "
        "No explanation, no code blocks.\n\n"
        "FormatConfig schema:\n"
        "{\n"
        '  "name": "<format name>",\n'
        '  "description": "<optional description>",\n'
        '  "columns": [\n'
        "    {\n"
        '      "target": "<target column name>",\n'
        '      "source": "<source column name>",   // OR\n'
        '      "expr": "<DSL expression>",          // OR\n'
        '      "constant": <value>,                 // exactly one of source/expr/constant\n'
        '      "dtype": "<optional cast dtype>",\n'
        '      "fillna": <optional fill value>\n'
        "    }\n"
        "  ],\n"
        '  "drop_unmapped": true\n'
        "}"
    )

    return "\n".join(parts)


def generate_config(
    path: str,
    target_schema: TargetSchema,
    llm: Any,
    example_configs: list[FormatConfig] | None = None,
    format_name: str | None = None,
    max_retries: int = 2,
    sample_rows: int = 15,
) -> FormatConfig:
    """Generate a FormatConfig for the given file using an LLM.

    Args:
        path: Path to the source data file.
        target_schema: The desired output schema.
        llm: A LangChain BaseChatModel-compatible object (uses ``invoke``).
        example_configs: Optional existing FormatConfigs to include as examples.
        format_name: Name for the generated config. Defaults to the file stem.
        max_retries: Number of additional attempts after the first failure.
        sample_rows: Number of rows to sample from the file for the prompt.

    Returns:
        A validated :class:`~schemashift.models.FormatConfig`.

    Raises:
        LLMGenerationError: When a valid config cannot be generated within the allowed attempts.
    """
    from schemashift.readers import read_file

    df = read_file(path).head(sample_rows).collect()
    file_columns = list(df.columns)

    inferred_name = format_name if format_name is not None else Path(path).stem
    prompt = build_prompt(df, target_schema, file_columns, example_configs, inferred_name)

    last_error: str = ""
    total_attempts = max_retries + 1
    all_attempts: list[dict] = []

    for attempt_num in range(1, total_attempts + 1):
        response: str | None = None
        attempt_error: str | None = None

        # Call the LLM
        response = llm.invoke(prompt).content

        # 1. Extract JSON text
        try:
            json_str = _extract_json(response)
        except LLMGenerationError as exc:
            last_error = str(exc)
            attempt_error = last_error
            all_attempts.append({"prompt": prompt, "response": response, "error": attempt_error})
            _log.warning(
                "LLM generation attempt %d/%d failed: %s", attempt_num, total_attempts, last_error
            )
            prompt += f"\n\nThe previous response had errors:\n{last_error}" + _RETRY_SUFFIX
            continue

        # 2. Parse JSON
        try:
            data: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            attempt_error = last_error
            all_attempts.append({"prompt": prompt, "response": response, "error": attempt_error})
            _log.warning(
                "LLM generation attempt %d/%d failed: %s", attempt_num, total_attempts, last_error
            )
            prompt += f"\n\nThe previous response had errors:\n{last_error}" + _RETRY_SUFFIX
            continue

        # 3. Inject name if missing
        if not data.get("name"):
            data["name"] = inferred_name

        # 4. Validate with Pydantic
        try:
            config = FormatConfig.model_validate(data)
        except (ValidationError, Exception) as exc:
            last_error = str(exc)
            attempt_error = last_error
            all_attempts.append({"prompt": prompt, "response": response, "error": attempt_error})
            _log.warning(
                "LLM generation attempt %d/%d failed: %s", attempt_num, total_attempts, last_error
            )
            prompt += f"\n\nThe previous response had errors:\n{last_error}" + _RETRY_SUFFIX
            continue

        # 5. Validate DSL expressions and dtypes
        errors = validate_config(config)
        if errors:
            last_error = "\n".join(errors)
            attempt_error = last_error
            all_attempts.append({"prompt": prompt, "response": response, "error": attempt_error})
            _log.warning(
                "LLM generation attempt %d/%d failed: %s", attempt_num, total_attempts, last_error
            )
            prompt += f"\n\nThe previous response had errors:\n{last_error}" + _RETRY_SUFFIX
            continue

        # 6. Dry run against the actual file
        try:
            dry_run(config, path, n_rows=5)
        except Exception as exc:
            last_error = str(exc)
            attempt_error = last_error
            all_attempts.append({"prompt": prompt, "response": response, "error": attempt_error})
            _log.warning(
                "LLM generation attempt %d/%d failed: %s", attempt_num, total_attempts, last_error
            )
            prompt += f"\n\nThe previous response had errors:\n{last_error}" + _RETRY_SUFFIX
            continue

        all_attempts.append({"prompt": prompt, "response": response, "error": None})
        return config

    raise LLMGenerationError(
        f"Failed to generate valid config after {total_attempts} attempts: {last_error}",
        attempts=all_attempts,
    )
