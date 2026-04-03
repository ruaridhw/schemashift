"""LLM-assisted config generation for schemashift."""

import logging
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import ValidationError

from . import dsl as _dsl_module
from .errors import LLMGenerationError
from .models import FormatConfig
from .readers import read_file
from .target_schema import TargetSchema
from .transform import dry_run, validate_config

_log = logging.getLogger(__name__)

_DSL_REFERENCE = _dsl_module.__doc__ or ""


def load_default_llm() -> Any:
    """Load a LangChain LLM from environment variables.

    Resolution order:

    1. Azure AI Foundry — when ``FOUNDRY_API_KEY`` and ``FOUNDRY_RESOURCE`` are set.
       Uses the Anthropic messages API at
       ``https://{FOUNDRY_RESOURCE}.services.ai.azure.com/anthropic``.
       ``MODEL_NAME`` selects the deployment (defaults to ``'claude-haiku-4-5'``).
    2. Direct Anthropic — when ``ANTHROPIC_API_KEY`` is set.

    Raises:
        ImportError: When ``langchain-anthropic`` is not installed.
        ValueError: When no recognised API key is found in the environment.
    """
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv optional; env vars may already be set

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError("langchain-anthropic is not installed. Run: pip install 'schemashift[llm]'") from exc

    foundry_key = os.getenv("FOUNDRY_API_KEY")
    foundry_resource = os.getenv("FOUNDRY_RESOURCE")
    if foundry_key and foundry_resource:
        model_name = os.getenv("MODEL_NAME", "claude-haiku-4-5")
        return ChatAnthropic(
            model=model_name,
            api_key=foundry_key,
            base_url=f"https://{foundry_resource}.services.ai.azure.com/anthropic",
        )  # ty: ignore[missing-argument, unknown-argument]

    if os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001", temperature=0
        )  # ty: ignore[missing-argument, unknown-argument]

    raise ValueError(
        "No LLM API key found. Set FOUNDRY_API_KEY + FOUNDRY_RESOURCE (Azure AI Foundry) or ANTHROPIC_API_KEY."
    )


def build_prompt(
    sample_df: pl.DataFrame,
    target_schema: TargetSchema,
    file_columns: list[str],
    example_configs: list[FormatConfig] | None = None,
    format_name: str = "unknown_format",
    user_prompt: str | None = None,
) -> str:
    """Build a prompt requesting a FormatConfig for the given file.

    Args:
        sample_df: A small sample of the source data.
        target_schema: The desired output schema.
        file_columns: Column names present in the source file.
        example_configs: Optional list of existing FormatConfigs to show as examples.
        format_name: Suggested name for the new format config.
        user_prompt: Optional extra context appended to the prompt (e.g. unit
            conventions, timestamp formats).

    Returns:
        A prompt string ready to send to an LLM.
    """
    parts: list[str] = []

    parts.append(
        "You are a data engineering assistant. Call the submit_format_config tool with a "
        f"FormatConfig that maps columns from a source file named '{format_name}' to the "
        "target schema below."
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

    # Additional user-provided context
    if user_prompt:
        parts.append(f"\n## Additional Context\n{user_prompt}")

    return "\n".join(parts)


def generate_config(
    path: str,
    target_schema: TargetSchema,
    llm: Any,  # ANNOT: use stronger typing
    example_configs: list[FormatConfig] | None = None,
    format_name: str | None = None,
    max_retries: int = 2,
    n_sample_rows: int = 15,
    user_prompt: str | None = None,
) -> FormatConfig:
    """Generate a FormatConfig for the given file using the LangChain agent API.

    Creates an agent via :func:`langchain.agents.create_agent` with a single
    ``submit_format_config`` tool.  The agent calls the tool with a candidate
    config; validation errors (Pydantic, DSL, dry-run) are returned as tool
    result strings so the agent can self-correct up to *max_retries* times.

    Args:
        path: Path to the source data file.
        target_schema: The desired output schema.
        llm: A LangChain ``BaseChatModel`` instance.
        example_configs: Optional existing FormatConfigs to include as examples.
        format_name: Name for the generated config. Defaults to the file stem.
        max_retries: Number of additional attempts after the first failure.
        n_sample_rows: Number of rows to sample from the file for the prompt.
        user_prompt: Optional extra context for the LLM (e.g. unit conventions,
            timestamp formats).

    Returns:
        A validated :class:`~schemashift.models.FormatConfig`.

    Raises:
        LLMGenerationError: When the agent fails to produce a valid config.
            ``error.attempts`` contains per-attempt details.
    """
    try:
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
    except ImportError as exc:
        raise ImportError("langchain is not installed. Run: pip install 'schemashift[llm]'") from exc

    df: pl.DataFrame = read_file(path).head(n_sample_rows).collect()  # ty: ignore[invalid-assignment]
    inferred_name = format_name if format_name is not None else Path(path).stem
    prompt = build_prompt(df, target_schema, list(df.columns), example_configs, inferred_name, user_prompt=user_prompt)

    # Side-channels: the tool captures its result and all attempt records here.
    result_box: list[FormatConfig] = []
    all_attempts: list[dict[str, Any]] = []

    @tool
    def submit_format_config(
        columns: list[dict],
        name: str = "",
        description: str = "",
        drop_unmapped: bool = True,
    ) -> str:
        """Submit the generated FormatConfig mapping source columns to the target schema.

        Each column dict must contain 'target' and exactly one of: 'source' (direct
        column rename), 'expr' (DSL expression string), or 'constant' (literal value).
        Optional per-column keys: 'dtype' (cast type), 'fillna' (fill-null value).
        Returns 'Config accepted.' on success, or an error description to fix and retry.
        """
        data: dict[str, Any] = {
            "name": name or inferred_name,
            "description": description,
            "columns": columns,
            "drop_unmapped": drop_unmapped,
        }

        try:
            config = FormatConfig.model_validate(data)
        except (ValidationError, Exception) as exc:
            error = str(exc)
            all_attempts.append({"response": data, "error": error})
            _log.warning("FormatConfig validation attempt failed: %s", error)
            return f"Validation error: {error}"

        dsl_errors = validate_config(config)
        if dsl_errors:
            error = "\n".join(dsl_errors)
            all_attempts.append({"response": data, "error": error})
            _log.warning("FormatConfig validation attempt failed: %s", error)
            return f"DSL errors:\n{error}"

        try:
            dry_run(config, path, n_rows=5)
        except Exception as exc:
            error = str(exc)
            all_attempts.append({"response": data, "error": error})
            _log.warning("FormatConfig validation attempt failed: %s", error)
            return f"Runtime error during dry run: {error}"

        all_attempts.append({"response": data, "error": None})
        result_box.append(config)
        return "Config accepted."

    # Each agent step is one LLM call + one tool call = 2 graph nodes.
    # Add a small buffer for the final LLM "I'm done" step.
    recursion_limit = (max_retries + 1) * 3 + 2

    agent = create_agent(llm, [submit_format_config])
    try:
        agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as exc:
        if result_box:
            return result_box[0]
        if not all_attempts:
            all_attempts.append({"response": "", "error": str(exc)})
        raise LLMGenerationError(str(exc), attempts=all_attempts) from exc

    if result_box:
        return result_box[0]

    raise LLMGenerationError(
        "Agent completed without submitting a valid config",
        attempts=all_attempts,
    )
