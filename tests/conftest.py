"""Shared pytest configuration and fixtures."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.llm`` tests when no LLM API key is available."""
    if config.option.markexpr and "llm" in config.option.markexpr:
        # User explicitly requested llm tests — only skip if key is missing.
        has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("FOUNDRY_API_KEY"))
        if not has_key:
            skip = pytest.mark.skip(reason="No ANTHROPIC_API_KEY or FOUNDRY_API_KEY set")
            for item in items:
                if item.get_closest_marker("llm"):
                    item.add_marker(skip)
