"""Shared test fixtures and helpers."""

import os

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


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


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Minimal fake LLM that supports tool-calling for use with create_agent."""

    def bind_tools(self, tools, **kwargs):
        return self


def make_tool_calling_llm(*tool_call_args: dict | None) -> FakeToolCallingModel:
    """Build a fake LLM that makes tool calls in sequence then finishes."""
    responses: list[AIMessage] = []
    for index, args in enumerate(tool_call_args):
        if args is None:
            continue
        responses.append(
            AIMessage(
                content="",
                tool_calls=[{"id": f"tc{index}", "name": "submit_format_config", "args": args}],
            )
        )
    responses.append(AIMessage(content="Done."))
    return FakeToolCallingModel(responses=responses)
