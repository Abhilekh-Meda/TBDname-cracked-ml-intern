"""Tests for the shared sub-agent loop — context compaction and nudge behaviour."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from litellm import Message
from litellm.types.utils import ChatCompletionMessageToolCall

from agent.replication._sub_agent import run_sub_agent


# ── helpers ──────────────────────────────────────────────────────────────


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/test", reasoning_effort=None),
        hf_token=None,
        tool_router=SimpleNamespace(
            get_tool_specs_for_llm=lambda: [],
            call_tool=AsyncMock(return_value=("ok", None)),
        ),
        send_event=AsyncMock(),
    )


def _submit_tool_call(call_id: str = "tc1") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function={"name": "submit_result", "arguments": json.dumps({"value": "done"})},
    )


def _make_response(tool_calls=None, content="", total_tokens=100):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(
        message=msg,
        finish_reason="tool_calls" if tool_calls else "stop",
    )
    return SimpleNamespace(choices=[choice], usage=SimpleNamespace(total_tokens=total_tokens))


def _base_messages(extra: int = 0) -> list[Message]:
    msgs = [
        Message(role="system", content="system prompt"),
        Message(role="user", content="first user message"),
    ]
    for i in range(extra):
        msgs.append(Message(role="user", content=f"history {i}"))
    return msgs


# ── context compaction ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sub_agent_compacts_when_summary_prompt_provided():
    """When context hits 170k and summary_prompt is given, summarize_messages is called."""
    session = _session()

    resp1 = _make_response(content="thinking", total_tokens=171_000)
    resp2 = _make_response(tool_calls=[_submit_tool_call()])

    compact_calls = []

    async def fake_summarize(messages, model_name, **kwargs):
        compact_calls.append(True)
        return ("a summary", 100)

    with patch("agent.replication._sub_agent.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch("agent.replication._sub_agent._resolve_llm_params", return_value={"model": "test"}), \
         patch("agent.replication._sub_agent.with_prompt_caching", side_effect=lambda m, t, _: (m, t)), \
         patch("agent.replication._sub_agent.check_for_doom_loop", return_value=None), \
         patch("agent.replication._sub_agent.telemetry.record_llm_call", new_callable=AsyncMock), \
         patch("agent.replication._sub_agent.summarize_messages", fake_summarize):

        mock_llm.side_effect = [resp1, resp2]
        result, ok = await run_sub_agent(
            messages=_base_messages(extra=12),  # enough history so middle is non-empty
            tool_specs=[],
            submit_tool_name="submit_result",
            session=session,
            agent_id="test",
            agent_label="test",
            summary_prompt="summarize this",
        )

    assert ok
    assert compact_calls, "summarize_messages should have been called"


@pytest.mark.asyncio
async def test_sub_agent_nudges_when_no_summary_prompt():
    """When context hits 170k and no summary_prompt, a nudge message is appended instead."""
    session = _session()

    resp1 = _make_response(content="thinking", total_tokens=171_000)
    resp2 = _make_response(tool_calls=[_submit_tool_call()])

    with patch("agent.replication._sub_agent.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch("agent.replication._sub_agent._resolve_llm_params", return_value={"model": "test"}), \
         patch("agent.replication._sub_agent.with_prompt_caching", side_effect=lambda m, t, _: (m, t)), \
         patch("agent.replication._sub_agent.check_for_doom_loop", return_value=None), \
         patch("agent.replication._sub_agent.telemetry.record_llm_call", new_callable=AsyncMock), \
         patch("agent.replication._sub_agent.summarize_messages", new_callable=AsyncMock) as mock_s:

        mock_llm.side_effect = [resp1, resp2]
        result, ok = await run_sub_agent(
            messages=_base_messages(),
            tool_specs=[],
            submit_tool_name="submit_result",
            session=session,
            agent_id="test",
            agent_label="test",
            summary_prompt=None,
        )

    assert ok
    mock_s.assert_not_called()
