"""Shared agentic loop for replication sub-agents.

Follows the same pattern as agent/tools/research_tool.py — doom loop detection,
prompt caching, context budget — but returns structured output via a submit tool
instead of prose.
"""

import json
import logging
import time
from typing import Any

from litellm import Message, acompletion

from agent.context_manager.manager import summarize_messages
from agent.core import telemetry
from agent.core.doom_loop import check_for_doom_loop
from agent.core.llm_params import _resolve_llm_params
from agent.core.prompt_caching import with_prompt_caching
from agent.core.session import Event

logger = logging.getLogger(__name__)

_CONTEXT_WARN = 170_000
_CONTEXT_MAX = 190_000
_MAX_ITERATIONS = 60


def _pick_model(main_model: str) -> str:
    if main_model.startswith("anthropic/"):
        return "anthropic/claude-sonnet-4-6"
    if main_model.startswith("bedrock/") and "anthropic" in main_model:
        return "bedrock/us.anthropic.claude-sonnet-4-6"
    return main_model


async def run_sub_agent(
    *,
    messages: list[Message],
    tool_specs: list[dict[str, Any]],
    submit_tool_name: str,
    session: Any,
    agent_id: str,
    agent_label: str,
    summary_prompt: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Run an agentic loop until the agent calls submit_tool_name.

    Returns (submit_args, True) on success, (None, False) on failure.
    """
    model = _pick_model(session.config.model_name)
    _pref = getattr(session.config, "reasoning_effort", None)
    _capped = "high" if _pref in ("max", "xhigh") else _pref
    llm_params = _resolve_llm_params(
        model,
        getattr(session, "hf_token", None),
        reasoning_effort=_capped,
    )

    allowed_tool_names = {
        spec["function"]["name"]
        for spec in tool_specs
        if spec["function"]["name"] != submit_tool_name
    }

    async def _log(text: str) -> None:
        try:
            await session.send_event(
                Event(
                    event_type="tool_log",
                    data={
                        "tool": "replication",
                        "log": text,
                        "agent_id": agent_id,
                        "label": agent_label,
                    },
                )
            )
        except Exception:
            pass

    _total_tokens = 0
    _compacted = False

    for _iteration in range(_MAX_ITERATIONS):
        doom_prompt = check_for_doom_loop(messages)
        if doom_prompt:
            logger.warning(
                "Replication sub-agent repetition guard activated at iteration %d",
                _iteration,
            )
            messages.append(Message(role="user", content=doom_prompt))

        if _total_tokens >= _CONTEXT_MAX:
            await _log("Context limit reached — aborting")
            return None, False

        if not _compacted and _total_tokens >= _CONTEXT_WARN:
            _compacted = True
            if summary_prompt is not None:
                _TAIL = 8
                head = messages[:2]
                tail = messages[-_TAIL:] if len(messages) > 2 + _TAIL else messages[2:]
                middle = messages[2: len(messages) - _TAIL] if len(messages) > 2 + _TAIL else []
                if not middle:
                    await _log("Context high but history too short to compact — aborting")
                    return None, False
                summary_text, _ = await summarize_messages(
                    middle,
                    model_name=model,
                    hf_token=getattr(session, "hf_token", None),
                    prompt=summary_prompt,
                    session=session,
                    kind="replication",
                )
                await _log("Compacted context to continue")
                messages = head + [Message(role="user", content=f"[Context summary]\n{summary_text}")] + tail
            else:
                messages.append(
                    Message(
                        role="user",
                        content=(
                            f"[SYSTEM: 75% of context used. Wrap up and call "
                            f"{submit_tool_name} with your findings now.]"
                        ),
                    )
                )

        try:
            _msgs, _tools = with_prompt_caching(
                messages, tool_specs, llm_params.get("model")
            )
            _t0 = time.monotonic()
            response = await acompletion(
                messages=_msgs,
                tools=_tools,
                tool_choice="auto",
                stream=False,
                timeout=120,
                **llm_params,
            )
            try:
                await telemetry.record_llm_call(
                    session,
                    model=model,
                    response=response,
                    latency_ms=int((time.monotonic() - _t0) * 1000),
                    finish_reason=(
                        response.choices[0].finish_reason if response.choices else None
                    ),
                    kind="replication",
                )
            except Exception as _telem_err:
                logger.debug("replication telemetry failed: %s", _telem_err)
        except Exception as e:
            logger.error("Replication sub-agent LLM error: %s", e)
            return None, False

        if response.usage:
            _total_tokens = response.usage.total_tokens

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            # Agent responded with text — nudge it to call the submit tool
            messages.append(Message(role="assistant", content=msg.content))
            messages.append(
                Message(
                    role="user",
                    content=f"[SYSTEM: You must call {submit_tool_name} to submit your findings.]",
                )
            )
            continue

        messages.append(
            Message(
                role="assistant",
                content=msg.content,
                tool_calls=msg.tool_calls,
            )
        )

        for tc in msg.tool_calls:
            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            tool_name = tc.function.name

            if tool_name == submit_tool_name:
                await _log("Submission received.")
                return tool_args, True

            if tool_name not in allowed_tool_names:
                result = f"Tool '{tool_name}' not available."
            else:
                try:
                    await _log(f"▸ {tool_name}  {json.dumps(tool_args)[:80]}")
                    result, _ = await session.tool_router.call_tool(
                        tool_name, tool_args, session=session, tool_call_id=tc.id
                    )
                    if len(result) > 8000:
                        result = result[:4800] + "\n...(truncated)...\n" + result[-3200:]
                except Exception as e:
                    result = f"Tool error: {e}"

            messages.append(
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                    name=tool_name,
                )
            )

    await _log("Iteration limit reached without submission.")
    return None, False
