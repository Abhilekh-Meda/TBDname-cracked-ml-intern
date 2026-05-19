"""Tests for the rubric builder — pure helpers and the agent loop."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import ChatCompletionMessageToolCall

from litellm import Message

from agent.replication.rubric_builder import (
    RubricBuildError,
    _apply_expansions,
    _compact_messages,
    _format_frontier,
    run_rubric_builder,
)
from agent.replication.types import MetricResult, PaperReading, RubricNode


# ── helpers ──────────────────────────────────────────────────────────────


def _reading(**kwargs) -> PaperReading:
    defaults = dict(
        arxiv_id="2406.04692",
        title="Test Paper",
        github_url="https://github.com/org/repo",
        metrics=[MetricResult(name="mAP", value=42.0, dataset="COCO val")],
    )
    return PaperReading(**{**defaults, **kwargs})


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


# ── _apply_expansions ────────────────────────────────────────────────────


def test_apply_expansions_attaches_children_to_parent():
    root = RubricNode(id="root", description="root")
    nodes_by_id = {"root": root}

    new_frontier = _apply_expansions(
        [
            {
                "parent_id": "root",
                "children": [
                    {"id": "env", "description": "env works", "is_leaf": False},
                    {"id": "results", "description": "results match", "is_leaf": False},
                ],
            }
        ],
        nodes_by_id,
    )

    assert len(root.children) == 2
    assert {c.id for c in root.children} == {"env", "results"}
    assert len(new_frontier) == 2


def test_apply_expansions_leaf_excluded_from_frontier():
    root = RubricNode(id="root", description="root")
    nodes_by_id = {"root": root}

    new_frontier = _apply_expansions(
        [
            {
                "parent_id": "root",
                "children": [
                    {"id": "env", "description": "env", "is_leaf": False},
                    {"id": "leaf", "description": "a leaf", "is_leaf": True},
                ],
            }
        ],
        nodes_by_id,
    )

    assert len(new_frontier) == 1
    assert new_frontier[0].id == "env"


def test_apply_expansions_leaf_description_stored():
    root = RubricNode(id="root", description="root")
    nodes_by_id = {"root": root}

    _apply_expansions(
        [
            {
                "parent_id": "root",
                "children": [
                    {"id": "leaf", "description": "mAP >= 45.2 on COCO val2017", "is_leaf": True},
                ],
            }
        ],
        nodes_by_id,
    )

    assert nodes_by_id["leaf"].description == "mAP >= 45.2 on COCO val2017"


def test_apply_expansions_skips_unknown_parent():
    nodes_by_id: dict = {}
    new_frontier = _apply_expansions(
        [{"parent_id": "nonexistent", "children": [{"id": "x", "description": "x", "is_leaf": True}]}],
        nodes_by_id,
    )
    assert new_frontier == []


def test_apply_expansions_all_leaves_returns_empty_frontier():
    root = RubricNode(id="root", description="root")
    nodes_by_id = {"root": root}

    new_frontier = _apply_expansions(
        [
            {
                "parent_id": "root",
                "children": [
                    {"id": "a", "description": "task a", "is_leaf": True},
                    {"id": "b", "description": "task b", "is_leaf": True},
                ],
            }
        ],
        nodes_by_id,
    )

    assert new_frontier == []
    assert len(root.children) == 2


def test_apply_expansions_sets_parent_id_on_children():
    root = RubricNode(id="root", description="root")
    nodes_by_id = {"root": root}

    _apply_expansions(
        [{"parent_id": "root", "children": [{"id": "env", "description": "env", "is_leaf": False}]}],
        nodes_by_id,
    )

    assert nodes_by_id["env"].parent_id == "root"


def test_apply_expansions_multi_layer():
    root = RubricNode(id="root", description="root")
    env = RubricNode(id="env", description="env", parent_id="root")
    root.children.append(env)
    nodes_by_id = {"root": root, "env": env}

    new_frontier = _apply_expansions(
        [
            {
                "parent_id": "env",
                "children": [
                    {"id": "env.deps", "description": "deps install without errors", "is_leaf": True},
                ],
            }
        ],
        nodes_by_id,
    )

    assert new_frontier == []
    assert nodes_by_id["env.deps"].description == "deps install without errors"


# ── _format_frontier ─────────────────────────────────────────────────────


def test_format_frontier_lists_all_nodes():
    nodes = [
        RubricNode(id="env", description="Environment works"),
        RubricNode(id="results", description="Results match"),
    ]
    output = _format_frontier(nodes)
    assert "env" in output
    assert "Environment works" in output
    assert "results" in output
    assert "Results match" in output


def test_format_frontier_empty():
    assert _format_frontier([]) == ""


# ── run_rubric_builder agent loop ─────────────────────────────────────────


def _make_tool_call(call_id: str, name: str, args: dict) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function={"name": name, "arguments": json.dumps(args)},
    )


def _make_response(tool_calls=None, content="", total_tokens=100):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )
    choice = SimpleNamespace(
        message=msg,
        finish_reason="tool_calls" if tool_calls else "stop",
    )
    resp = SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )
    return resp


@pytest.mark.asyncio
async def test_run_rubric_builder_single_layer_all_leaves():
    """Agent submits root expansion with all leaves → returns immediately."""
    layer1 = _make_tool_call(
        "tc1",
        "submit_layer",
        {
            "expansions": [
                {
                    "parent_id": "root",
                    "children": [
                        {"id": "env", "description": "env installs without errors", "is_leaf": True},
                        {"id": "result", "description": "mAP >= 40.0 on COCO val", "is_leaf": True},
                    ],
                }
            ]
        },
    )

    session = _session()

    with patch("agent.replication.rubric_builder.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch("agent.replication.rubric_builder._resolve_llm_params", return_value={"model": "test"}), \
         patch("agent.replication.rubric_builder.with_prompt_caching", side_effect=lambda m, t, _: (m, t)), \
         patch("agent.replication.rubric_builder.check_for_doom_loop", return_value=None), \
         patch("agent.replication.rubric_builder.telemetry.record_llm_call", new_callable=AsyncMock):

        mock_llm.return_value = _make_response(tool_calls=[layer1])
        rubric = await run_rubric_builder("2406.04692", "https://github.com/org/repo", _reading(), session)

    assert rubric.id == "root"
    assert len(rubric.children) == 2
    assert {c.id for c in rubric.children} == {"env", "result"}


@pytest.mark.asyncio
async def test_run_rubric_builder_two_layers():
    """Agent submits layer 1 with internal nodes, then layer 2 with all leaves."""
    layer1 = _make_tool_call(
        "tc1",
        "submit_layer",
        {
            "expansions": [
                {
                    "parent_id": "root",
                    "children": [
                        {"id": "env", "description": "env", "is_leaf": False},
                    ],
                }
            ]
        },
    )
    layer2 = _make_tool_call(
        "tc2",
        "submit_layer",
        {
            "expansions": [
                {
                    "parent_id": "env",
                    "children": [
                        {"id": "env.deps", "description": "deps install without errors", "is_leaf": True},
                    ],
                }
            ]
        },
    )

    session = _session()
    responses = [_make_response(tool_calls=[layer1]), _make_response(tool_calls=[layer2])]

    with patch("agent.replication.rubric_builder.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch("agent.replication.rubric_builder._resolve_llm_params", return_value={"model": "test"}), \
         patch("agent.replication.rubric_builder.with_prompt_caching", side_effect=lambda m, t, _: (m, t)), \
         patch("agent.replication.rubric_builder.check_for_doom_loop", return_value=None), \
         patch("agent.replication.rubric_builder.telemetry.record_llm_call", new_callable=AsyncMock):

        mock_llm.side_effect = responses
        rubric = await run_rubric_builder("2406.04692", "https://github.com/org/repo", _reading(), session)

    env = next(c for c in rubric.children if c.id == "env")
    assert len(env.children) == 1
    assert env.children[0].description == "deps install without errors"


@pytest.mark.asyncio
async def test_run_rubric_builder_raises_on_llm_error():
    """If LLM errors immediately, RubricBuildError is raised carrying partial rubric and frontier."""
    session = _session()

    with patch("agent.replication.rubric_builder.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch("agent.replication.rubric_builder._resolve_llm_params", return_value={"model": "test"}), \
         patch("agent.replication.rubric_builder.with_prompt_caching", side_effect=lambda m, t, _: (m, t)), \
         patch("agent.replication.rubric_builder.check_for_doom_loop", return_value=None), \
         patch("agent.replication.rubric_builder.telemetry.record_llm_call", new_callable=AsyncMock):

        mock_llm.side_effect = Exception("LLM unavailable")
        with pytest.raises(RubricBuildError) as exc_info:
            await run_rubric_builder("2406.04692", "https://github.com/org/repo", _reading(), session)

    assert exc_info.value.partial_rubric.id == "root"
    assert isinstance(exc_info.value.frontier, list)


@pytest.mark.asyncio
async def test_run_rubric_builder_tool_call_before_submit():
    """Agent makes a tool call, gets result, then submits layer."""
    tool_call = _make_tool_call("tc0", "hf_papers", {"operation": "paper_details", "arxiv_id": "2406.04692"})
    submit = _make_tool_call(
        "tc1",
        "submit_layer",
        {
            "expansions": [
                {
                    "parent_id": "root",
                    "children": [
                        {"id": "leaf", "description": "mAP >= 40.0 on COCO val", "is_leaf": True},
                    ],
                }
            ]
        },
    )

    session = _session()
    responses = [_make_response(tool_calls=[tool_call]), _make_response(tool_calls=[submit])]

    with patch("agent.replication.rubric_builder.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch("agent.replication.rubric_builder._resolve_llm_params", return_value={"model": "test"}), \
         patch("agent.replication.rubric_builder.with_prompt_caching", side_effect=lambda m, t, _: (m, t)), \
         patch("agent.replication.rubric_builder.check_for_doom_loop", return_value=None), \
         patch("agent.replication.rubric_builder.telemetry.record_llm_call", new_callable=AsyncMock):

        mock_llm.side_effect = responses
        rubric = await run_rubric_builder("2406.04692", "https://github.com/org/repo", _reading(), session)

    assert len(rubric.children) == 1
    assert rubric.children[0].description == "mAP >= 40.0 on COCO val"


# ── _compact_messages ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_messages_summarizes_middle_and_keeps_head_tail():
    messages = (
        [Message(role="system", content="sys"), Message(role="user", content="first")]
        + [Message(role="user", content=f"middle {i}") for i in range(12)]
        + [Message(role="user", content=f"tail {i}") for i in range(8)]
    )

    with patch("agent.replication.rubric_builder.summarize_messages", new_callable=AsyncMock) as mock_s:
        mock_s.return_value = ("the summary", 50)
        result = await _compact_messages(messages, "test-model", None, SimpleNamespace())

    assert result[0].content == "sys"
    assert result[1].content == "first"
    assert "the summary" in result[2].content
    assert len(result) == 2 + 1 + 8  # head + summary + tail


@pytest.mark.asyncio
async def test_compact_messages_skips_when_nothing_to_summarize():
    """Fewer messages than head + tail → nothing in the middle → returns unchanged."""
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="first"),
        Message(role="user", content="recent"),
    ]

    with patch("agent.replication.rubric_builder.summarize_messages", new_callable=AsyncMock) as mock_s:
        result = await _compact_messages(messages, "test-model", None, SimpleNamespace())

    mock_s.assert_not_called()
    assert result == messages


# ── run_rubric_builder — error paths ─────────────────────────────────────


@pytest.mark.asyncio
async def test_run_rubric_builder_raises_on_iteration_limit():
    """Loop exhausts all 60 iterations without submit_layer → RubricBuildError."""
    session = _session()

    with patch("agent.replication.rubric_builder.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch("agent.replication.rubric_builder._resolve_llm_params", return_value={"model": "test"}), \
         patch("agent.replication.rubric_builder.with_prompt_caching", side_effect=lambda m, t, _: (m, t)), \
         patch("agent.replication.rubric_builder.check_for_doom_loop", return_value=None), \
         patch("agent.replication.rubric_builder.telemetry.record_llm_call", new_callable=AsyncMock):

        mock_llm.return_value = _make_response(content="still thinking")
        with pytest.raises(RubricBuildError) as exc_info:
            await run_rubric_builder("2406.04692", "https://github.com/org/repo", _reading(), session)

    assert "Iteration limit" in str(exc_info.value)
    assert exc_info.value.partial_rubric.id == "root"
    assert isinstance(exc_info.value.frontier, list)


@pytest.mark.asyncio
async def test_run_rubric_builder_raises_on_context_max():
    """Context hits 190k even after compaction → RubricBuildError."""
    session = _session()

    # Iteration 0: tokens hit 170k → compaction triggered, no tool calls → nudge loop continues
    resp1 = _make_response(content="thinking", total_tokens=171_000)
    # Iteration 1: tokens still at 190k → raises
    resp2 = _make_response(content="still thinking", total_tokens=191_000)

    with patch("agent.replication.rubric_builder.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch("agent.replication.rubric_builder._resolve_llm_params", return_value={"model": "test"}), \
         patch("agent.replication.rubric_builder.with_prompt_caching", side_effect=lambda m, t, _: (m, t)), \
         patch("agent.replication.rubric_builder.check_for_doom_loop", return_value=None), \
         patch("agent.replication.rubric_builder.telemetry.record_llm_call", new_callable=AsyncMock), \
         patch("agent.replication.rubric_builder.summarize_messages", new_callable=AsyncMock) as mock_s:

        mock_s.return_value = ("summary", 100)
        mock_llm.side_effect = [resp1, resp2]
        with pytest.raises(RubricBuildError) as exc_info:
            await run_rubric_builder("2406.04692", "https://github.com/org/repo", _reading(), session)

    assert "Context limit" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_rubric_builder_compacts_at_170k_and_continues():
    """Context hits 170k → _compact_messages called → loop continues to completion."""
    layer = _make_tool_call(
        "tc1",
        "submit_layer",
        {
            "expansions": [
                {
                    "parent_id": "root",
                    "children": [
                        {"id": "env", "description": "env works", "is_leaf": True},
                    ],
                }
            ]
        },
    )

    session = _session()
    resp1 = _make_response(content="reading paper", total_tokens=171_000)
    resp2 = _make_response(tool_calls=[layer])

    compact_calls = []

    async def fake_compact(messages, model, hf_token, session):
        compact_calls.append(True)
        return messages  # return unchanged so loop can continue

    with patch("agent.replication.rubric_builder.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch("agent.replication.rubric_builder._resolve_llm_params", return_value={"model": "test"}), \
         patch("agent.replication.rubric_builder.with_prompt_caching", side_effect=lambda m, t, _: (m, t)), \
         patch("agent.replication.rubric_builder.check_for_doom_loop", return_value=None), \
         patch("agent.replication.rubric_builder.telemetry.record_llm_call", new_callable=AsyncMock), \
         patch("agent.replication.rubric_builder._compact_messages", fake_compact):

        mock_llm.side_effect = [resp1, resp2]
        rubric = await run_rubric_builder("2406.04692", "https://github.com/org/repo", _reading(), session)

    assert compact_calls, "_compact_messages should have been called"
    assert rubric.id == "root"
    assert len(rubric.children) == 1
