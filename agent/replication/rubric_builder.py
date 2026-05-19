"""Rubric builder agent — generates a replication rubric tree by paper and repo exploration.

The agent works layer by layer: it expands the current frontier of nodes, submitting each
layer via submit_layer, which continues the loop rather than terminating it. The loop ends
when every node in the frontier is a leaf (no children to expand).
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
from agent.replication.types import PaperReading, RubricNode

logger = logging.getLogger(__name__)

RUBRIC_BUILDER_TOOL_NAMES = {"hf_papers", "github_read_file", "github_list_repos", "web_search"}

_SUBMIT_LAYER_TOOL = "submit_layer"
_MAX_ITERATIONS = 60
_CONTEXT_WARN = 170_000
_CONTEXT_MAX = 190_000

_SUMMARY_PROMPT = (
    "You are summarizing a conversation in which an AI agent is building a replication rubric "
    "for an ML paper. The agent's job is to decompose 'replicate <paper>' into a tree of "
    "verifiable sub-tasks, layer by layer. Each node in the tree has an ID and a description "
    "of what it verifies. Internal nodes group related sub-tasks; leaf nodes are atomic — "
    "specific enough that an LLM judge can read experiment outputs and say pass or fail. "
    "The agent expands the tree one layer at a time: it reads the paper and repo, then submits "
    "children for each node in the current 'frontier' (the set of nodes that exist but have not "
    "yet been expanded). When all frontier nodes are leaves, the rubric is complete.\n\n"
    "Summarize what has happened in this conversation so far. Include:\n"
    "- The paper being replicated: title, arxiv_id, github_url, and key reported metrics\n"
    "- Every rubric node created so far: its ID, description, and whether it is a leaf or internal\n"
    "- The current frontier: the IDs and descriptions of nodes that still need to be expanded\n"
    "- Key findings from reading the paper or repo that should inform how remaining nodes are expanded\n"
    "This summary will replace the full conversation history and must contain everything the "
    "agent needs to continue expanding the frontier and completing the rubric."
)


class RubricBuildError(Exception):
    """Raised when the rubric builder cannot complete the rubric tree.

    Carries the partial rubric tree and the unexpanded frontier so the caller
    can inspect what was built and potentially retry from where it stopped.
    """

    def __init__(self, message: str, partial_rubric: RubricNode, frontier: list[RubricNode]):
        super().__init__(message)
        self.partial_rubric = partial_rubric
        self.frontier = frontier


_SUBMIT_LAYER_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _SUBMIT_LAYER_TOOL,
        "description": (
            "Submit your expansion of the current frontier nodes. "
            "Every node in the current frontier must appear as a parent_id exactly once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expansions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "parent_id": {
                                "type": "string",
                                "description": "ID of the frontier node being expanded",
                            },
                            "children": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "string",
                                            "description": "Unique node ID using dot notation, e.g. env.deps",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "What this node verifies — be specific enough that an LLM judge can check it against experiment outputs",
                                        },
                                        "is_leaf": {
                                            "type": "boolean",
                                            "description": "True if this node is atomic and verifiable against experiment outputs",
                                        },
                                    },
                                    "required": ["id", "description", "is_leaf"],
                                },
                            },
                        },
                        "required": ["parent_id", "children"],
                    },
                }
            },
            "required": ["expansions"],
        },
    },
}

_SYSTEM_PROMPT = """\
You are a rubric builder for ML paper replication. Your job is to construct a tree of verifiable
sub-tasks that together constitute a complete, faithful replication of a paper.

After the replication runs, an LLM judge will walk each leaf node and decide pass/fail by reading
the experiment outputs (logs, metric files, eval output). Write leaf descriptions that give the
judge enough specificity to make that call.

## Workflow

Work layer by layer. At each step you expand the current frontier nodes into children, then call
submit_layer. The tool response tells you which nodes to expand next. Continue until all nodes
are leaves.

Layer 1 — Read the paper abstract and table of contents first. Decompose the root into 3-5
high-level categories. Typical categories: environment setup, methodology implementation,
evaluation, results match.

Layer 2+ — For each category, read the relevant paper sections and explore the repo to understand
what specifically needs to be verified. Decompose into sub-tasks.

## When to mark a node as a leaf

- The task is a single verifiable thing (a specific metric threshold, a specific file exists, a
  specific behavior in the output)
- The description is specific enough that a judge reading experiment logs can say pass or fail
- Further decomposition adds no value

## Rules for leaf descriptions

- Include concrete thresholds where the paper states them (e.g. "mAP >= 45.2 on COCO val2017")
- Reference specific output artifacts where relevant (e.g. "eval_results.json contains top-1 accuracy")
- Avoid vague descriptions like "results are good" — be precise about what counts as passing

## When to keep a node internal

- The task covers multiple distinct verifiable things
- You need more paper/repo context before you can write a precise leaf description
"""


def _initial_prompt(arxiv_id: str, github_url: str, reading: PaperReading) -> str:
    metrics_str = "\n".join(
        f"  - {m.name}: {m.value} on {m.dataset}" for m in reading.metrics
    )
    return (
        f"Build a replication rubric for this paper.\n\n"
        f"arxiv_id: {arxiv_id}\n"
        f"github_url: {github_url or '(not available)'}\n"
        f"title: {reading.title}\n"
        f"metrics:\n{metrics_str}\n\n"
        f"Current frontier (nodes to expand):\n"
        f"  - root: \"Replicate {reading.title}\"\n\n"
        f"Read the paper and repo as needed, then submit your expansion of the root node."
    )


def _format_frontier(frontier: list[RubricNode]) -> str:
    return "\n".join(f"  - {node.id}: {node.description}" for node in frontier)


def _apply_expansions(
    expansions: list[dict],
    nodes_by_id: dict[str, RubricNode],
) -> list[RubricNode]:
    """Attach submitted children to their parents. Returns new frontier (non-leaf children)."""
    new_frontier: list[RubricNode] = []
    for exp in expansions:
        parent_id = exp.get("parent_id", "")
        parent = nodes_by_id.get(parent_id)
        if parent is None:
            continue
        for child_data in exp.get("children", []):
            try:
                node = RubricNode(
                    id=child_data["id"],
                    description=child_data["description"],
                    parent_id=parent_id,
                )
                parent.children.append(node)
                nodes_by_id[node.id] = node
                if not child_data.get("is_leaf", False):
                    new_frontier.append(node)
            except (KeyError, TypeError):
                continue
    return new_frontier


def _pick_model(main_model: str) -> str:
    if main_model.startswith("anthropic/"):
        return "anthropic/claude-sonnet-4-6"
    if main_model.startswith("bedrock/") and "anthropic" in main_model:
        return "bedrock/us.anthropic.claude-sonnet-4-6"
    return main_model


async def _compact_messages(
    messages: list[Message],
    model: str,
    hf_token: str | None,
    session: Any,
) -> list[Message]:
    """Summarize the middle of the message history, keeping system + first user + recent tail."""
    _TAIL = 8
    head = messages[:2]
    tail = messages[-_TAIL:] if len(messages) > 2 + _TAIL else messages[2:]
    middle = messages[2: len(messages) - _TAIL] if len(messages) > 2 + _TAIL else []
    if not middle:
        return messages
    summary_text, _ = await summarize_messages(
        middle,
        model_name=model,
        hf_token=hf_token,
        prompt=_SUMMARY_PROMPT,
        session=session,
        kind="replication",
    )
    summary_msg = Message(role="user", content=f"[Context summary]\n{summary_text}")
    return head + [summary_msg] + tail


async def run_rubric_builder(
    arxiv_id: str,
    github_url: str,
    reading: PaperReading,
    session: Any,
) -> RubricNode:
    """Build a rubric tree by having an agent explore the paper and repo layer by layer.

    The agent expands nodes one layer at a time via submit_layer calls. The loop terminates
    when every frontier node is a leaf. Raises RubricBuildError if the agent fails.
    """
    agent_id = "rubric-builder"
    agent_label = f"rubric-builder: {arxiv_id}"

    root = RubricNode(id="root", description=f"Replicate {reading.title}")
    nodes_by_id: dict[str, RubricNode] = {"root": root}
    frontier: list[RubricNode] = [root]

    tool_specs = [
        spec
        for spec in session.tool_router.get_tool_specs_for_llm()
        if spec["function"]["name"] in RUBRIC_BUILDER_TOOL_NAMES
    ]
    tool_specs.append(_SUBMIT_LAYER_SPEC)

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
        if spec["function"]["name"] != _SUBMIT_LAYER_TOOL
    }

    async def _log(text: str) -> None:
        try:
            await session.send_event(
                Event(
                    event_type="tool_log",
                    data={"tool": "replication", "log": text, "agent_id": agent_id, "label": agent_label},
                )
            )
        except Exception:
            pass

    messages: list[Message] = [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(role="user", content=_initial_prompt(arxiv_id, github_url, reading)),
    ]

    _total_tokens = 0
    _compacted = False

    for _iteration in range(_MAX_ITERATIONS):
        doom_prompt = check_for_doom_loop(messages)
        if doom_prompt:
            logger.warning("Rubric builder repetition guard at iteration %d", _iteration)
            messages.append(Message(role="user", content=doom_prompt))

        if _total_tokens >= _CONTEXT_MAX:
            raise RubricBuildError("Context limit reached after compaction", root, frontier)

        if not _compacted and _total_tokens >= _CONTEXT_WARN:
            _compacted = True
            await _log("Compacting context to continue")
            messages = await _compact_messages(
                messages, model, getattr(session, "hf_token", None), session
            )

        try:
            _msgs, _tools = with_prompt_caching(messages, tool_specs, llm_params.get("model"))
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
                logger.debug("rubric builder telemetry failed: %s", _telem_err)
        except Exception as e:
            logger.error("Rubric builder LLM error: %s", e)
            raise RubricBuildError(f"LLM error: {e}", root, frontier)

        if response.usage:
            _total_tokens = response.usage.total_tokens

        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append(Message(role="assistant", content=msg.content))
            messages.append(
                Message(
                    role="user",
                    content=(
                        f"[SYSTEM: You must call {_SUBMIT_LAYER_TOOL} to submit your expansion "
                        f"of the current frontier nodes:\n{_format_frontier(frontier)}]"
                    ),
                )
            )
            continue

        messages.append(
            Message(role="assistant", content=msg.content, tool_calls=msg.tool_calls)
        )

        for tc in msg.tool_calls:
            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            tool_name = tc.function.name

            if tool_name == _SUBMIT_LAYER_TOOL:
                new_frontier = _apply_expansions(
                    tool_args.get("expansions", []), nodes_by_id
                )
                frontier = new_frontier

                if not frontier:
                    await _log("Rubric complete — all nodes are leaves.")
                    messages.append(
                        Message(
                            role="tool",
                            content="All nodes are leaves. Rubric complete.",
                            tool_call_id=tc.id,
                            name=_SUBMIT_LAYER_TOOL,
                        )
                    )
                    return root

                continuation = (
                    f"Layer recorded. Now expand these {len(frontier)} nodes:\n"
                    + _format_frontier(frontier)
                )
                await _log(f"Layer done — {len(frontier)} nodes remaining")
                messages.append(
                    Message(
                        role="tool",
                        content=continuation,
                        tool_call_id=tc.id,
                        name=_SUBMIT_LAYER_TOOL,
                    )
                )

            elif tool_name not in allowed_tool_names:
                messages.append(
                    Message(
                        role="tool",
                        content=f"Tool '{tool_name}' not available.",
                        tool_call_id=tc.id,
                        name=tool_name,
                    )
                )
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

    raise RubricBuildError("Iteration limit reached", root, frontier)
