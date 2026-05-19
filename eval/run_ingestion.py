"""Eval harness for the ingestion pipeline.

Usage:
    uv run eval/run_ingestion.py 2310.06825
    uv run eval/run_ingestion.py https://arxiv.org/abs/2310.06825
    uv run eval/run_ingestion.py 2310.06825 --model anthropic/claude-opus-4-7

Runs paper_reader → resource_checker → rubric_builder, logs every event to
terminal and writes organized output to eval/runs/<timestamp>-<arxiv_id>/.

Per-run files:
  input.json            — arxiv_id, model, timestamp
  events.jsonl          — every tool_log + llm_call event in order
  llm_calls.jsonl       — every LLM request+response (full messages, tools, response)
  paper_reading.json    — paper_reader output
  resource_report.json  — resource_checker output
  rubric.json           — rubric tree (partial if error)
  rubric_error.json     — error + unexpanded frontier (only on failure)
  output.json           — final PaperTask or {"error": ...}
"""

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import litellm
from dotenv import load_dotenv
from litellm.integrations.custom_logger import CustomLogger

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from agent.core.session import Event
from agent.core.tools import ToolRouter
from agent.replication.ingestion import _fetch_paper_metadata, normalize_arxiv_id
from agent.replication.paper_reader import run_paper_reader
from agent.replication.resource_checker import run_resource_checker
from agent.replication.rubric_builder import RubricBuildError, run_rubric_builder
from agent.replication.types import PaperTask


# ── serialization ─────────────────────────────────────────────────────────────


def _to_dict(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if hasattr(obj, "value"):  # enum
        return obj.value
    return obj


def _msg_to_dict(msg) -> dict:
    if isinstance(msg, dict):
        return msg
    result = {}
    for attr in ("role", "content", "name", "tool_call_id"):
        val = getattr(msg, attr, None)
        if val is not None:
            result[attr] = val
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": getattr(tc, "type", "function"),
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tcs
        ]
    return result


def _response_to_dict(response) -> dict:
    try:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return dict(response)
    except Exception:
        return {"raw": str(response)}


# ── litellm callback — captures every LLM call ───────────────────────────────


class _LLMCallLogger(CustomLogger):
    def __init__(self, calls_path: Path):
        super().__init__()
        self._path = calls_path

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "latency_ms": int((end_time - start_time).total_seconds() * 1000),
            "model": kwargs.get("model"),
            "messages": [_msg_to_dict(m) for m in (kwargs.get("messages") or [])],
            "tools": kwargs.get("tools"),
            "response": _response_to_dict(response_obj),
        }
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": kwargs.get("model"),
            "messages": [_msg_to_dict(m) for m in (kwargs.get("messages") or [])],
            "error": str(response_obj),
        }
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")


# ── session shim ─────────────────────────────────────────────────────────────


def _make_session(model: str, hf_token: str | None, tool_router: ToolRouter, events: list, out_dir: Path):
    events_path = out_dir / "events.jsonl"

    async def send_event(event: Event):
        ts = datetime.now(timezone.utc).isoformat()
        record = {"ts": ts, "type": event.event_type, **(event.data or {})}
        events.append(record)
        with events_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        _print_event(ts, event)

    config = SimpleNamespace(model_name=model, reasoning_effort=None)
    return SimpleNamespace(config=config, hf_token=hf_token, tool_router=tool_router, send_event=send_event)


# ── terminal printing ─────────────────────────────────────────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"


def _ts_prefix(ts: str) -> str:
    return f"{_DIM}{ts[11:19]}{_RESET}"


def _print_event(ts: str, event: Event):
    d = event.data or {}
    prefix = _ts_prefix(ts)
    if event.event_type == "tool_log":
        label = d.get("label", d.get("agent_id", ""))
        log = d.get("log", "")
        print(f"{prefix} {_CYAN}[{label}]{_RESET} {log}")
    elif event.event_type == "llm_call":
        model = d.get("model", "")
        tokens = d.get("total_tokens", 0)
        cost = d.get("cost_usd", 0.0)
        latency = d.get("latency_ms", 0)
        kind = d.get("kind", "")
        print(f"{prefix} {_YELLOW}[llm/{kind}]{_RESET} {model}  tokens={tokens}  cost=${cost:.4f}  latency={latency}ms")


def _print_stage(name: str, elapsed: float | None = None):
    if elapsed is None:
        print(f"\n{_GREEN}▶ {name}{_RESET}")
    else:
        print(f"{_GREEN}✓ {name}{_RESET} {_DIM}({elapsed:.1f}s){_RESET}")


def _print_error(name: str, err: Exception):
    print(f"{_RED}✗ {name}: {err}{_RESET}")


# ── main ──────────────────────────────────────────────────────────────────────


async def run(paper_input: str, model: str):
    hf_token = os.environ.get("HF_TOKEN")
    arxiv_id = normalize_arxiv_id(paper_input) or paper_input.strip()

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(__file__).parent / "runs" / f"{timestamp}-{arxiv_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []

    litellm.callbacks = [_LLMCallLogger(out_dir / "llm_calls.jsonl")]

    input_record = {"arxiv_id": arxiv_id, "paper_input": paper_input, "model": model, "timestamp": timestamp}
    (out_dir / "input.json").write_text(json.dumps(input_record, indent=2))

    print(f"\nIngestion run: {_CYAN}{arxiv_id}{_RESET}  model={model}")
    print(f"Output dir: {out_dir}\n")

    async with ToolRouter(mcp_servers={}, hf_token=hf_token) as tool_router:
        session = _make_session(model, hf_token, tool_router, events, out_dir)

        # ── stage 1: paper reader ─────────────────────────────────────────
        _print_stage("paper_reader")
        t0 = time.monotonic()
        reading = await run_paper_reader(paper_input, session)
        elapsed = time.monotonic() - t0

        if reading is None:
            _print_error("paper_reader", ValueError("returned None"))
            _write_output(out_dir, None, error="paper_reader failed")
            return

        _print_stage("paper_reader", elapsed)
        (out_dir / "paper_reading.json").write_text(json.dumps(_to_dict(reading), indent=2))

        # ── stage 2: resource checker ─────────────────────────────────────
        _print_stage("resource_checker")
        t0 = time.monotonic()
        report = await run_resource_checker(arxiv_id, reading.github_url, session)
        elapsed = time.monotonic() - t0

        if report is None:
            _print_error("resource_checker", ValueError("returned None"))
            _write_output(out_dir, None, error="resource_checker failed")
            return

        _print_stage("resource_checker", elapsed)
        (out_dir / "resource_report.json").write_text(json.dumps(_to_dict(report), indent=2))

        # ── stage 3: rubric builder ───────────────────────────────────────
        _print_stage("rubric_builder")
        t0 = time.monotonic()
        rubric = None
        try:
            rubric = await run_rubric_builder(arxiv_id, reading.github_url, reading, session)
            elapsed = time.monotonic() - t0
            _print_stage("rubric_builder", elapsed)
        except RubricBuildError as e:
            elapsed = time.monotonic() - t0
            _print_error("rubric_builder", e)
            rubric = e.partial_rubric
            (out_dir / "rubric_error.json").write_text(
                json.dumps({
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "frontier": [n.id for n in e.frontier],
                }, indent=2)
            )

        (out_dir / "rubric.json").write_text(json.dumps(_to_dict(rubric), indent=2))

        # ── final output ──────────────────────────────────────────────────
        if rubric is not None:
            metadata = await _fetch_paper_metadata(arxiv_id)
            task = PaperTask(
                arxiv_id=arxiv_id,
                title=reading.title,
                github_url=reading.github_url,
                github_stars=metadata.get("githubStars", 0),
                abstract=metadata.get("summary", ""),
                rubric=rubric,
                datasets=report.datasets,
                models=report.models,
                repo_ready=report.repo_ready,
                repo_notes=report.repo_notes,
            )
            _write_output(out_dir, task)
            _print_summary(task)
        else:
            _write_output(out_dir, None, error="rubric_builder failed completely")


def _write_output(out_dir: Path, task: PaperTask | None, error: str | None = None):
    if task is not None:
        (out_dir / "output.json").write_text(json.dumps(_to_dict(task), indent=2))
    else:
        (out_dir / "output.json").write_text(json.dumps({"error": error}, indent=2))


def _print_summary(task: PaperTask):
    print(f"\n{'─'*60}")
    print(f"Title:    {task.title}")
    print(f"arxiv_id: {task.arxiv_id}")
    print(f"github:   {task.github_url or '(none)'}")
    print(f"repo_ready: {task.repo_ready}  —  {task.repo_notes[:80] if task.repo_notes else ''}")

    if task.datasets:
        print("\nDatasets:")
        for d in task.datasets:
            print(f"  {d.status.value:10} {d.name}  {d.hf_id or ''}")
    if task.models:
        print("\nModels:")
        for m in task.models:
            print(f"  {m.status.value:10} {m.name}  {m.hf_id or ''}")

    leaves = task.rubric.all_leaves()
    print(f"\nRubric: {len(leaves)} leaf nodes")
    for leaf in leaves[:10]:
        print(f"  [{leaf.id}] {leaf.description[:90]}")
    if len(leaves) > 10:
        print(f"  ... and {len(leaves) - 10} more (see rubric.json)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ingestion pipeline on a paper")
    parser.add_argument("paper", help="arxiv ID, arxiv URL, or paper title")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-6", help="LLM model name")
    args = parser.parse_args()
    asyncio.run(run(args.paper, args.model))
