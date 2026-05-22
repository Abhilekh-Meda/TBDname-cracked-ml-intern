"""Trajectory eval harness for the resource checker.

Usage:
    uv run eval/run_resource_checker.py 2405.14734
    uv run eval/run_resource_checker.py 2405.14734 --model openai/gpt-4.1

Fetches paper context then runs resource_checker. Writes to
eval/runs/<timestamp>-<arxiv_id>-rc/ and generates view.html.

Per-run files:
  input.json            — arxiv_id, model, timestamp
  paper_context.json    — output of fetch_paper_context
  events.jsonl          — every tool_log + llm_call event in order
  llm_calls.jsonl       — every LLM request+response
  resource_report.json  — resource checker output or {"error": ...}
  view.html             — self-contained viewer
"""

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
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
from agent.replication.ingestion import fetch_paper_context, normalize_arxiv_id
from agent.replication.resource_checker import run_resource_checker
from agent.replication.types import PaperContext

# ── serialization ──────────────────────────────────────────────────────────────


def _to_dict(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if hasattr(obj, "value"):
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


# ── litellm callback ───────────────────────────────────────────────────────────


class _LLMCallLogger(CustomLogger):
    def __init__(self, path: Path):
        super().__init__()
        self._path = path

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


# ── session shim ───────────────────────────────────────────────────────────────


def _make_session(model: str, hf_token: str | None, tool_router, events: list, out_dir: Path):
    events_path = out_dir / "events.jsonl"

    async def send_event(event: Event):
        ts = datetime.now(timezone.utc).isoformat()
        record = {"ts": ts, "type": event.event_type, **(event.data or {})}
        events.append(record)
        with (out_dir / "agent_events.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        _print_event(ts, event)

    config = SimpleNamespace(model_name=model, reasoning_effort=None)
    return SimpleNamespace(config=config, hf_token=hf_token, tool_router=tool_router, send_event=send_event)


# ── terminal printing ──────────────────────────────────────────────────────────

_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _print_event(ts: str, event: Event):
    d = event.data or {}
    prefix = f"{_DIM}{ts[11:19]}{_RESET}"
    if event.event_type == "tool_log":
        print(f"{prefix} {_CYAN}[{d.get('label', '')}]{_RESET} {d.get('log', '')}")
    elif event.event_type == "llm_call":
        tokens = d.get("total_tokens", 0)
        cost = d.get("cost_usd", 0.0)
        latency = d.get("latency_ms", 0)
        print(f"{prefix} {_YELLOW}[llm]{_RESET} tokens={tokens}  cost=${cost:.4f}  latency={latency}ms")


def _print_stage(name: str, elapsed: float | None = None):
    if elapsed is None:
        print(f"\n{_GREEN}▶ {name}{_RESET}")
    else:
        print(f"{_GREEN}✓ {name}{_RESET} {_DIM}({elapsed:.1f}s){_RESET}")


def _print_error(msg: str):
    print(f"{_RED}✗ {msg}{_RESET}")


# ── main ───────────────────────────────────────────────────────────────────────


async def run(paper_input: str, model: str):
    hf_token = os.environ.get("HF_TOKEN")
    arxiv_id = normalize_arxiv_id(paper_input) or paper_input.strip()

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(__file__).parent / "runs" / f"{timestamp}-{arxiv_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []
    litellm.callbacks = [_LLMCallLogger(out_dir / "llm_call_trace.jsonl")]

    (out_dir / "run_config.json").write_text(json.dumps(
        {"arxiv_id": arxiv_id, "model": model, "timestamp": timestamp}, indent=2
    ))

    print(f"\nResource checker run: {_CYAN}{arxiv_id}{_RESET}  model={model}")
    print(f"Output dir: {out_dir}\n")

    # ── fetch paper context (no LLM) ──────────────────────────────────────
    _print_stage("fetch_paper_context")
    t0 = time.monotonic()
    try:
        ctx = await fetch_paper_context(arxiv_id)
    except ValueError as e:
        _print_error(str(e))
        return
    elapsed = time.monotonic() - t0
    _print_stage("fetch_paper_context", elapsed)

    (out_dir / "fetched_paper_context.json").write_text(json.dumps(_to_dict(ctx), indent=2))
    print(f"  title:   {ctx.title}")
    print(f"  github:  {ctx.github_url or '(none)'}")
    print(f"  authors: {ctx.authors[:80]}")
    print(f"  text:    {len(ctx.full_text):,} chars")

    # ── resource checker ──────────────────────────────────────────────────
    async with ToolRouter(mcp_servers={}, hf_token=hf_token) as tool_router:
        session = _make_session(model, hf_token, tool_router, events, out_dir)

        _print_stage("resource_checker")
        t0 = time.monotonic()
        report = await run_resource_checker(ctx, session)
        elapsed = time.monotonic() - t0

        if report is None:
            _print_error("resource_checker returned None")
            (out_dir / "resource_report.json").write_text(json.dumps({"error": "agent failed"}, indent=2))
        else:
            _print_stage("resource_checker", elapsed)
            (out_dir / "resource_report.json").write_text(json.dumps(_to_dict(report), indent=2))
            _print_report(report)

    _build_viewer(out_dir)
    print(f"\nView: {out_dir / 'view.html'}")


def _print_report(report):
    from agent.replication.types import ResourceReport
    print(f"\n{'─'*60}")
    print(f"github_url:  {report.github_url}")
    print(f"  evidence:  {report.github_url_evidence[:100]}")
    print(f"repo_runnable: {report.repo_runnable}")
    print(f"  evidence:  {report.repo_runnable_evidence[:100]}")
    if report.datasets:
        print(f"\nDatasets ({len(report.datasets)}):")
        for d in report.datasets:
            print(f"  [{d.status.value:9}] {d.name}  {d.hf_id or ''}")
            print(f"             {d.source_evidence[:90]}")
    if report.models:
        print(f"\nModels ({len(report.models)}):")
        for m in report.models:
            print(f"  [{m.status.value:9}] {m.name}  {m.hf_id or ''}")
            print(f"             {m.source_evidence[:90]}")


# ── HTML viewer ────────────────────────────────────────────────────────────────


def _build_viewer(out_dir: Path):
    report_raw = (out_dir / "resource_report.json").read_text()
    ctx_raw = (out_dir / "paper_context.json").read_text()

    llm_calls = []
    lc_path = out_dir / "llm_calls.jsonl"
    if lc_path.exists():
        for line in lc_path.read_text().splitlines():
            if line.strip():
                llm_calls.append(json.loads(line))

    events = []
    ev_path = out_dir / "events.jsonl"
    if ev_path.exists():
        for line in ev_path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))

    report = json.loads(report_raw)
    ctx = json.loads(ctx_raw)

    html = _render_html(ctx, report, llm_calls, events)
    (out_dir / "view.html").write_text(html)


def _render_html(ctx: dict, report: dict, llm_calls: list, events: list) -> str:
    def je(v):
        return json.dumps(v, indent=2, ensure_ascii=False)

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # ── report section ────────────────────────────────────────────────────
    if "error" in report:
        report_html = f'<div class="error">Agent failed: {esc(report["error"])}</div>'
    else:
        def status_badge(s):
            colors = {"available": "#2d7a2d", "gated": "#a07020", "missing": "#8b2020", "unknown": "#555"}
            c = colors.get(s, "#555")
            return f'<span class="badge" style="background:{c}">{esc(s)}</span>'

        def resource_rows(items):
            if not items:
                return "<p class='dim'>None found.</p>"
            rows = []
            for it in items:
                rows.append(
                    f'<div class="resource-row">'
                    f'{status_badge(it.get("status","unknown"))}'
                    f'<strong>{esc(it["name"])}</strong>'
                    + (f' <span class="dim">{esc(it["hf_id"])}</span>' if it.get("hf_id") else "")
                    + f'<div class="evidence">"{esc(it.get("source_evidence") or it.get("evidence",""))}"</div>'
                    f'</div>'
                )
            return "\n".join(rows)

        runnable_color = "#2d7a2d" if report.get("repo_runnable") else "#8b2020"
        report_html = f"""
<div class="card">
  <h3>GitHub Repository</h3>
  <div class="url-row"><a href="{esc(report.get('github_url',''))}" target="_blank">{esc(report.get('github_url','(none)'))}</a></div>
  <div class="evidence">"{esc(report.get('github_url_evidence',''))}"</div>
</div>
<div class="card">
  <h3>Repo Runnable <span class="badge" style="background:{runnable_color}">{"yes" if report.get("repo_runnable") else "no"}</span></h3>
  <div class="evidence">"{esc(report.get('repo_runnable_evidence',''))}"</div>
</div>
<div class="card">
  <h3>Datasets ({len(report.get('datasets',[]))})</h3>
  {resource_rows(report.get('datasets',[]))}
</div>
<div class="card">
  <h3>Models ({len(report.get('models',[]))})</h3>
  {resource_rows(report.get('models',[]))}
</div>
"""

    # ── llm calls section ─────────────────────────────────────────────────
    def render_message(msg):
        role = msg.get("role", "")
        content = msg.get("content") or ""
        tcs = msg.get("tool_calls", [])
        role_colors = {"system": "#444", "user": "#1a4a7a", "assistant": "#2d5a2d", "tool": "#5a3a6a"}
        color = role_colors.get(role, "#333")
        parts = [f'<div class="msg" style="border-left:3px solid {color}">']
        parts.append(f'<div class="msg-role" style="color:{color}">{esc(role)}</div>')
        if content:
            if len(content) > 2000:
                parts.append(f'<details><summary class="dim">({len(content):,} chars)</summary><pre>{esc(content)}</pre></details>')
            else:
                parts.append(f'<pre>{esc(content)}</pre>')
        for tc in tcs:
            fn = tc.get("function", {})
            try:
                args_pretty = json.dumps(json.loads(fn.get("arguments", "{}")), indent=2)
            except Exception:
                args_pretty = fn.get("arguments", "")
            parts.append(f'<div class="tool-call">⚙ <strong>{esc(fn.get("name",""))}</strong><pre>{esc(args_pretty)}</pre></div>')
        parts.append("</div>")
        return "\n".join(parts)

    calls_html_parts = []
    for i, call in enumerate(llm_calls):
        msgs_html = "\n".join(render_message(m) for m in call.get("messages", []))
        resp = call.get("response", {})
        choices = resp.get("choices", [])
        resp_msg = choices[0].get("message", {}) if choices else {}
        resp_html = render_message(resp_msg) if resp_msg else ""
        latency = call.get("latency_ms", 0)
        usage = resp.get("usage", {})
        tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
        calls_html_parts.append(f"""
<details class="llm-call">
  <summary>Call {i+1} — {esc(call.get('model',''))} — {tokens} tokens — {latency}ms</summary>
  <div class="call-body">
    <h4>Messages sent</h4>{msgs_html}
    <h4>Response</h4>{resp_html}
  </div>
</details>""")
    calls_html = "\n".join(calls_html_parts) or "<p class='dim'>No LLM calls recorded.</p>"

    # ── events section ────────────────────────────────────────────────────
    events_rows = []
    for ev in events:
        ts = ev.get("ts", "")[:19].replace("T", " ")
        typ = ev.get("type", "")
        if typ == "tool_log":
            events_rows.append(f'<tr><td class="dim">{esc(ts)}</td><td><span class="badge" style="background:#1a4a7a">tool</span></td><td>{esc(ev.get("log",""))}</td></tr>')
        elif typ == "llm_call":
            tok = ev.get("total_tokens", 0)
            cost = ev.get("cost_usd", 0.0)
            events_rows.append(f'<tr><td class="dim">{esc(ts)}</td><td><span class="badge" style="background:#5a3a2a">llm</span></td><td>tokens={tok}  cost=${cost:.4f}  latency={ev.get("latency_ms",0)}ms</td></tr>')
    events_html = f'<table class="ev-table"><tbody>{"".join(events_rows)}</tbody></table>' if events_rows else "<p class='dim'>No events.</p>"

    title = esc(ctx.get("title", ""))
    arxiv_id = esc(ctx.get("arxiv_id", ""))

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RC: {title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0d0d0d; color: #d0d0d0; font-family: monospace; font-size: 13px; }}
h1 {{ padding: 16px 20px; background: #161616; border-bottom: 1px solid #2a2a2a; font-size: 15px; }}
h1 span {{ color: #888; font-weight: normal; }}
.tabs {{ display: flex; background: #161616; border-bottom: 1px solid #2a2a2a; }}
.tab {{ padding: 10px 20px; cursor: pointer; color: #888; border-bottom: 2px solid transparent; }}
.tab.active {{ color: #d0d0d0; border-bottom-color: #4a9eff; }}
.panel {{ display: none; padding: 20px; max-width: 960px; }}
.panel.active {{ display: block; }}
.card {{ background: #161616; border: 1px solid #2a2a2a; border-radius: 4px; padding: 14px; margin-bottom: 12px; }}
.card h3 {{ font-size: 13px; margin-bottom: 8px; color: #aaa; }}
.url-row {{ margin-bottom: 6px; }}
.url-row a {{ color: #4a9eff; }}
.evidence {{ color: #888; font-style: italic; margin-top: 6px; line-height: 1.5; }}
.resource-row {{ padding: 8px 0; border-top: 1px solid #222; }}
.resource-row:first-child {{ border-top: none; }}
.badge {{ display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 11px; color: #fff; margin-right: 6px; }}
.dim {{ color: #555; }}
.msg {{ padding: 8px 10px; margin: 4px 0; background: #111; border-radius: 3px; }}
.msg-role {{ font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }}
pre {{ white-space: pre-wrap; word-break: break-word; line-height: 1.5; }}
.tool-call {{ background: #1a1a2a; border-radius: 3px; padding: 6px 10px; margin-top: 4px; }}
.llm-call {{ background: #161616; border: 1px solid #2a2a2a; border-radius: 4px; margin-bottom: 8px; }}
.llm-call summary {{ padding: 10px 14px; cursor: pointer; color: #aaa; }}
.llm-call summary:hover {{ color: #d0d0d0; }}
.call-body {{ padding: 0 14px 14px; }}
h4 {{ color: #666; font-size: 11px; text-transform: uppercase; margin: 12px 0 6px; }}
.ev-table {{ width: 100%; border-collapse: collapse; }}
.ev-table td {{ padding: 4px 8px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }}
.error {{ color: #e05555; padding: 12px; background: #1a0a0a; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Resource Checker — {title} <span>({arxiv_id})</span></h1>
<div class="tabs">
  <div class="tab active" onclick="show('report',this)">Report</div>
  <div class="tab" onclick="show('llm','this')">LLM Calls ({len(llm_calls)})</div>
  <div class="tab" onclick="show('events',this)">Events ({len(events)})</div>
  <div class="tab" onclick="show('paper',this)">Paper Context</div>
</div>
<div id="report" class="panel active">{report_html}</div>
<div id="llm" class="panel">{calls_html}</div>
<div id="events" class="panel">{events_html}</div>
<div id="paper" class="panel">
  <div class="card">
    <h3>Authors</h3><div>{esc(ctx.get('authors',''))}</div>
  </div>
  <div class="card">
    <h3>Abstract</h3><pre>{esc(ctx.get('abstract',''))}</pre>
  </div>
  <div class="card">
    <h3>Full Text <span class="dim">({len(ctx.get('full_text',''))//1000}k chars)</span></h3>
    <details><summary class="dim">expand</summary><pre>{esc(ctx.get('full_text','')[:50000])}</pre></details>
  </div>
</div>
<script>
function show(id, el) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run resource checker on a paper")
    parser.add_argument("paper", help="arxiv ID or URL")
    parser.add_argument("--model", default="openai/gpt-4.1")
    args = parser.parse_args()
    asyncio.run(run(args.paper, args.model))
