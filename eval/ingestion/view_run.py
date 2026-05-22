"""Generate a self-contained HTML viewer for an ingestion run.

Usage:
    uv run eval/view_run.py eval/runs/20260519T020511-2405.14734
    uv run eval/view_run.py eval/runs/20260519T020511-2405.14734 --open

Writes <run_dir>/view.html and optionally opens it in the browser.
"""

import argparse
import json
import sys
import webbrowser
from pathlib import Path


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return lines


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _rubric_html(node: dict, depth: int = 0) -> str:
    nid = node.get("id", "")
    desc = node.get("description", "")
    children = node.get("children", [])
    status = node.get("status", "pending")
    is_leaf = not children

    status_color = {"pass": "#22c55e", "fail": "#ef4444", "pending": "#94a3b8"}.get(status, "#94a3b8")
    leaf_badge = '<span class="leaf-badge">leaf</span>' if is_leaf else ""

    child_html = "".join(_rubric_html(c, depth + 1) for c in children)
    toggle = f'<span class="toggle" onclick="toggleNode(this)">{"▶" if children else " "}</span>' if children else '<span class="toggle-spacer"></span>'

    return f"""
<div class="rubric-node depth-{min(depth,5)}" data-depth="{depth}">
  <div class="rubric-row">
    {toggle}
    <span class="node-status" style="color:{status_color}">●</span>
    <span class="node-id">{_esc(nid)}</span>
    {leaf_badge}
    <span class="node-desc">{_esc(desc)}</span>
  </div>
  {"<div class='rubric-children'>" + child_html + "</div>" if children else ""}
</div>"""


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _msg_html(msg: dict) -> str:
    role = msg.get("role", "")
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls", [])
    name = msg.get("name", "")

    role_color = {
        "system": "#7c3aed",
        "user": "#0369a1",
        "assistant": "#065f46",
        "tool": "#92400e",
    }.get(role, "#475569")

    label = f"{role}" + (f" [{name}]" if name else "")

    parts = [f'<div class="msg-role" style="color:{role_color}">{_esc(label)}</div>']

    if content:
        if isinstance(content, list):
            for block in content:
                btype = block.get("type", "")
                if btype == "thinking":
                    parts.append(f'<details class="thinking-block"><summary>thinking</summary><pre>{_esc(block.get("thinking",""))}</pre></details>')
                elif btype == "text":
                    parts.append(f'<pre class="msg-content">{_esc(block.get("text",""))}</pre>')
                else:
                    parts.append(f'<pre class="msg-content">{_esc(json.dumps(block, indent=2))}</pre>')
        else:
            parts.append(f'<pre class="msg-content">{_esc(str(content))}</pre>')

    for tc in tool_calls or []:
        fn = tc.get("function", {})
        try:
            args = json.dumps(json.loads(fn.get("arguments", "{}")), indent=2)
        except Exception:
            args = fn.get("arguments", "")
        parts.append(f'<div class="tool-call"><span class="tool-name">{_esc(fn.get("name",""))}</span><pre>{_esc(args)}</pre></div>')

    return f'<div class="msg msg-{role}">{"".join(parts)}</div>'


def _response_html(resp: dict) -> str:
    choices = resp.get("choices", [])
    usage = resp.get("usage", {})
    parts = []

    for choice in choices:
        msg = choice.get("message", {})
        parts.append(_msg_html(msg))

    if usage:
        toks = f'prompt={usage.get("prompt_tokens",0)}  completion={usage.get("completion_tokens",0)}  total={usage.get("total_tokens",0)}'
        parts.append(f'<div class="usage-line">{_esc(toks)}</div>')

    return "".join(parts)


def generate(run_dir: Path) -> Path:
    inp = _load_json(run_dir / "input.json") or {}
    events = _load_jsonl(run_dir / "events.jsonl")
    llm_calls = _load_jsonl(run_dir / "llm_calls.jsonl")
    paper_reading = _load_json(run_dir / "paper_reading.json")
    resource_report = _load_json(run_dir / "resource_report.json")
    rubric = _load_json(run_dir / "rubric.json")
    output = _load_json(run_dir / "output.json")
    rubric_error = _load_json(run_dir / "rubric_error.json")

    arxiv_id = inp.get("arxiv_id", run_dir.name)
    model = inp.get("model", "")
    timestamp = inp.get("timestamp", "")
    title = (output or {}).get("title", arxiv_id) if isinstance(output, dict) and "title" in output else arxiv_id

    # stats
    llm_events = [e for e in events if e.get("type") == "llm_call"]
    total_cost = sum(e.get("cost_usd", 0) for e in llm_events)
    total_tokens = sum(e.get("total_tokens", 0) for e in llm_events)
    leaf_count = len(rubric.get("children", [])) if rubric else 0

    def count_leaves(node):
        if not node.get("children"):
            return 1
        return sum(count_leaves(c) for c in node["children"])
    leaf_count = count_leaves(rubric) if rubric else 0

    # ── timeline ─────────────────────────────────────────────────────────────
    timeline_rows = []
    for e in events:
        ts = e.get("ts", "")[:19].replace("T", " ")
        etype = e.get("type", "")
        if etype == "tool_log":
            label = e.get("label", e.get("agent_id", ""))
            log = e.get("log", "")
            timeline_rows.append(
                f'<tr class="ev-tool"><td class="ev-ts">{_esc(ts)}</td>'
                f'<td class="ev-label">{_esc(label)}</td>'
                f'<td class="ev-log">{_esc(log)}</td></tr>'
            )
        elif etype == "llm_call":
            kind = e.get("kind", "")
            tokens = e.get("total_tokens", 0)
            cost = e.get("cost_usd", 0.0)
            latency = e.get("latency_ms", 0)
            timeline_rows.append(
                f'<tr class="ev-llm"><td class="ev-ts">{_esc(ts)}</td>'
                f'<td class="ev-label">llm/{_esc(kind)}</td>'
                f'<td class="ev-log">tokens={tokens}  cost=${cost:.4f}  latency={latency}ms</td></tr>'
            )

    # ── llm calls ─────────────────────────────────────────────────────────────
    llm_call_blocks = []
    for i, call in enumerate(llm_calls):
        ts = call.get("ts", "")[:19].replace("T", " ")
        mdl = call.get("model", "")
        latency = call.get("latency_ms", 0)
        msgs = call.get("messages", [])
        resp = call.get("response", {})
        n_msgs = len(msgs)

        msg_html = "".join(_msg_html(m) for m in msgs)
        resp_html = _response_html(resp)

        llm_call_blocks.append(f"""
<details class="llm-call">
  <summary>
    <span class="call-num">#{i+1}</span>
    <span class="call-ts">{_esc(ts)}</span>
    <span class="call-model">{_esc(mdl)}</span>
    <span class="call-meta">{n_msgs} messages · {latency}ms</span>
  </summary>
  <div class="call-body">
    <div class="call-section-label">Request</div>
    <div class="messages">{msg_html}</div>
    <div class="call-section-label">Response</div>
    <div class="messages">{resp_html}</div>
  </div>
</details>""")

    # ── rubric ────────────────────────────────────────────────────────────────
    rubric_html = _rubric_html(rubric) if rubric else "<p>No rubric.</p>"
    error_html = ""
    if rubric_error:
        error_html = f'<div class="error-box"><strong>Error:</strong> {_esc(rubric_error.get("error",""))}<br><strong>Frontier:</strong> {_esc(", ".join(rubric_error.get("frontier",[])))}<details><summary>Traceback</summary><pre>{_esc(rubric_error.get("traceback",""))}</pre></details></div>'

    # ── paper reading ─────────────────────────────────────────────────────────
    def kv_table(d: dict) -> str:
        rows = ""
        for k, v in d.items():
            if isinstance(v, (list, dict)):
                v_str = f"<pre>{_esc(json.dumps(v, indent=2))}</pre>"
            else:
                v_str = _esc(str(v))
            rows += f"<tr><td class='kv-key'>{_esc(k)}</td><td class='kv-val'>{v_str}</td></tr>"
        return f"<table class='kv-table'>{rows}</table>"

    paper_html = kv_table(paper_reading) if paper_reading else "<p>Not available.</p>"
    resource_html = kv_table(resource_report) if resource_report else "<p>Not available.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} — ingestion run</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e2e8f0; font-size: 14px; line-height: 1.5; }}
  a {{ color: #38bdf8; }}

  /* header */
  .header {{ background: #1e293b; border-bottom: 1px solid #334155; padding: 20px 28px; }}
  .header h1 {{ font-size: 18px; font-weight: 600; color: #f1f5f9; margin-bottom: 6px; }}
  .header-meta {{ color: #94a3b8; font-size: 12px; display: flex; gap: 20px; flex-wrap: wrap; }}
  .header-meta span {{ display: flex; align-items: center; gap: 4px; }}

  /* stats */
  .stats {{ display: flex; gap: 12px; padding: 16px 28px; background: #1e293b; border-bottom: 1px solid #334155; flex-wrap: wrap; }}
  .stat {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px 16px; min-width: 120px; }}
  .stat-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }}
  .stat-value {{ font-size: 20px; font-weight: 700; color: #f1f5f9; margin-top: 2px; }}

  /* tabs */
  .tabs {{ display: flex; gap: 2px; padding: 16px 28px 0; background: #1e293b; border-bottom: 1px solid #334155; }}
  .tab {{ padding: 8px 16px; border-radius: 6px 6px 0 0; cursor: pointer; color: #94a3b8; font-size: 13px; font-weight: 500; border: 1px solid transparent; border-bottom: none; transition: all .15s; }}
  .tab:hover {{ color: #e2e8f0; background: #0f172a; }}
  .tab.active {{ color: #38bdf8; background: #0f172a; border-color: #334155; }}
  .tab-content {{ display: none; padding: 24px 28px; }}
  .tab-content.active {{ display: block; }}

  /* timeline */
  .timeline-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .timeline-table th {{ text-align: left; padding: 8px 12px; color: #64748b; font-weight: 500; border-bottom: 1px solid #1e293b; }}
  .ev-ts {{ color: #64748b; white-space: nowrap; padding: 5px 12px; }}
  .ev-label {{ white-space: nowrap; padding: 5px 12px; font-weight: 600; }}
  .ev-log {{ padding: 5px 12px; color: #cbd5e1; word-break: break-word; }}
  .ev-tool .ev-label {{ color: #38bdf8; }}
  .ev-llm .ev-label {{ color: #fbbf24; }}
  tr:hover td {{ background: #1e293b; }}

  /* llm calls */
  .llm-call {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; margin-bottom: 8px; overflow: hidden; }}
  .llm-call summary {{ padding: 12px 16px; cursor: pointer; display: flex; align-items: center; gap: 12px; list-style: none; }}
  .llm-call summary::-webkit-details-marker {{ display: none; }}
  .llm-call[open] summary {{ border-bottom: 1px solid #334155; }}
  .call-num {{ color: #64748b; font-size: 12px; min-width: 28px; }}
  .call-ts {{ color: #64748b; font-size: 12px; }}
  .call-model {{ color: #a5b4fc; font-weight: 600; flex: 1; }}
  .call-meta {{ color: #64748b; font-size: 12px; }}
  .call-body {{ padding: 16px; }}
  .call-section-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: #64748b; margin: 12px 0 8px; }}
  .call-section-label:first-child {{ margin-top: 0; }}

  /* messages */
  .messages {{ display: flex; flex-direction: column; gap: 8px; }}
  .msg {{ border-radius: 6px; padding: 10px 12px; background: #0f172a; border: 1px solid #1e293b; }}
  .msg-role {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }}
  .msg-content {{ font-size: 12px; white-space: pre-wrap; word-break: break-word; color: #cbd5e1; max-height: 400px; overflow-y: auto; }}
  .tool-call {{ margin-top: 8px; background: #1e293b; border-radius: 4px; padding: 8px; }}
  .tool-name {{ font-size: 12px; font-weight: 700; color: #fbbf24; display: block; margin-bottom: 4px; }}
  .tool-call pre {{ font-size: 11px; color: #94a3b8; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }}
  .usage-line {{ font-size: 11px; color: #64748b; margin-top: 8px; padding-top: 8px; border-top: 1px solid #1e293b; }}
  .thinking-block {{ margin-top: 6px; }}
  .thinking-block summary {{ font-size: 11px; color: #7c3aed; cursor: pointer; }}
  .thinking-block pre {{ font-size: 11px; color: #94a3b8; white-space: pre-wrap; max-height: 300px; overflow-y: auto; margin-top: 6px; }}

  /* rubric */
  .rubric-node {{ padding-left: 0; }}
  .rubric-row {{ display: flex; align-items: flex-start; gap: 6px; padding: 5px 8px; border-radius: 4px; }}
  .rubric-row:hover {{ background: #1e293b; }}
  .rubric-children {{ padding-left: 22px; border-left: 1px solid #1e293b; margin-left: 10px; }}
  .toggle {{ cursor: pointer; color: #64748b; font-size: 10px; min-width: 14px; padding-top: 2px; user-select: none; }}
  .toggle-spacer {{ min-width: 14px; display: inline-block; }}
  .node-status {{ font-size: 10px; padding-top: 3px; }}
  .node-id {{ font-size: 12px; font-weight: 700; color: #38bdf8; white-space: nowrap; }}
  .node-desc {{ font-size: 12px; color: #cbd5e1; }}
  .leaf-badge {{ font-size: 10px; background: #1e293b; color: #64748b; border-radius: 3px; padding: 1px 5px; white-space: nowrap; }}
  .depth-0 > .rubric-row {{ background: #1e293b; border-radius: 6px; margin-bottom: 4px; }}

  /* kv table */
  .kv-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .kv-table tr {{ border-bottom: 1px solid #1e293b; }}
  .kv-table tr:hover td {{ background: #1e293b; }}
  .kv-key {{ color: #94a3b8; font-weight: 600; padding: 8px 12px; width: 180px; vertical-align: top; white-space: nowrap; }}
  .kv-val {{ color: #e2e8f0; padding: 8px 12px; word-break: break-word; }}
  .kv-val pre {{ font-size: 11px; color: #94a3b8; white-space: pre-wrap; }}

  /* error */
  .error-box {{ background: #450a0a; border: 1px solid #ef4444; border-radius: 8px; padding: 16px; margin-bottom: 16px; color: #fca5a5; font-size: 13px; }}
  .error-box pre {{ font-size: 11px; color: #fca5a5; white-space: pre-wrap; margin-top: 8px; max-height: 300px; overflow-y: auto; }}

  /* section header */
  .section-title {{ font-size: 13px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 16px; }}
</style>
</head>
<body>

<div class="header">
  <h1>{_esc(title)}</h1>
  <div class="header-meta">
    <span>arxiv: <strong>{_esc(arxiv_id)}</strong></span>
    <span>model: <strong>{_esc(model)}</strong></span>
    <span>run: {_esc(timestamp)}</span>
  </div>
</div>

<div class="stats">
  <div class="stat"><div class="stat-label">Total Cost</div><div class="stat-value">${total_cost:.4f}</div></div>
  <div class="stat"><div class="stat-label">Total Tokens</div><div class="stat-value">{total_tokens:,}</div></div>
  <div class="stat"><div class="stat-label">LLM Calls</div><div class="stat-value">{len(llm_calls)}</div></div>
  <div class="stat"><div class="stat-label">Leaf Nodes</div><div class="stat-value">{leaf_count}</div></div>
  <div class="stat"><div class="stat-label">Events</div><div class="stat-value">{len(events)}</div></div>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('timeline')">Timeline</div>
  <div class="tab" onclick="showTab('llm')">LLM Calls ({len(llm_calls)})</div>
  <div class="tab" onclick="showTab('rubric')">Rubric ({leaf_count} leaves)</div>
  <div class="tab" onclick="showTab('paper')">Paper Reading</div>
  <div class="tab" onclick="showTab('resources')">Resources</div>
</div>

<div id="tab-timeline" class="tab-content active">
  <table class="timeline-table">
    <thead><tr><th>Time</th><th>Agent</th><th>Event</th></tr></thead>
    <tbody>{"".join(timeline_rows)}</tbody>
  </table>
</div>

<div id="tab-llm" class="tab-content">
  {"".join(llm_call_blocks) if llm_call_blocks else "<p>No LLM calls recorded.</p>"}
</div>

<div id="tab-rubric" class="tab-content">
  {error_html}
  <div class="rubric-tree">{rubric_html}</div>
</div>

<div id="tab-paper" class="tab-content">
  {paper_html}
</div>

<div id="tab-resources" class="tab-content">
  {resource_html}
</div>

<script>
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}
function toggleNode(el) {{
  const children = el.closest('.rubric-node').querySelector('.rubric-children');
  if (!children) return;
  const hidden = children.style.display === 'none';
  children.style.display = hidden ? '' : 'none';
  el.textContent = hidden ? '▼' : '▶';
}}
// collapse all internal nodes by default except depth 0
document.querySelectorAll('.rubric-children').forEach(el => {{
  const depth = parseInt(el.closest('.rubric-node').dataset.depth || 0);
  if (depth >= 1) {{
    el.style.display = 'none';
    const toggle = el.closest('.rubric-node').querySelector('.toggle');
    if (toggle) toggle.textContent = '▶';
  }}
}});
</script>
</body>
</html>"""

    out_path = run_dir / "view.html"
    out_path.write_text(html)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML viewer for an ingestion run")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--open", action="store_true", help="Open in browser after generating")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    out = generate(run_dir)
    print(f"Generated: {out}")
    if args.open:
        webbrowser.open(out.as_uri())
