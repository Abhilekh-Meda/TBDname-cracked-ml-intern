"""Paper reader sub-agent — extracts metadata and rubric data from a paper."""

import uuid
from typing import Any

from litellm import Message

from agent.replication._sub_agent import run_sub_agent
from agent.replication.types import MetricResult, PaperReading

PAPER_READER_TOOL_NAMES = {"hf_papers", "web_search"}

_SUBMIT_TOOL_NAME = "submit_paper_reading"

_SUBMIT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _SUBMIT_TOOL_NAME,
        "description": "Submit your extracted paper information.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "The arxiv ID (e.g. '2406.04692')",
                },
                "title": {"type": "string"},
                "github_url": {
                    "type": "string",
                    "description": "GitHub repo URL, empty string if not found",
                },
                "metrics": {
                    "type": "array",
                    "description": "Main evaluation metrics (first entry is the primary headline result)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Metric name (e.g. 'accuracy', 'mAP@50', 'F1')",
                            },
                            "value": {
                                "type": "number",
                                "description": "Reported numeric value",
                            },
                            "dataset": {
                                "type": "string",
                                "description": "Dataset and split (e.g. 'ImageNet val', 'COCO test-dev')",
                            },
                        },
                        "required": ["name", "value", "dataset"],
                    },
                    "minItems": 1,
                },
            },
            "required": [
                "arxiv_id",
                "title",
                "github_url",
                "metrics",
            ],
        },
    },
}

_SYSTEM_PROMPT = """\
You are a paper reading agent for ML paper replication. Your job is to extract
structured information from a paper so it can be replicated.

Given a paper identifier (arxiv ID, URL, or title), do the following:

1. Fetch paper details:
   hf_papers(operation="paper_details", arxiv_id=...) — get metadata and github URL.
   If you only have a title, use hf_papers(operation="search", query=...) first.

2. Read the paper:
   hf_papers(operation="read_paper", arxiv_id=...) — get TOC.
   Then read the experiments and results sections by number.

3. Extract the main results:
   - The primary headline metric the authors emphasize (first in the list)
   - Any closely related secondary metrics (e.g. mAP@50 if primary is mAP@75, top-5 if primary is top-1)
   - The exact numeric values and the dataset/split each was measured on

4. Call submit_paper_reading with your findings.

Report all metrics the authors highlight in the abstract or main results table.
If github_url is not in paper_details, set it to an empty string.
"""


async def run_paper_reader(paper_input: str, session: Any) -> PaperReading | None:
    """Run the paper reader agent. Returns PaperReading on success, None on failure."""
    agent_id = uuid.uuid4().hex[:8]

    tool_specs = [
        spec
        for spec in session.tool_router.get_tool_specs_for_llm()
        if spec["function"]["name"] in PAPER_READER_TOOL_NAMES
    ]
    tool_specs.append(_SUBMIT_TOOL_SPEC)

    messages = [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(role="user", content=f"Paper: {paper_input}"),
    ]

    result, ok = await run_sub_agent(
        messages=messages,
        tool_specs=tool_specs,
        submit_tool_name=_SUBMIT_TOOL_NAME,
        session=session,
        agent_id=agent_id,
        agent_label=f"paper-reader: {paper_input[:50]}",
    )

    if not ok or result is None:
        return None

    try:
        metrics = [
            MetricResult(
                name=m["name"],
                value=float(m["value"]),
                dataset=m["dataset"],
            )
            for m in result["metrics"]
        ]
        return PaperReading(
            arxiv_id=result["arxiv_id"],
            title=result["title"],
            github_url=result.get("github_url", ""),
            metrics=metrics,
        )
    except (KeyError, TypeError, ValueError):
        return None
