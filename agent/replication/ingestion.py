"""Ingestion pipeline — converts a paper identifier into a PaperTask.

Entry point: ingest(paper_input, session) -> PaperTask
"""

import asyncio
import re
from typing import Any

import httpx

from agent.replication.paper_reader import run_paper_reader
from agent.replication.resource_checker import run_resource_checker
from agent.replication.types import (
    MetricResult,
    PaperReading,
    PaperTask,
    ResourceReport,
    RubricNode,
    RubricStatus,
)

_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")


def normalize_arxiv_id(value: str) -> str | None:
    """Extract an arxiv ID from a URL, prefixed string, or bare ID.

    Returns None if no arxiv ID pattern is found (e.g. free-text title).
    """
    match = _ARXIV_ID_RE.search(value)
    return match.group(1) if match else None


def build_rubric(reading: PaperReading) -> RubricNode:
    """Build the rubric tree from extracted paper information."""
    primary: MetricResult = reading.metrics[0]
    threshold = primary.value * 0.95

    env = RubricNode(id="env", description="Environment works", check="", parent_id="root")
    env.children = [
        RubricNode(
            id="env.deps",
            description="Dependencies install",
            check="pip install -r requirements.txt",
            parent_id="env",
        ),
        RubricNode(
            id="env.imports",
            description="Key imports succeed",
            check="python -c 'import torch; print(torch.__version__)'",
            parent_id="env",
        ),
    ]

    eval_node = RubricNode(
        id="eval", description="Evaluation runs", check="", parent_id="root"
    )
    eval_node.children = [
        RubricNode(
            id="eval.runs",
            description="Eval script exits cleanly",
            check=reading.eval_command_hint or "python eval.py",
            parent_id="eval",
        ),
    ]

    result_node = RubricNode(
        id="result", description="Result matches reported value", check="", parent_id="root"
    )
    result_node.children = [
        RubricNode(
            id="result.match",
            description=(
                f"Parsed {primary.name} >= {threshold:.4f} "
                f"(reported: {primary.value})"
            ),
            check=f"grep -i '{primary.name}' eval_output.txt",
            parent_id="result",
        ),
    ]

    root = RubricNode(
        id="root",
        description=f"Replicate {reading.title}",
        check="",
        children=[env, eval_node, result_node],
    )
    return root


async def _fetch_paper_metadata(arxiv_id: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"https://huggingface.co/api/papers/{arxiv_id}")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}


def _merge(
    reading: PaperReading,
    report: ResourceReport,
    metadata: dict,
) -> PaperTask:
    return PaperTask(
        arxiv_id=reading.arxiv_id,
        title=reading.title,
        github_url=reading.github_url,
        github_stars=metadata.get("githubStars", 0),
        abstract=metadata.get("summary", ""),
        rubric=build_rubric(reading),
        datasets=report.datasets,
        models=report.models,
        repo_ready=report.repo_ready,
        repo_notes=report.repo_notes,
    )


async def ingest(paper_input: str, session: Any) -> PaperTask:
    """Convert a paper identifier into a PaperTask.

    paper_input can be an arxiv ID, arxiv/HF URL, or free-text paper title.
    Raises ValueError if the paper reader fails.
    """
    arxiv_id = normalize_arxiv_id(paper_input)

    if arxiv_id:
        reading, report, metadata = await asyncio.gather(
            run_paper_reader(paper_input, session),
            run_resource_checker(arxiv_id, "", session),
            _fetch_paper_metadata(arxiv_id),
        )
    else:
        # Need arxiv_id from the paper reader before starting resource check
        reading = await run_paper_reader(paper_input, session)
        if reading is None:
            raise ValueError(f"Paper reader failed for input: {paper_input!r}")
        arxiv_id = reading.arxiv_id
        report, metadata = await asyncio.gather(
            run_resource_checker(arxiv_id, reading.github_url, session),
            _fetch_paper_metadata(arxiv_id),
        )

    if reading is None:
        raise ValueError(f"Paper reader failed for input: {paper_input!r}")

    if report is None:
        report = ResourceReport(repo_ready=False, repo_notes="Resource check failed.")

    return _merge(reading, report, metadata)
