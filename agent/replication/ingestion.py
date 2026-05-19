"""Ingestion pipeline — converts a paper identifier into a PaperTask.

Entry point: ingest(paper_input, session) -> PaperTask
"""

import asyncio
import re
from typing import Any

import httpx

from agent.replication.paper_reader import run_paper_reader
from agent.replication.resource_checker import run_resource_checker
from agent.replication.rubric_builder import run_rubric_builder
from agent.replication.types import (
    PaperTask,
    ResourceReport,
    RubricNode,
)

_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")


def normalize_arxiv_id(value: str) -> str | None:
    """Extract an arxiv ID from a URL, prefixed string, or bare ID.

    Returns None if no arxiv ID pattern is found (e.g. free-text title).
    """
    match = _ARXIV_ID_RE.search(value)
    return match.group(1) if match else None


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
    reading: Any,
    report: ResourceReport,
    metadata: dict,
    rubric: RubricNode,
) -> PaperTask:
    return PaperTask(
        arxiv_id=reading.arxiv_id,
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


async def ingest(paper_input: str, session: Any) -> PaperTask:
    """Convert a paper identifier into a PaperTask.

    paper_input can be an arxiv ID, arxiv/HF URL, or free-text paper title.
    Raises ValueError if the paper reader fails.
    """
    reading = await run_paper_reader(paper_input, session)
    if reading is None:
        raise ValueError(f"Paper reader failed for input: {paper_input!r}")

    arxiv_id = normalize_arxiv_id(paper_input) or reading.arxiv_id

    report = await run_resource_checker(arxiv_id, reading.github_url, session)
    if report is None:
        raise ValueError(f"Resource checker failed for arxiv_id: {arxiv_id}")

    metadata, rubric = await asyncio.gather(
        _fetch_paper_metadata(arxiv_id),
        run_rubric_builder(arxiv_id, reading.github_url, reading, session),
    )

    return _merge(reading, report, metadata, rubric)
