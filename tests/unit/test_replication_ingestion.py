"""Tests for replication ingestion — arxiv ID normalisation, merge logic, and ingest() orchestration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.replication.ingestion import (
    _fetch_paper_metadata,
    _merge,
    ingest,
    normalize_arxiv_id,
)
from agent.replication.types import (
    MetricResult,
    PaperReading,
    ResourceInfo,
    ResourceReport,
    ResourceStatus,
    RubricNode,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _reading(**kwargs) -> PaperReading:
    defaults = dict(
        arxiv_id="2406.04692",
        title="Test Paper",
        github_url="https://github.com/org/repo",
        metrics=[MetricResult(name="accuracy", value=85.0, dataset="ImageNet val")],
    )
    return PaperReading(**{**defaults, **kwargs})


def _rubric() -> RubricNode:
    return RubricNode(id="root", description="Replicate Test Paper")


def _session() -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(model_name="anthropic/test"))


# ── normalize_arxiv_id ───────────────────────────────────────────────────


def test_normalize_bare_arxiv_id():
    assert normalize_arxiv_id("2406.04692") == "2406.04692"


def test_normalize_arxiv_url():
    assert normalize_arxiv_id("https://arxiv.org/abs/2406.04692") == "2406.04692"


def test_normalize_hf_papers_url():
    assert (
        normalize_arxiv_id("https://huggingface.co/papers/2406.04692") == "2406.04692"
    )


def test_normalize_arxiv_with_version_suffix():
    assert normalize_arxiv_id("https://arxiv.org/abs/2406.04692v2") == "2406.04692"


def test_normalize_five_digit_arxiv_id():
    assert normalize_arxiv_id("2605.14269") == "2605.14269"


def test_normalize_returns_none_for_free_text():
    assert normalize_arxiv_id("Mixture of Agents paper") is None


def test_normalize_returns_none_for_empty_string():
    assert normalize_arxiv_id("") is None


# ── _merge ───────────────────────────────────────────────────────────────


def test_merge_builds_paper_task_from_reading_and_report():
    reading = _reading()
    report = ResourceReport(
        repo_ready=True,
        repo_notes="Looks good.",
        datasets=[ResourceInfo(name="ImageNet", status=ResourceStatus.AVAILABLE)],
        models=[],
    )
    metadata = {"githubStars": 500, "summary": "An abstract."}
    rubric = _rubric()

    task = _merge(reading, report, metadata, rubric)

    assert task.arxiv_id == "2406.04692"
    assert task.title == "Test Paper"
    assert task.github_stars == 500
    assert task.abstract == "An abstract."
    assert task.repo_ready is True
    assert len(task.datasets) == 1
    assert task.datasets[0].name == "ImageNet"
    assert task.rubric is rubric


def test_merge_uses_zero_stars_when_metadata_missing():
    task = _merge(_reading(), ResourceReport(repo_ready=False, repo_notes=""), {}, _rubric())
    assert task.github_stars == 0
    assert task.abstract == ""


# ── ingest() orchestration ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_paper_reader_always_runs_first(monkeypatch):
    reading = _reading()
    report = ResourceReport(repo_ready=True, repo_notes="ok")
    order = []

    async def fake_paper_reader(paper_input, session):
        order.append("paper_reader")
        return reading

    async def fake_resource_checker(arxiv_id, github_url, session):
        order.append("resource_checker")
        return report

    async def fake_fetch_metadata(arxiv_id):
        return {}

    async def fake_rubric_builder(arxiv_id, github_url, reading, session):
        return _rubric()

    monkeypatch.setattr("agent.replication.ingestion.run_paper_reader", fake_paper_reader)
    monkeypatch.setattr("agent.replication.ingestion.run_resource_checker", fake_resource_checker)
    monkeypatch.setattr("agent.replication.ingestion._fetch_paper_metadata", fake_fetch_metadata)
    monkeypatch.setattr("agent.replication.ingestion.run_rubric_builder", fake_rubric_builder)

    await ingest("2406.04692", _session())

    assert order[0] == "paper_reader"
    assert "resource_checker" in order


@pytest.mark.asyncio
async def test_ingest_resource_checker_and_rubric_builder_run_after_paper_reader(monkeypatch):
    reading = _reading(arxiv_id="2406.04692")
    report = ResourceReport(repo_ready=True, repo_notes="ok")
    order = []

    async def fake_paper_reader(paper_input, session):
        order.append("paper_reader")
        return reading

    async def fake_resource_checker(arxiv_id, github_url, session):
        order.append("resource_checker")
        return report

    async def fake_fetch_metadata(arxiv_id):
        return {}

    async def fake_rubric_builder(arxiv_id, github_url, reading, session):
        order.append("rubric_builder")
        return _rubric()

    monkeypatch.setattr("agent.replication.ingestion.run_paper_reader", fake_paper_reader)
    monkeypatch.setattr("agent.replication.ingestion.run_resource_checker", fake_resource_checker)
    monkeypatch.setattr("agent.replication.ingestion._fetch_paper_metadata", fake_fetch_metadata)
    monkeypatch.setattr("agent.replication.ingestion.run_rubric_builder", fake_rubric_builder)

    await ingest("Mixture of Agents paper", _session())

    assert order.index("paper_reader") < order.index("resource_checker")
    assert order.index("paper_reader") < order.index("rubric_builder")
    assert order.index("resource_checker") < order.index("rubric_builder")


@pytest.mark.asyncio
async def test_ingest_raises_when_paper_reader_fails(monkeypatch):
    async def fake_paper_reader(paper_input, session):
        return None

    monkeypatch.setattr("agent.replication.ingestion.run_paper_reader", fake_paper_reader)

    with pytest.raises(ValueError, match="Paper reader failed"):
        await ingest("2406.04692", _session())


@pytest.mark.asyncio
async def test_ingest_uses_degraded_report_when_resource_checker_fails(monkeypatch):
    reading = _reading()

    async def fake_paper_reader(paper_input, session):
        return reading

    async def fake_resource_checker(arxiv_id, github_url, session):
        return None

    async def fake_fetch_metadata(arxiv_id):
        return {}

    async def fake_rubric_builder(arxiv_id, github_url, reading, session):
        return _rubric()

    monkeypatch.setattr("agent.replication.ingestion.run_paper_reader", fake_paper_reader)
    monkeypatch.setattr("agent.replication.ingestion.run_resource_checker", fake_resource_checker)
    monkeypatch.setattr("agent.replication.ingestion._fetch_paper_metadata", fake_fetch_metadata)
    monkeypatch.setattr("agent.replication.ingestion.run_rubric_builder", fake_rubric_builder)

    task = await ingest("2406.04692", _session())

    assert task.repo_ready is False
    assert "failed" in task.repo_notes.lower()


@pytest.mark.asyncio
async def test_ingest_rubric_from_builder_is_used(monkeypatch):
    reading = _reading()
    rubric = RubricNode(id="root", description="custom rubric")

    async def fake_paper_reader(paper_input, session):
        return reading

    async def fake_resource_checker(arxiv_id, github_url, session):
        return ResourceReport(repo_ready=True, repo_notes="ok")

    async def fake_fetch_metadata(arxiv_id):
        return {}

    async def fake_rubric_builder(arxiv_id, github_url, reading, session):
        return rubric

    monkeypatch.setattr("agent.replication.ingestion.run_paper_reader", fake_paper_reader)
    monkeypatch.setattr("agent.replication.ingestion.run_resource_checker", fake_resource_checker)
    monkeypatch.setattr("agent.replication.ingestion._fetch_paper_metadata", fake_fetch_metadata)
    monkeypatch.setattr("agent.replication.ingestion.run_rubric_builder", fake_rubric_builder)

    task = await ingest("2406.04692", _session())

    assert task.rubric is rubric
