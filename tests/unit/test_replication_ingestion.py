"""Tests for replication ingestion — arxiv ID normalisation, rubric construction,
merge logic, and ingest() orchestration."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.replication.ingestion import (
    _fetch_paper_metadata,
    _merge,
    build_rubric,
    ingest,
    normalize_arxiv_id,
)
from agent.replication.types import (
    MetricResult,
    PaperReading,
    ResourceInfo,
    ResourceReport,
    ResourceStatus,
    RubricStatus,
)


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


# ── build_rubric ─────────────────────────────────────────────────────────


def _reading(**kwargs) -> PaperReading:
    defaults = dict(
        arxiv_id="2406.04692",
        title="Test Paper",
        github_url="https://github.com/org/repo",
        metrics=[MetricResult(name="accuracy", value=85.0, dataset="ImageNet val")],
        eval_command_hint="python eval.py --dataset imagenet",
    )
    return PaperReading(**{**defaults, **kwargs})


def test_build_rubric_root_description_includes_title():
    rubric = build_rubric(_reading(title="My Paper"))
    assert "My Paper" in rubric.description


def test_build_rubric_has_three_top_level_children():
    rubric = build_rubric(_reading())
    assert len(rubric.children) == 3
    ids = {c.id for c in rubric.children}
    assert ids == {"env", "eval", "result"}


def test_build_rubric_env_has_two_leaves():
    rubric = build_rubric(_reading())
    env = next(c for c in rubric.children if c.id == "env")
    assert len(env.children) == 2
    assert {c.id for c in env.children} == {"env.deps", "env.imports"}


def test_build_rubric_result_leaf_uses_metric_and_threshold():
    rubric = build_rubric(
        _reading(metrics=[MetricResult(name="mAP", value=84.2, dataset="COCO val")])
    )
    result = next(c for c in rubric.children if c.id == "result")
    leaf = result.children[0]
    assert "mAP" in leaf.description
    assert "84.2" in leaf.description
    # threshold should be 95% of reported
    assert "79.99" in leaf.description or "79.9" in leaf.description


def test_build_rubric_eval_command_hint_used_in_check():
    rubric = build_rubric(_reading(eval_command_hint="python run_eval.py --split val"))
    eval_node = next(c for c in rubric.children if c.id == "eval")
    assert "python run_eval.py --split val" in eval_node.children[0].check


def test_build_rubric_fallback_check_when_no_eval_hint():
    rubric = build_rubric(_reading(eval_command_hint=""))
    eval_node = next(c for c in rubric.children if c.id == "eval")
    assert eval_node.children[0].check == "python eval.py"


def test_build_rubric_all_nodes_start_pending():
    rubric = build_rubric(_reading())
    for leaf in rubric.all_leaves():
        assert leaf.status == RubricStatus.PENDING


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

    task = _merge(reading, report, metadata)

    assert task.arxiv_id == "2406.04692"
    assert task.title == "Test Paper"
    assert task.github_stars == 500
    assert task.abstract == "An abstract."
    assert task.repo_ready is True
    assert len(task.datasets) == 1
    assert task.datasets[0].name == "ImageNet"


def test_merge_uses_zero_stars_when_metadata_missing():
    task = _merge(_reading(), ResourceReport(repo_ready=False, repo_notes=""), {})
    assert task.github_stars == 0
    assert task.abstract == ""


# ── ingest() orchestration ───────────────────────────────────────────────


def _session() -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(model_name="anthropic/test"))


@pytest.mark.asyncio
async def test_ingest_runs_agents_in_parallel_when_arxiv_id_known(monkeypatch):
    reading = _reading()
    report = ResourceReport(repo_ready=True, repo_notes="ok")

    calls = []

    async def fake_paper_reader(paper_input, session):
        calls.append(("paper_reader", paper_input))
        return reading

    async def fake_resource_checker(arxiv_id, github_url, session):
        calls.append(("resource_checker", arxiv_id))
        return report

    async def fake_fetch_metadata(arxiv_id):
        return {"githubStars": 10, "summary": "s"}

    monkeypatch.setattr(
        "agent.replication.ingestion.run_paper_reader", fake_paper_reader
    )
    monkeypatch.setattr(
        "agent.replication.ingestion.run_resource_checker", fake_resource_checker
    )
    monkeypatch.setattr(
        "agent.replication.ingestion._fetch_paper_metadata", fake_fetch_metadata
    )

    task = await ingest("2406.04692", _session())

    assert task.arxiv_id == "2406.04692"
    assert any(c[0] == "paper_reader" for c in calls)
    assert any(c[0] == "resource_checker" for c in calls)


@pytest.mark.asyncio
async def test_ingest_runs_paper_reader_first_for_free_text(monkeypatch):
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

    monkeypatch.setattr(
        "agent.replication.ingestion.run_paper_reader", fake_paper_reader
    )
    monkeypatch.setattr(
        "agent.replication.ingestion.run_resource_checker", fake_resource_checker
    )
    monkeypatch.setattr(
        "agent.replication.ingestion._fetch_paper_metadata", fake_fetch_metadata
    )

    await ingest("Mixture of Agents paper", _session())

    # paper_reader must come before resource_checker for free text
    assert order.index("paper_reader") < order.index("resource_checker")


@pytest.mark.asyncio
async def test_ingest_raises_when_paper_reader_fails(monkeypatch):
    async def fake_paper_reader(paper_input, session):
        return None

    async def fake_resource_checker(arxiv_id, github_url, session):
        return None

    async def fake_fetch_metadata(arxiv_id):
        return {}

    monkeypatch.setattr(
        "agent.replication.ingestion.run_paper_reader", fake_paper_reader
    )
    monkeypatch.setattr(
        "agent.replication.ingestion.run_resource_checker", fake_resource_checker
    )
    monkeypatch.setattr(
        "agent.replication.ingestion._fetch_paper_metadata", fake_fetch_metadata
    )

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

    monkeypatch.setattr(
        "agent.replication.ingestion.run_paper_reader", fake_paper_reader
    )
    monkeypatch.setattr(
        "agent.replication.ingestion.run_resource_checker", fake_resource_checker
    )
    monkeypatch.setattr(
        "agent.replication.ingestion._fetch_paper_metadata", fake_fetch_metadata
    )

    task = await ingest("2406.04692", _session())

    assert task.repo_ready is False
    assert "failed" in task.repo_notes.lower()
