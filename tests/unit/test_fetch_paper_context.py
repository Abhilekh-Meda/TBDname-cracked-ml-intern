"""Tests for fetch_paper_context — deterministic paper fetching."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.replication.ingestion import fetch_paper_context
from agent.replication.types import PaperContext


def _make_response(status: int, json_data: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=json_data or {})
    resp.text = text
    return resp


HF_META = {
    "id": "2405.14734",
    "title": "SimPO: Simple Preference Optimization",
    "summary": "We propose SimPO, a reference-free reward.",
    "githubRepo": "https://github.com/princeton-nlp/simpo",
    "githubStars": 1200,
    "authors": [{"name": "Yu Meng"}, {"name": "Mengzhou Xia"}],
}

ARXIV_HTML = """
<html><body>
<h1 class="ltx_title">SimPO: Simple Preference Optimization</h1>
<div class="ltx_abstract"><p>We propose SimPO.</p></div>
<h2 class="ltx_title">3 Method</h2>
<p>SimPO uses a reference-free reward based on the average log-likelihood.</p>
<h2 class="ltx_title">4 Experiments</h2>
<p>We evaluate on AlpacaEval 2 and Arena-Hard.</p>
</body></html>
"""


@pytest.mark.asyncio
async def test_returns_paper_context_dataclass():
    with patch("agent.replication.ingestion.httpx.AsyncClient") as mock_client_cls:
        client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(side_effect=[
            _make_response(200, HF_META),
            _make_response(200, text=ARXIV_HTML),
        ])

        result = await fetch_paper_context("2405.14734")

    assert isinstance(result, PaperContext)


@pytest.mark.asyncio
async def test_extracts_metadata_fields():
    with patch("agent.replication.ingestion.httpx.AsyncClient") as mock_client_cls:
        client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(side_effect=[
            _make_response(200, HF_META),
            _make_response(200, text=ARXIV_HTML),
        ])

        result = await fetch_paper_context("2405.14734")

    assert result.arxiv_id == "2405.14734"
    assert result.title == "SimPO: Simple Preference Optimization"
    assert result.github_url == "https://github.com/princeton-nlp/simpo"
    assert result.abstract == "We propose SimPO, a reference-free reward."
    assert "Yu Meng" in result.authors
    assert "Mengzhou Xia" in result.authors


@pytest.mark.asyncio
async def test_full_text_contains_sections():
    with patch("agent.replication.ingestion.httpx.AsyncClient") as mock_client_cls:
        client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(side_effect=[
            _make_response(200, HF_META),
            _make_response(200, text=ARXIV_HTML),
        ])

        result = await fetch_paper_context("2405.14734")

    assert "Method" in result.full_text
    assert "Experiments" in result.full_text
    assert "reference-free" in result.full_text


@pytest.mark.asyncio
async def test_falls_back_to_ar5iv_when_arxiv_fails():
    with patch("agent.replication.ingestion.httpx.AsyncClient") as mock_client_cls:
        client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(side_effect=[
            _make_response(200, HF_META),   # HF metadata OK
            _make_response(404),             # arxiv HTML fails
            _make_response(200, text=ARXIV_HTML),  # ar5iv succeeds
        ])

        result = await fetch_paper_context("2405.14734")

    assert "Method" in result.full_text


@pytest.mark.asyncio
async def test_graceful_degradation_when_html_unavailable():
    """When both HTML sources fail, returns abstract-only full_text."""
    with patch("agent.replication.ingestion.httpx.AsyncClient") as mock_client_cls:
        client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(side_effect=[
            _make_response(200, HF_META),
            _make_response(404),
            _make_response(404),
        ])

        result = await fetch_paper_context("2405.14734")

    assert result.title == "SimPO: Simple Preference Optimization"
    assert result.abstract != ""
    assert "Full text not available" in result.full_text


@pytest.mark.asyncio
async def test_raises_when_paper_not_found():
    with patch("agent.replication.ingestion.httpx.AsyncClient") as mock_client_cls:
        client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(return_value=_make_response(404))

        with pytest.raises(ValueError, match="not found"):
            await fetch_paper_context("9999.99999")


@pytest.mark.asyncio
async def test_empty_github_url_when_not_in_metadata():
    meta_no_github = {**HF_META, "githubRepo": None}
    with patch("agent.replication.ingestion.httpx.AsyncClient") as mock_client_cls:
        client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(side_effect=[
            _make_response(200, meta_no_github),
            _make_response(200, text=ARXIV_HTML),
        ])

        result = await fetch_paper_context("2405.14734")

    assert result.github_url == ""
