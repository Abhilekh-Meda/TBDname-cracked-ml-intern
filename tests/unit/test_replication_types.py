"""Tests for replication type primitives — RubricNode tree behaviour."""

from agent.replication.types import (
    MetricResult,
    PaperTask,
    ResourceInfo,
    ResourceStatus,
    RubricNode,
    RubricStatus,
)


# ── RubricNode leaf detection ────────────────────────────────────────────


def test_rubric_node_with_no_children_is_leaf():
    node = RubricNode(id="a", description="x", check="cmd")
    assert node.is_leaf()


def test_rubric_node_with_children_is_not_leaf():
    child = RubricNode(id="b", description="y", check="cmd2")
    node = RubricNode(id="a", description="x", check="", children=[child])
    assert not node.is_leaf()


# ── RubricNode.all_leaves ────────────────────────────────────────────────


def test_all_leaves_on_leaf_returns_self():
    node = RubricNode(id="a", description="x", check="cmd")
    assert node.all_leaves() == [node]


def test_all_leaves_returns_only_leaf_nodes():
    leaf1 = RubricNode(id="c", description="c", check="cmd_c")
    leaf2 = RubricNode(id="d", description="d", check="cmd_d")
    mid = RubricNode(id="b", description="b", check="", children=[leaf1, leaf2])
    root = RubricNode(id="a", description="a", check="", children=[mid])

    leaves = root.all_leaves()
    assert leaves == [leaf1, leaf2]


def test_all_leaves_flattens_deep_tree():
    leaf = RubricNode(id="deep", description="deep", check="cmd")
    mid2 = RubricNode(id="m2", description="m2", check="", children=[leaf])
    mid1 = RubricNode(id="m1", description="m1", check="", children=[mid2])
    root = RubricNode(id="root", description="root", check="", children=[mid1])

    assert root.all_leaves() == [leaf]


# ── RubricNode.passed ────────────────────────────────────────────────────


def test_leaf_passes_when_status_is_pass():
    node = RubricNode(id="a", description="x", check="cmd", status=RubricStatus.PASS)
    assert node.passed()


def test_leaf_fails_when_status_is_fail():
    node = RubricNode(id="a", description="x", check="cmd", status=RubricStatus.FAIL)
    assert not node.passed()


def test_leaf_fails_when_status_is_pending():
    node = RubricNode(id="a", description="x", check="cmd", status=RubricStatus.PENDING)
    assert not node.passed()


def test_parent_passes_when_all_leaves_pass():
    leaf1 = RubricNode(id="c", description="c", check="", status=RubricStatus.PASS)
    leaf2 = RubricNode(id="d", description="d", check="", status=RubricStatus.PASS)
    parent = RubricNode(id="p", description="p", check="", children=[leaf1, leaf2])
    assert parent.passed()


def test_parent_fails_when_any_leaf_fails():
    leaf1 = RubricNode(id="c", description="c", check="", status=RubricStatus.PASS)
    leaf2 = RubricNode(id="d", description="d", check="", status=RubricStatus.FAIL)
    parent = RubricNode(id="p", description="p", check="", children=[leaf1, leaf2])
    assert not parent.passed()


def test_parent_ignores_own_status_and_delegates_to_children():
    # A non-leaf node's own status field is not consulted — children determine pass/fail.
    leaf = RubricNode(id="c", description="c", check="", status=RubricStatus.PASS)
    parent = RubricNode(
        id="p",
        description="p",
        check="",
        status=RubricStatus.FAIL,  # should be ignored
        children=[leaf],
    )
    assert parent.passed()


# ── ResourceStatus enum ──────────────────────────────────────────────────


def test_resource_status_round_trips_from_string():
    assert ResourceStatus("available") == ResourceStatus.AVAILABLE
    assert ResourceStatus("gated") == ResourceStatus.GATED
    assert ResourceStatus("missing") == ResourceStatus.MISSING
    assert ResourceStatus("unknown") == ResourceStatus.UNKNOWN


# ── ResourceInfo defaults ────────────────────────────────────────────────


def test_resource_info_defaults():
    r = ResourceInfo(name="ImageNet", status=ResourceStatus.AVAILABLE)
    assert r.hf_id is None
    assert r.notes == ""
    assert r.source_url == ""
    assert r.size_hint == ""


def test_resource_info_with_source_and_size():
    r = ResourceInfo(
        name="COCO",
        status=ResourceStatus.AVAILABLE,
        hf_id="org/coco",
        source_url="https://huggingface.co/datasets/org/coco",
        size_hint="18 GB",
    )
    assert r.source_url == "https://huggingface.co/datasets/org/coco"
    assert r.size_hint == "18 GB"


# ── MetricResult ─────────────────────────────────────────────────────────


def test_metric_result_fields():
    m = MetricResult(name="mAP@50", value=73.5, dataset="COCO val2017")
    assert m.name == "mAP@50"
    assert m.value == 73.5
    assert m.dataset == "COCO val2017"


# ── PaperTask construction ───────────────────────────────────────────────


def _leaf(node_id: str) -> RubricNode:
    return RubricNode(id=node_id, description="x", check="cmd")


def test_paper_task_defaults():
    task = PaperTask(
        arxiv_id="2406.04692",
        title="Test Paper",
        github_url="https://github.com/org/repo",
        github_stars=100,
        abstract="An abstract.",
        rubric=_leaf("root"),
    )
    assert task.datasets == []
    assert task.models == []
    assert task.repo_ready is False
    assert task.repo_notes == ""


def test_paper_task_with_resources():
    ds = ResourceInfo(name="COCO", status=ResourceStatus.AVAILABLE, hf_id="org/coco")
    model = ResourceInfo(name="ResNet-50", status=ResourceStatus.GATED)
    task = PaperTask(
        arxiv_id="2406.04692",
        title="Test Paper",
        github_url="",
        github_stars=0,
        abstract="",
        rubric=_leaf("root"),
        datasets=[ds],
        models=[model],
        repo_ready=True,
        repo_notes="Looks good.",
    )
    assert len(task.datasets) == 1
    assert task.datasets[0].hf_id == "org/coco"
    assert task.models[0].status == ResourceStatus.GATED
    assert task.repo_ready is True
