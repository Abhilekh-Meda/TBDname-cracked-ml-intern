"""Types for the replication pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RubricStatus(str, Enum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"


class ResourceStatus(str, Enum):
    AVAILABLE = "available"
    GATED = "gated"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass
class RubricNode:
    id: str
    description: str
    parent_id: Optional[str] = None
    status: RubricStatus = RubricStatus.PENDING
    children: list[RubricNode] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return not self.children

    def all_leaves(self) -> list[RubricNode]:
        if self.is_leaf():
            return [self]
        leaves: list[RubricNode] = []
        for child in self.children:
            leaves.extend(child.all_leaves())
        return leaves

    def passed(self) -> bool:
        if self.is_leaf():
            return self.status == RubricStatus.PASS
        return all(child.passed() for child in self.children)


@dataclass
class ResourceInfo:
    name: str
    status: ResourceStatus
    hf_id: Optional[str] = None
    notes: str = ""
    source_url: str = ""
    size_hint: str = ""


@dataclass
class MetricResult:
    name: str
    value: float
    dataset: str


@dataclass
class PaperReading:
    """Structured output of the paper reader agent."""

    arxiv_id: str
    title: str
    github_url: str
    metrics: list[MetricResult]


@dataclass
class ResourceReport:
    """Structured output of the resource checker agent."""

    repo_ready: bool
    repo_notes: str
    datasets: list[ResourceInfo] = field(default_factory=list)
    models: list[ResourceInfo] = field(default_factory=list)


@dataclass
class PaperTask:
    """Full ingestion output — input to the replication stage."""

    arxiv_id: str
    title: str
    github_url: str
    github_stars: int
    abstract: str
    rubric: RubricNode
    datasets: list[ResourceInfo] = field(default_factory=list)
    models: list[ResourceInfo] = field(default_factory=list)
    repo_ready: bool = False
    repo_notes: str = ""
