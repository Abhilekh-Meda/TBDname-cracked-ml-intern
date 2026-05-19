"""Resource checker sub-agent — verifies repo readiness and resource availability."""

import uuid
from typing import Any

from litellm import Message

from agent.replication._sub_agent import run_sub_agent
from agent.replication.types import ResourceInfo, ResourceReport, ResourceStatus

RESOURCE_CHECKER_TOOL_NAMES = {
    "hf_papers",
    "hf_inspect_dataset",
    "github_read_file",
    "github_list_repos",
    "hf_repo_files",
    "web_search",
}

_SUBMIT_TOOL_NAME = "submit_resource_report"

_SUBMIT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _SUBMIT_TOOL_NAME,
        "description": "Submit your resource availability report.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_ready": {
                    "type": "boolean",
                    "description": "True if the repo has real runnable code, not a placeholder",
                },
                "repo_notes": {
                    "type": "string",
                    "description": "Notes on repo state, missing scripts, caveats",
                },
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "hf_id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["available", "gated", "missing", "unknown"],
                            },
                            "notes": {"type": "string"},
                            "source_url": {
                                "type": "string",
                                "description": "Direct download URL or HF Hub URL",
                            },
                            "size_hint": {
                                "type": "string",
                                "description": "Estimated size (e.g. '1.2 GB', '50k samples')",
                            },
                        },
                        "required": ["name", "status"],
                    },
                },
                "models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "hf_id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["available", "gated", "missing", "unknown"],
                            },
                            "notes": {"type": "string"},
                            "source_url": {
                                "type": "string",
                                "description": "Direct download URL or HF Hub URL",
                            },
                            "size_hint": {
                                "type": "string",
                                "description": "Estimated size (e.g. '7B params', '4.2 GB')",
                            },
                        },
                        "required": ["name", "status"],
                    },
                },
            },
            "required": ["repo_ready", "repo_notes", "datasets", "models"],
        },
    },
}

_SUMMARY_PROMPT = (
    "You are summarizing a conversation in which an AI agent is verifying whether the resources "
    "needed to replicate an ML paper are accessible. The agent checks three things: (1) whether "
    "the paper's GitHub repo contains real runnable code or is just a placeholder, (2) whether "
    "the datasets required to run the main evaluation exist on HuggingFace Hub and are publicly "
    "accessible (status: available, gated, missing, or unknown), and (3) whether pretrained model "
    "checkpoints required by the paper exist and are accessible. The agent calls a submit tool "
    "when it has enough information to report on all resources.\n\n"
    "Summarize what has happened in this conversation so far. Include:\n"
    "- The paper being checked: arxiv_id and github_url\n"
    "- Repository findings: whether the repo has real code, missing scripts, or other issues\n"
    "- Every dataset checked: name, HuggingFace ID if found, availability status, and any notes\n"
    "- Every model/checkpoint checked: name, HuggingFace ID if found, availability status, and any notes\n"
    "- What still needs to be verified before submit_resource_report can be called\n"
    "This summary will replace the full conversation history and must contain everything the "
    "agent needs to finish checking resources and submit."
    "Include important details, it is better to give too much information than too little."
)

_SYSTEM_PROMPT = """\
You are a resource checker agent for ML paper replication. Your job is to verify
that the resources needed to replicate a paper are available before any compute
is spent.

Given a paper's arxiv ID and optional GitHub URL, do the following:

1. Check linked resources:
   hf_papers(operation="find_all_resources", arxiv_id=...) — linked datasets and models.

2. Check the repo (if github_url provided):
   github_read_file to read the README and top-level file listing.
   - Is this real runnable code or a placeholder / coming-soon?
   - Are there obvious missing pieces (no eval script, no training code)?

3. For each dataset required:
   hf_inspect_dataset to check if it exists on HF Hub.
   Status: "available" (public), "gated" (requires approval), "missing" (not on HF), "unknown".

4. For each base model or checkpoint required:
   hf_repo_files to check if it exists on HF Hub.
   Same status categories.

5. Call submit_resource_report with your findings.

Focus on resources needed to run the main evaluation, not every dataset mentioned.
"""


async def run_resource_checker(
    arxiv_id: str,
    github_url: str,
    session: Any,
) -> ResourceReport | None:
    """Run the resource checker agent. Returns ResourceReport on success, None on failure."""
    agent_id = uuid.uuid4().hex[:8]

    tool_specs = [
        spec
        for spec in session.tool_router.get_tool_specs_for_llm()
        if spec["function"]["name"] in RESOURCE_CHECKER_TOOL_NAMES
    ]
    tool_specs.append(_SUBMIT_TOOL_SPEC)

    user_msg = f"arxiv_id: {arxiv_id}"
    if github_url:
        user_msg += f"\ngithub_url: {github_url}"

    messages = [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(role="user", content=user_msg),
    ]

    result, ok = await run_sub_agent(
        messages=messages,
        tool_specs=tool_specs,
        submit_tool_name=_SUBMIT_TOOL_NAME,
        session=session,
        agent_id=agent_id,
        agent_label=f"resource-checker: {arxiv_id}",
        summary_prompt=_SUMMARY_PROMPT,
    )

    if not ok or result is None:
        return None

    try:
        datasets = [
            ResourceInfo(
                name=d["name"],
                status=ResourceStatus(d["status"]),
                hf_id=d.get("hf_id"),
                notes=d.get("notes", ""),
                source_url=d.get("source_url", ""),
                size_hint=d.get("size_hint", ""),
            )
            for d in result.get("datasets", [])
        ]
        models = [
            ResourceInfo(
                name=m["name"],
                status=ResourceStatus(m["status"]),
                hf_id=m.get("hf_id"),
                notes=m.get("notes", ""),
                source_url=m.get("source_url", ""),
                size_hint=m.get("size_hint", ""),
            )
            for m in result.get("models", [])
        ]
        return ResourceReport(
            repo_ready=bool(result["repo_ready"]),
            repo_notes=result.get("repo_notes", ""),
            datasets=datasets,
            models=models,
        )
    except (KeyError, TypeError, ValueError):
        return None
