"""Resource checker sub-agent — verifies what is needed to replicate a paper."""

import uuid
from typing import Any

from litellm import Message

from agent.replication._sub_agent import run_sub_agent
from agent.replication.types import PaperContext, ResourceInfo, ResourceReport, ResourceStatus

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
        "description": "Submit the completed resource report.",
        "parameters": {
            "type": "object",
            "properties": {
                "github_url_evidence": {
                    "type": "string",
                    "description": (
                        "Explain how you verified this is the correct canonical repo. "
                        "E.g. 'Repo owner princeton-nlp matches author institution Princeton NLP. "
                        "README mentions SimPO and the paper arxiv ID.' "
                        "If you had to search for it, explain what you searched and what you found."
                    ),
                },
                "github_url": {
                    "type": "string",
                    "description": (
                        "The canonical GitHub repository for this paper — maintained by the paper's "
                        "authors or their institution, not a community fork or wrapper. "
                        "Start from the candidate URL in the paper metadata, verify it, and correct "
                        "it if needed using github_list_repos or web_search. "
                        "Empty string if no repo exists."
                    ),
                },
                "repo_runnable_evidence": {
                    "type": "string",
                    "description": (
                        "List the specific files or README sections that confirm your answer. "
                        "E.g. 'README has training instructions. Found train.py, eval.py, "
                        "requirements.txt. Config files in configs/.' "
                        "Or: 'README says coming soon. No training scripts found.'"
                    ),
                },
                "repo_runnable": {
                    "type": "boolean",
                    "description": (
                        "Whether the repo contains real, runnable code for reproducing the paper's "
                        "main results — training or evaluation scripts, install instructions, and "
                        "environment/requirements files. False if it is a placeholder, coming-soon, "
                        "or inference-only wrapper."
                    ),
                },
                "datasets": {
                    "type": "array",
                    "description": (
                        "Datasets the paper uses for its MAIN experiments and evaluation — "
                        "found by reading the Methods and Experiments sections of the paper text. "
                        "Do NOT include datasets only mentioned in related work or baselines. "
                        "For each dataset, check HF Hub with hf_inspect_dataset first; "
                        "if not found there, use web_search to find a download URL."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Dataset name as used in the paper.",
                            },
                            "evidence": {
                                "type": "string",
                                "description": (
                                    "Quote or paraphrase the paper sentence that references this dataset. "
                                    "E.g. 'Section 4.1: We evaluate on the KITTI Eigen-split test set.'"
                                ),
                            },
                            "hf_id": {
                                "type": "string",
                                "description": "HuggingFace Hub ID (org/name) if it exists there, else null.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["available", "gated", "missing", "unknown"],
                                "description": (
                                    "available: public on HF Hub. "
                                    "gated: exists but requires access approval. "
                                    "missing: not found on HF Hub (may exist elsewhere). "
                                    "unknown: could not determine."
                                ),
                            },
                            "source_url": {
                                "type": "string",
                                "description": "Direct download URL if not on HF Hub.",
                            },
                            "size_hint": {
                                "type": "string",
                                "description": "Approximate size if known (e.g. '50k samples', '2.3 GB').",
                            },
                        },
                        "required": ["name", "evidence", "status"],
                    },
                },
                "models": {
                    "type": "array",
                    "description": (
                        "Pretrained model checkpoints the paper's method initializes from — "
                        "base models, backbone weights, or pretrained parameters required before "
                        "any training can begin. Found by reading the Methods section. "
                        "Do NOT include baseline or comparison models. "
                        "Check HF Hub with hf_repo_files first; if not found there, "
                        "use web_search to find where the weights are hosted."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Model name as used in the paper.",
                            },
                            "evidence": {
                                "type": "string",
                                "description": (
                                    "Quote or paraphrase the paper sentence where this model appears. "
                                    "Include the section number. E.g. 'Section 3: We initialize "
                                    "from Llama-3-8B-Instruct.'"
                                ),
                            },
                            "hf_id": {
                                "type": "string",
                                "description": "HuggingFace Hub ID (org/name) if it exists there, else null.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["available", "gated", "missing", "unknown"],
                            },
                            "source_url": {"type": "string"},
                            "size_hint": {"type": "string"},
                        },
                        "required": ["name", "evidence", "status"],
                    },
                },
            },
            "required": [
                "github_url_evidence",
                "github_url",
                "repo_runnable_evidence",
                "repo_runnable",
                "datasets",
                "models",
            ],
        },
    },
}

_SUMMARY_PROMPT = (
    "You are summarizing a conversation in which an AI agent is verifying the resources needed "
    "to replicate an ML paper. The agent has the full paper text in context and must: "
    "(1) confirm the canonical GitHub repo URL and verify it belongs to the paper's authors, "
    "(2) check whether the repo contains real runnable training/evaluation code, "
    "(3) identify datasets used in the paper's main experiments (from the paper text, not HF links), "
    "(4) identify pretrained model checkpoints required before training can begin. "
    "For each resource found, the agent records where in the paper it was referenced.\n\n"
    "Summarize what has happened so far. Include:\n"
    "- The paper and the github URL being checked, and what was found about the repo\n"
    "- Every dataset found so far: name, HF Hub ID if found, availability status, and the paper evidence\n"
    "- Every model checkpoint found: name, HF Hub ID if found, availability status, and the paper evidence\n"
    "- What still needs to be checked before submit_resource_report can be called\n"
    "Include important details — it is better to give too much information than too little."
)

_SYSTEM_PROMPT = """\
You are a resource checker for ML paper replication. Your job is to identify and verify \
every external resource needed to run the paper's main experiments from scratch.

You are given the full text of the paper. Read it carefully.

## Your tasks

### 1. Verify the GitHub URL
The paper metadata provides a candidate GitHub URL. Verify it is the canonical author repo:
- Check that the repo owner or org matches the author names or institution in the paper text.
- Read the README to confirm it describes this paper's method.
- If the URL looks wrong (community fork, wrapper, unrelated project), use github_list_repos \
or web_search to find the correct one.

### 2. Check if the repo is runnable
A repo is runnable if it contains training or evaluation scripts for the paper's main experiments \
— not just inference or a demo. Check for: README install instructions, requirements.txt or \
environment.yml, training/eval scripts. A placeholder or coming-soon repo is NOT runnable.

### 3. Find datasets from the paper text
Read the Methods and Experiments sections. List every dataset the paper evaluates on. \
Do NOT include datasets only mentioned in related work. For each dataset, search HF Hub to \
check availability.

### 4. Find required model checkpoints
Read the Methods section. Identify pretrained models the method initializes from \
(base models, backbone weights). These are required downloads before training. \
Do NOT include baseline comparison models. Check each on HF Hub.

## Important
- Extract datasets and models from the paper text — do not use find_all_resources.
- Record exactly where in the paper you found each resource (section + quote).
- Finish all checks, then call submit_resource_report once with everything.
"""


async def run_resource_checker(
    paper_context: PaperContext,
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

    user_msg = (
        f"arxiv_id: {paper_context.arxiv_id}\n"
        f"candidate github_url: {paper_context.github_url or '(none in metadata)'}\n"
        f"authors: {paper_context.authors}\n\n"
        f"<paper>\n{paper_context.full_text}\n</paper>"
    )

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
        agent_label=f"resource-checker: {paper_context.arxiv_id}",
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
                source_evidence=d.get("evidence", ""),
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
                evidence=m.get("evidence", ""),
                source_url=m.get("source_url", ""),
                size_hint=m.get("size_hint", ""),
            )
            for m in result.get("models", [])
        ]
        return ResourceReport(
            github_url=result.get("github_url", paper_context.github_url),
            github_url_evidence=result.get("github_url_evidence", ""),
            repo_runnable=bool(result["repo_runnable"]),
            repo_runnable_evidence=result.get("repo_runnable_evidence", ""),
            datasets=datasets,
            models=models,
        )
    except (KeyError, TypeError, ValueError):
        return None
