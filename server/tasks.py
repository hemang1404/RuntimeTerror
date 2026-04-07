"""
IncidentEnv — Tasks and Deterministic Graders.

Defines 3 tasks (easy / medium / hard) and a deterministic grading function
that scores agent performance on a 0.0–1.0 scale.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

# ── Task definitions ─────────────────────────────────────────────

TASK_DEFINITIONS = {
    "easy_triage": {
        "difficulty": "easy",
        "data_file": "task_easy.json",
        "description": "Single clear-cut root cause with obvious signal in logs.",
        "max_investigation_steps": 10,
        "max_remediation_steps": 5,
    },
    "medium_triage": {
        "difficulty": "medium",
        "data_file": "task_medium.json",
        "description": "Requires correlating signals from multiple services and metrics.",
        "max_investigation_steps": 10,
        "max_remediation_steps": 5,
    },
    "hard_triage": {
        "difficulty": "hard",
        "data_file": "task_hard.json",
        "description": "Subtle bugs requiring deep investigation: race conditions, encoding issues, clock skew.",
        "max_investigation_steps": 10,
        "max_remediation_steps": 5,
    },
}

DATA_DIR = Path(__file__).parent / "data"


def load_scenarios(task_id: str) -> list[dict[str, Any]]:
    """Load all incident scenarios for a given task."""
    task_def = TASK_DEFINITIONS.get(task_id)
    if task_def is None:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_DEFINITIONS)}")
    data_path = DATA_DIR / task_def["data_file"]
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_scenario(task_id: str, seed: int | None = None) -> dict[str, Any]:
    """Pick a random scenario for the task, optionally seeded."""
    scenarios = load_scenarios(task_id)
    if seed is not None:
        rng = random.Random(seed)
        return rng.choice(scenarios)
    return random.choice(scenarios)


def get_task_info(task_id: str) -> dict[str, Any]:
    """Return metadata for a task (without loading data)."""
    task_def = TASK_DEFINITIONS.get(task_id)
    if task_def is None:
        raise ValueError(f"Unknown task: {task_id}")
    return {
        "task_id": task_id,
        **task_def,
    }


# ── Deterministic Grader ─────────────────────────────────────────

def _keyword_overlap(text: str, keywords: list[str]) -> float:
    """Fraction of *keywords* found (case-insensitive) in *text*."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def grade_episode(
    root_cause_submitted: str,
    root_cause_keywords: list[str],
    queries_made: list[dict[str, Any]],
    buggy_file: str,
    fixes_attempted: int,
    fixes_passed: int,
    max_investigation_steps: int,
    investigation_steps_used: int,
    total_steps_used: int,
    max_total_steps: int,
) -> float:
    """Compute a deterministic score in [0.0, 1.0] for one episode.

    Components (weighted):
        root_cause_accuracy  (30%)  – keyword overlap
        investigation_quality(15%)  – did agent inspect the buggy file?
        fix_quality          (35%)  – fraction of fixes that passed tests
        efficiency           (10%)  – step-usage ratio
        decision_quality     (10%)  – submitted root cause at all?

    Returns
    -------
    float
        Score clamped to [0.0, 1.0].
    """
    # 1. Root-cause accuracy (30%)
    root_cause_score = _keyword_overlap(root_cause_submitted, root_cause_keywords)

    # 2. Investigation quality (15%)
    inspected_buggy = any(
        q.get("action_type") == "inspect_code" and q.get("file") == buggy_file
        for q in queries_made
    )
    investigation_score = 1.0 if inspected_buggy else 0.0

    # 3. Fix quality (35%)
    if fixes_attempted > 0:
        fix_score = fixes_passed / fixes_attempted
    else:
        fix_score = 0.0

    # 4. Efficiency (10%) — fewer steps is better
    if max_total_steps > 0:
        efficiency_score = max(0.0, 1.0 - (total_steps_used / max_total_steps))
    else:
        efficiency_score = 0.0

    # 5. Decision quality (10%) — did they submit a root cause?
    decision_score = 1.0 if root_cause_submitted.strip() else 0.0

    raw_score = (
        0.30 * root_cause_score
        + 0.15 * investigation_score
        + 0.35 * fix_score
        + 0.10 * efficiency_score
        + 0.10 * decision_score
    )

    return round(max(0.0, min(1.0, raw_score)), 4)
