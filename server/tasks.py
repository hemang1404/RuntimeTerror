"""
NitpickAI — Tasks and Deterministic Graders.

Defines 3 tasks (easy / medium / hard) and a deterministic grading function
that scores agent debugging performance on a 0.0–1.0 scale.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# ── Task definitions ─────────────────────────────────────────────

TASK_DEFINITIONS = {
    "easy_debug": {
        "difficulty": "easy",
        "data_file": "task_easy.json",
        "description": "Obvious bugs: off-by-one, missing returns, wrong operators.",
        "max_steps": 15,
    },
    "medium_debug": {
        "difficulty": "medium",
        "data_file": "task_medium.json",
        "description": "Requires execution to detect: type coercion, mutable defaults, missing keys.",
        "max_steps": 20,
    },
    "hard_debug": {
        "difficulty": "hard",
        "data_file": "task_hard.json",
        "description": "Multi-step debugging: closures, generator exhaustion, cache corruption, float precision.",
        "max_steps": 25,
    },
}

DATA_DIR = Path(__file__).parent / "data"


def load_scenarios(task_id: str) -> list[dict[str, Any]]:
    """Load all debugging scenarios for a given task."""
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
    issue_submitted: str,
    bug_keywords: list[str],
    execution_history: list[dict[str, Any]],
    fixes_attempted: int,
    fixes_passed: int,
    total_steps_used: int,
    max_total_steps: int,
) -> float:
    """Compute a deterministic score in [0.0, 1.0] for one debugging episode.

    Components (weighted):
        issue_accuracy       (25%)  – keyword overlap with ground truth bug
        run_code_quality     (15%)  – did agent run code to investigate?
        fix_quality          (40%)  – fraction of fixes that passed tests
        efficiency           (10%)  – step-usage ratio
        decision_quality     (10%)  – submitted issue? attempted fix?

    Returns
    -------
    float
        Score clamped to [0.0, 1.0].
    """
    # 1. Issue accuracy (25%)
    issue_score = _keyword_overlap(issue_submitted, bug_keywords)

    # 2. Run code quality (15%) — did agent run code or tests?
    ran_code = any(
        h.get("action_type") in ("run_code", "run_tests")
        for h in execution_history
    )
    run_code_score = 1.0 if ran_code else 0.0

    # 3. Fix quality (40%)
    if fixes_attempted > 0:
        fix_score = fixes_passed / fixes_attempted
    else:
        fix_score = 0.0

    # 4. Efficiency (10%) — fewer steps is better
    if max_total_steps > 0:
        efficiency_score = max(0.0, 1.0 - (total_steps_used / max_total_steps))
    else:
        efficiency_score = 0.0

    # 5. Decision quality (10%) — did they submit an issue and attempt a fix?
    did_issue = 1.0 if issue_submitted.strip() else 0.0
    did_fix = 1.0 if fixes_attempted > 0 else 0.0
    decision_score = (did_issue + did_fix) / 2.0

    raw_score = (
        0.25 * issue_score
        + 0.15 * run_code_score
        + 0.40 * fix_score
        + 0.10 * efficiency_score
        + 0.10 * decision_score
    )

    return round(max(0.0, min(1.0, raw_score)), 4)
