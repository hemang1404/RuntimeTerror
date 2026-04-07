"""
IncidentEnv — Trajectory-Level Reward Engine.

Computes per-step rewards for every agent action.  Rewards are designed to
give meaningful signal over the full trajectory — not just a binary score at
the end.
"""

from __future__ import annotations

from typing import Any


def _keyword_overlap(text: str, keywords: list[str]) -> float:
    """Fraction of *keywords* found (case-insensitive) in *text*."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def compute_investigation_reward(
    action: dict[str, Any],
    simulator: Any,  # IncidentSimulator
    queries_made: list[dict[str, Any]],
) -> float:
    """Return the immediate reward for one investigation-phase action."""
    action_type = action.get("action_type", "")

    # ── Duplicate query penalty ──────────────────────────────────
    query_sig = (action_type, action.get("service"), action.get("keyword"),
                 action.get("metric"), action.get("file"), action.get("command"))
    for prev in queries_made:
        prev_sig = (prev.get("action_type"), prev.get("service"), prev.get("keyword"),
                    prev.get("metric"), prev.get("file"), prev.get("command"))
        if query_sig == prev_sig:
            return -0.1  # exact duplicate

    # ── query_logs ──────────────────────────────────────────────
    if action_type == "query_logs":
        service = action.get("service", "")
        keyword = action.get("keyword", "")
        if service not in simulator.available_services:
            return -0.05
        # Check if the log output contains root-cause-relevant keywords
        log_output = simulator.query_logs(service, keyword)
        overlap = _keyword_overlap(log_output, simulator.root_cause_keywords)
        if overlap >= 0.3:
            return 0.05
        return 0.0

    # ── query_metrics ───────────────────────────────────────────
    if action_type == "query_metrics":
        metric = action.get("metric", "")
        if metric not in simulator.available_metrics:
            return -0.05
        return 0.05

    # ── inspect_code ────────────────────────────────────────────
    if action_type == "inspect_code":
        file = action.get("file", "")
        if file not in simulator.available_files:
            return -0.05
        if file == simulator.buggy_file:
            return 0.1  # inspecting the right file
        return 0.0

    # ── run_diagnostic ──────────────────────────────────────────
    if action_type == "run_diagnostic":
        command = action.get("command", "")
        if command in simulator.available_commands:
            return 0.05
        return -0.05

    # ── submit_root_cause ───────────────────────────────────────
    if action_type == "submit_root_cause":
        root_cause = action.get("root_cause", "")
        similarity = _keyword_overlap(root_cause, simulator.root_cause_keywords)
        if similarity >= 0.5:
            return 0.4
        if similarity >= 0.3:
            return 0.2
        return -0.2

    return 0.0


def compute_remediation_reward(
    action: dict[str, Any],
    test_passed: bool | None = None,
    some_passed: bool = False,
    timed_out: bool = False,
) -> float:
    """Return the immediate reward for one remediation-phase action."""
    action_type = action.get("action_type", "")

    if action_type == "suggest_fix":
        if test_passed is True:
            return 0.5
        if some_passed:
            return 0.2
        if timed_out:
            return -0.05
        return -0.1

    if action_type == "submit_resolution":
        # Small bonus for wrapping up (penalised elsewhere if no fix attempted)
        return 0.1

    return 0.0


def truncation_penalty() -> float:
    """Penalty applied when the episode is truncated (max steps exceeded)."""
    return -0.3
