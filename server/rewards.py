"""
RuntimeTerror — Trajectory-Level Reward Engine.

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


def compute_run_code_reward(
    action: dict[str, Any],
    execution_output: str,
    bug_keywords: list[str],
    history: list[dict[str, Any]],
) -> float:
    """Return the immediate reward for a run_code action."""
    # Duplicate detection
    code = action.get("code", "")
    for prev in history:
        if prev.get("action_type") == "run_code" and prev.get("code", "") == code:
            return -0.1  # exact duplicate

    # Check if output reveals bug-related information
    if execution_output:
        overlap = _keyword_overlap(execution_output, bug_keywords)
        if overlap >= 0.2:
            return 0.05  # output reveals something relevant
        # Even error messages are informative
        error_signals = ["Error", "Exception", "Traceback", "assert", "False"]
        if any(sig in execution_output for sig in error_signals):
            return 0.05
    return 0.0


def compute_run_tests_reward(
    test_passed: bool,
    tests_run: int,
    tests_passed: int,
    history: list[dict[str, Any]],
) -> float:
    """Return the immediate reward for a run_tests action."""
    # Diminishing returns for running tests repeatedly without changes
    test_runs = sum(1 for h in history if h.get("action_type") == "run_tests")
    if test_runs >= 3:
        return -0.05  # too many test runs

    if tests_run > 0 and tests_passed < tests_run:
        return 0.05  # test failures = useful signal
    if test_passed:
        return 0.05
    return 0.0


def compute_create_issue_reward(
    issue_description: str,
    bug_keywords: list[str],
) -> float:
    """Return the immediate reward for a create_issue action."""
    similarity = _keyword_overlap(issue_description, bug_keywords)
    if similarity >= 0.5:
        return 0.3
    if similarity >= 0.3:
        return 0.15
    return -0.2


def compute_suggest_fix_reward(
    test_passed: bool,
    some_passed: bool = False,
    timed_out: bool = False,
) -> float:
    """Return the immediate reward for a suggest_fix action."""
    if test_passed:
        return 0.5
    if some_passed:
        return 0.2
    if timed_out:
        return -0.1
    return -0.3


def compute_request_changes_reward(
    fixes_attempted: int,
    fixes_passed: int,
) -> float:
    """Return the immediate reward for a request_changes (finalize) action."""
    if fixes_passed > 0:
        return 1.0   # successful fix + finalize = big bonus
    if fixes_attempted > 0:
        return 0.0   # tried but failed
    return -0.3      # finalized without even trying


def truncation_penalty() -> float:
    """Penalty applied when the episode is truncated (max steps exceeded)."""
    return -0.3


# ── Incident Environment Rewards (WIP) ──────────────────────────────────
# Stubs for the incident response workflow under development.


def compute_investigation_reward(
    action: dict,
    simulator: object,
    queries_made: list[dict],
) -> float:
    """Return the immediate reward for an investigation action (WIP).

    Parameters
    ----------
    action : dict
        The action taken by the agent.
    simulator : IncidentSimulator
        The simulator instance (used for ground truth).
    queries_made : list[dict]
        History of previous queries for duplicate detection.
    """
    atype = action.get("action_type", "")

    # Penalize exact duplicate queries
    for prev in queries_made:
        if prev == action:
            return -0.1

    if atype == "submit_root_cause":
        # Reward based on root cause accuracy
        root_cause = action.get("root_cause", "")
        keywords = getattr(simulator, "root_cause_keywords", [])
        overlap = _keyword_overlap(root_cause, keywords)
        if overlap >= 0.5:
            return 0.3
        if overlap >= 0.3:
            return 0.15
        return -0.1

    # Small positive reward for exploratory actions
    if atype in ("query_logs", "query_metrics", "inspect_code", "run_diagnostic"):
        return 0.05

    return 0.0


def compute_remediation_reward(
    action: dict,
    test_passed: bool = False,
    some_passed: bool = False,
    timed_out: bool = False,
) -> float:
    """Return the immediate reward for a remediation action (WIP).

    Parameters
    ----------
    action : dict
        The action taken by the agent.
    test_passed : bool
        Whether all tests passed after the fix.
    some_passed : bool
        Whether some (but not all) tests passed.
    timed_out : bool
        Whether test execution timed out.
    """
    atype = action.get("action_type", "")

    if atype == "suggest_fix":
        if test_passed:
            return 0.5
        if some_passed:
            return 0.2
        if timed_out:
            return -0.1
        return -0.3

    if atype == "submit_resolution":
        return 0.0  # Score determined by finalize_grading

    return 0.0

