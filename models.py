"""
IncidentEnv — Typed Pydantic Models.

Defines the Action, Observation, and State models for the
incident response simulation environment, compliant with OpenEnv spec.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# We try importing from the openenv package first; if it is not installed we
# provide lightweight stand-ins so the project can still be developed and
# tested without openenv-core being present.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:  # pragma: no cover – standalone dev fallback
    class Action(BaseModel):  # type: ignore[no-redef]
        """Fallback Action base (openenv-core not installed)."""
        metadata: dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):  # type: ignore[no-redef]
        """Fallback Observation base."""
        done: bool = False
        reward: float | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        """Fallback State base."""
        episode_id: str | None = None
        step_count: int = 0


# ───────────────────────────── Action ─────────────────────────────────────


class IncidentAction(Action):
    """Agent's action in the incident response environment.

    Phase 1 — Investigation:
        query_logs        – search service logs
        query_metrics     – get time-series metric data
        inspect_code      – view a source file
        run_diagnostic    – run a shell-like diagnostic command
        submit_root_cause – declare the root cause (ends Phase 1)

    Phase 2 — Remediation:
        suggest_fix         – submit a code patch (env runs tests)
        submit_resolution   – finalise and end episode
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: query_logs, query_metrics, inspect_code, run_diagnostic, "
            "submit_root_cause, suggest_fix, submit_resolution"
        ),
    )

    # Investigation arguments
    service: str = Field(default="", description="Target service name for log queries")
    keyword: str = Field(default="", description="Log search keyword / filter")
    metric: str = Field(default="", description="Metric name (e.g. cpu, error_rate)")
    time_range: str = Field(
        default="5m", description="Time range: 1m | 5m | 15m | 1h"
    )
    file: str = Field(default="", description="File path to inspect or patch")
    command: str = Field(default="", description="Diagnostic command string")

    # Root-cause / fix arguments
    root_cause: str = Field(default="", description="Agent's root-cause explanation")
    patch_code: str = Field(
        default="", description="Complete fixed file content (for suggest_fix)"
    )
    message: str = Field(default="", description="General message / notes")


# ───────────────────────────── Observation ────────────────────────────────


class IncidentObservation(Observation):
    """What the agent observes after each action (or on reset)."""

    # Alert info (available from step 0)
    alert_title: str = ""
    alert_description: str = ""
    severity_level: str = ""
    affected_service: str = ""

    # Investigation results (populated per-step)
    output: str = ""
    output_type: str = ""  # logs | metrics | code | diagnostic | test_result | info

    # Discovery helpers – tell the agent what it can query
    available_services: list[str] = Field(default_factory=list)
    available_files: list[str] = Field(default_factory=list)
    available_metrics: list[str] = Field(default_factory=list)
    available_commands: list[str] = Field(default_factory=list)

    # Episode state
    phase: str = "investigation"  # investigation | remediation
    step_number: int = 0
    max_steps: int = 15  # 10 investigation + 5 remediation
    task_id: str = ""
    difficulty: str = ""

    # Fix results (Phase 2 only)
    test_output: str = ""
    tests_passed: bool | None = None


# ───────────────────────────── State ──────────────────────────────────────


class IncidentState(State):
    """Internal environment state (returned by ``state()``)."""

    task_id: str = ""
    difficulty: str = ""
    phase: str = "investigation"
    incident_id: str = ""

    # Investigation tracking
    queries_made: list[dict[str, Any]] = Field(default_factory=list)
    root_cause_submitted: str = ""
    root_cause_correct: bool = False
    root_cause_similarity: float = 0.0

    # Remediation tracking
    fixes_attempted: int = 0
    fixes_passed: int = 0

    # Scoring
    cumulative_reward: float = 0.0
    grader_score: float = 0.0
