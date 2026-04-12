"""
NitpickAI — Typed Pydantic Models.

Defines the Action, Observation, and State models for the
interactive debugging environment, compliant with OpenEnv spec.
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


class DebugAction(Action):
    """Agent's action in the interactive debugging environment.

    Actions:
        run_code          – execute a code snippet in the sandbox
        run_tests         – run the visible test suite against current code
        create_issue      – describe the identified bug
        suggest_fix       – submit patched source code
        request_changes   – finalize the episode (submit final decision)
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: run_code, run_tests, create_issue, suggest_fix, "
            "request_changes"
        ),
    )

    # run_code arguments
    code: str = Field(
        default="",
        description="Code snippet to execute in the sandbox (for run_code)",
    )

    # suggest_fix arguments
    patch_code: str = Field(
        default="",
        description="Complete fixed source code (for suggest_fix)",
    )

    # create_issue arguments
    issue_description: str = Field(
        default="",
        description="Description of the identified bug (for create_issue)",
    )

    # request_changes arguments
    message: str = Field(
        default="",
        description="Final notes / summary (for request_changes)",
    )


# ───────────────────────────── Observation ────────────────────────────────


class DebugObservation(Observation):
    """What the agent observes after each action (or on reset)."""

    # Source code under test
    code: str = ""
    visible_tests: list[str] = Field(default_factory=list)

    # Execution results
    execution_output: str = ""
    test_results: str = ""
    tests_passed: bool | None = None

    # Episode state
    step_number: int = 0
    max_steps: int = 20
    task_id: str = ""
    difficulty: str = ""

    # Action feedback
    action_feedback: str = ""


# ───────────────────────────── State ──────────────────────────────────────


class DebugState(State):
    """Internal environment state (returned by ``state()``)."""

    task_id: str = ""
    difficulty: str = ""

    # Source code
    code: str = ""
    visible_tests: list[str] = Field(default_factory=list)
    hidden_tests: list[str] = Field(default_factory=list)

    # Debugging progress
    execution_history: list[dict[str, Any]] = Field(default_factory=list)
    last_output: str = ""

    # Issue tracking
    issue_submitted: str = ""
    issue_correct: bool = False
    issue_similarity: float = 0.0

    # Fix tracking
    fixes_attempted: int = 0
    fixes_passed: int = 0

    # Scoring
    cumulative_reward: float = 0.0
    grader_score: float = 0.0
