"""
IncidentEnv — Core Environment.

Two-phase episode controller:
  Phase 1 (Investigation): agent queries logs, metrics, code, diagnostics
  Phase 2 (Remediation):   agent submits code fixes validated by test execution

Implements the OpenEnv Environment interface: reset(), step(), state.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..models import IncidentAction, IncidentObservation, IncidentState

from .executor import CodeExecutor, ExecutionResult
from .rewards import (
    compute_investigation_reward,
    compute_remediation_reward,
    truncation_penalty,
    _keyword_overlap,
)
from .simulator import IncidentSimulator
from .tasks import (
    TASK_DEFINITIONS,
    grade_episode,
    pick_scenario,
)

# Try importing from openenv; fall back to a plain base class.
try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:

    class Environment:  # type: ignore[no-redef]
        """Lightweight fallback when openenv-core is not installed."""

        pass


# ──────────────────────────────────────────────────────────────────


VALID_INVESTIGATION_ACTIONS = {
    "query_logs",
    "query_metrics",
    "inspect_code",
    "run_diagnostic",
    "submit_root_cause",
}
VALID_REMEDIATION_ACTIONS = {
    "suggest_fix",
    "submit_resolution",
}


class IncidentEnvironment(Environment):
    """Two-phase incident-response simulation environment."""

    def __init__(self) -> None:
        super().__init__()
        self._state = IncidentState()
        self._simulator: IncidentSimulator | None = None
        self._executor = CodeExecutor()
        self._scenario: dict[str, Any] = {}
        self._task_def: dict[str, Any] = {}

    # ── OpenEnv API ──────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "easy_triage",
        **kwargs: Any,
    ) -> IncidentObservation:
        """Start a new incident episode.

        Parameters
        ----------
        task_id : str
            One of ``easy_triage``, ``medium_triage``, ``hard_triage``.
        seed : int | None
            Seed for reproducible scenario selection.
        """
        task_def = TASK_DEFINITIONS.get(task_id)
        if task_def is None:
            raise ValueError(
                f"Unknown task_id: {task_id!r}. "
                f"Available: {list(TASK_DEFINITIONS)}"
            )

        self._task_def = task_def
        self._scenario = pick_scenario(task_id, seed=seed)
        self._simulator = IncidentSimulator(self._scenario)

        max_steps = (
            task_def["max_investigation_steps"]
            + task_def["max_remediation_steps"]
        )

        self._state = IncidentState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=task_def["difficulty"],
            phase="investigation",
            incident_id=self._scenario["id"],
            queries_made=[],
            root_cause_submitted="",
            root_cause_correct=False,
            root_cause_similarity=0.0,
            fixes_attempted=0,
            fixes_passed=0,
            cumulative_reward=0.0,
            grader_score=0.0,
        )

        alert = self._scenario.get("alert", {})
        return IncidentObservation(
            done=False,
            reward=0.0,
            alert_title=alert.get("title", ""),
            alert_description=alert.get("description", ""),
            severity_level=alert.get("severity", ""),
            affected_service=alert.get("affected_service", ""),
            output="Incident assigned. Begin investigation.",
            output_type="info",
            available_services=self._simulator.available_services,
            available_files=self._simulator.available_files,
            available_metrics=self._simulator.available_metrics,
            available_commands=self._simulator.available_commands,
            phase="investigation",
            step_number=0,
            max_steps=max_steps,
            task_id=task_id,
            difficulty=task_def["difficulty"],
        )

    def step(self, action: IncidentAction | dict, **kwargs: Any) -> IncidentObservation:
        """Process one agent action and return the observation + reward."""
        if isinstance(action, dict):
            action = IncidentAction(**action)

        if self._simulator is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        action_dict = action.model_dump()
        max_steps = (
            self._task_def["max_investigation_steps"]
            + self._task_def["max_remediation_steps"]
        )

        # ── Truncation check ────────────────────────────────────
        if self._state.step_count > max_steps:
            penalty = truncation_penalty()
            self._state.cumulative_reward += penalty
            self._finalize_grading()
            return self._make_observation(
                output="Episode truncated — max steps exceeded.",
                output_type="info",
                reward=penalty,
                done=True,
            )

        # ── Phase routing ───────────────────────────────────────
        if self._state.phase == "investigation":
            return self._handle_investigation(action, action_dict)
        else:
            return self._handle_remediation(action, action_dict)

    @property
    def state(self) -> IncidentState:
        return self._state

    # ── Investigation phase ──────────────────────────────────────

    def _handle_investigation(
        self, action: IncidentAction, action_dict: dict
    ) -> IncidentObservation:
        assert self._simulator is not None
        atype = action.action_type

        if atype not in VALID_INVESTIGATION_ACTIONS:
            return self._make_observation(
                output=(
                    f"Invalid investigation action: '{atype}'. "
                    f"Valid: {sorted(VALID_INVESTIGATION_ACTIONS)}"
                ),
                output_type="info",
                reward=-0.05,
                done=False,
            )

        reward = compute_investigation_reward(
            action_dict, self._simulator, self._state.queries_made
        )
        self._state.cumulative_reward += reward
        self._state.queries_made.append(action_dict)

        # Dispatch to simulator
        output = ""
        output_type = ""

        if atype == "query_logs":
            output = self._simulator.query_logs(action.service, action.keyword)
            output_type = "logs"

        elif atype == "query_metrics":
            output = self._simulator.query_metrics(action.metric, action.time_range)
            output_type = "metrics"

        elif atype == "inspect_code":
            output = self._simulator.inspect_code(action.file)
            output_type = "code"

        elif atype == "run_diagnostic":
            output = self._simulator.run_diagnostic(action.command)
            output_type = "diagnostic"

        elif atype == "submit_root_cause":
            similarity = _keyword_overlap(
                action.root_cause, self._simulator.root_cause_keywords
            )
            self._state.root_cause_submitted = action.root_cause
            self._state.root_cause_similarity = round(similarity, 4)
            self._state.root_cause_correct = similarity >= 0.5

            # Transition to Phase 2
            self._state.phase = "remediation"
            output = (
                f"Root cause submitted (confidence: {similarity:.0%}). "
                "Transitioning to remediation phase. "
                "You can now suggest code fixes."
            )
            output_type = "info"

        return self._make_observation(
            output=output,
            output_type=output_type,
            reward=reward,
            done=False,
        )

    # ── Remediation phase ────────────────────────────────────────

    def _handle_remediation(
        self, action: IncidentAction, action_dict: dict
    ) -> IncidentObservation:
        assert self._simulator is not None
        atype = action.action_type

        if atype not in VALID_REMEDIATION_ACTIONS:
            return self._make_observation(
                output=(
                    f"Invalid remediation action: '{atype}'. "
                    f"Valid: {sorted(VALID_REMEDIATION_ACTIONS)}"
                ),
                output_type="info",
                reward=-0.05,
                done=False,
            )

        if atype == "suggest_fix":
            self._state.fixes_attempted += 1
            patch = {action.file: action.patch_code} if action.file else {}

            exec_result: ExecutionResult = self._executor.run_fix(
                original_files=self._simulator.original_files,
                patch=patch,
                test_code=self._simulator.test_code,
            )

            some_passed = exec_result.tests_passed > 0 and not exec_result.passed
            reward = compute_remediation_reward(
                action_dict,
                test_passed=exec_result.passed,
                some_passed=some_passed,
                timed_out=exec_result.timed_out,
            )
            self._state.cumulative_reward += reward

            if exec_result.passed:
                self._state.fixes_passed += 1

            self._state.queries_made.append(action_dict)

            return self._make_observation(
                output=f"Fix applied to {action.file}.",
                output_type="test_result",
                reward=reward,
                done=False,
                test_output=exec_result.stdout or exec_result.stderr,
                tests_passed=exec_result.passed,
            )

        if atype == "submit_resolution":
            reward = compute_remediation_reward(action_dict)
            if self._state.fixes_attempted == 0:
                reward = -0.2  # penalise submitting without even trying
            self._state.cumulative_reward += reward
            self._state.queries_made.append(action_dict)
            self._finalize_grading()
            return self._make_observation(
                output=(
                    f"Incident resolved. Grader score: {self._state.grader_score:.4f}"
                ),
                output_type="info",
                reward=reward,
                done=True,
            )

        return self._make_observation(output="", output_type="info", reward=0.0, done=False)

    # ── Helpers ──────────────────────────────────────────────────

    def _finalize_grading(self) -> None:
        """Run the deterministic grader and store the score."""
        assert self._simulator is not None
        max_steps = (
            self._task_def["max_investigation_steps"]
            + self._task_def["max_remediation_steps"]
        )
        inv_steps = sum(
            1 for q in self._state.queries_made
            if q.get("action_type") in VALID_INVESTIGATION_ACTIONS
        )
        self._state.grader_score = grade_episode(
            root_cause_submitted=self._state.root_cause_submitted,
            root_cause_keywords=self._simulator.root_cause_keywords,
            queries_made=self._state.queries_made,
            buggy_file=self._simulator.buggy_file,
            fixes_attempted=self._state.fixes_attempted,
            fixes_passed=self._state.fixes_passed,
            max_investigation_steps=self._task_def["max_investigation_steps"],
            investigation_steps_used=inv_steps,
            total_steps_used=self._state.step_count,
            max_total_steps=max_steps,
        )

    def _make_observation(
        self,
        output: str,
        output_type: str,
        reward: float,
        done: bool,
        test_output: str = "",
        tests_passed: bool | None = None,
    ) -> IncidentObservation:
        assert self._simulator is not None
        alert = self._scenario.get("alert", {})
        max_steps = (
            self._task_def["max_investigation_steps"]
            + self._task_def["max_remediation_steps"]
        )
        return IncidentObservation(
            done=done,
            reward=reward,
            alert_title=alert.get("title", ""),
            alert_description=alert.get("description", ""),
            severity_level=alert.get("severity", ""),
            affected_service=alert.get("affected_service", ""),
            output=output,
            output_type=output_type,
            available_services=self._simulator.available_services,
            available_files=self._simulator.available_files,
            available_metrics=self._simulator.available_metrics,
            available_commands=self._simulator.available_commands,
            phase=self._state.phase,
            step_number=self._state.step_count,
            max_steps=max_steps,
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            test_output=test_output,
            tests_passed=tests_passed,
        )
