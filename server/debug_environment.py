"""
RuntimeTerror — Core Debugging Environment.

Single-phase iterative debugging loop:
  Agent receives buggy code + visible tests, runs code, identifies bug,
  submits fix, and validates via test execution.

Implements the OpenEnv Environment interface: reset(), step(), state.
"""

from __future__ import annotations

import uuid
from typing import Any

from models import DebugAction, DebugObservation, DebugState

from .executor import CodeExecutor, ExecutionResult
from .rewards import (
    compute_run_code_reward,
    compute_run_tests_reward,
    compute_create_issue_reward,
    compute_suggest_fix_reward,
    compute_request_changes_reward,
    truncation_penalty,
)
from .code_simulator import CodeSimulator
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

VALID_ACTIONS = {
    "run_code",
    "run_tests",
    "create_issue",
    "suggest_fix",
    "request_changes",
}


class DebugEnvironment(Environment):
    """Interactive code debugging environment."""

    def __init__(self) -> None:
        super().__init__()
        self._state = DebugState()
        self._simulator: CodeSimulator | None = None
        self._executor = CodeExecutor()
        self._scenario: dict[str, Any] = {}
        self._task_def: dict[str, Any] = {}
        # Track the current source code (may be patched by suggest_fix)
        self._current_code: str = ""

    # ── OpenEnv API ──────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "easy_debug",
        **kwargs: Any,
    ) -> DebugObservation:
        """Start a new debugging episode.

        Parameters
        ----------
        task_id : str
            One of ``easy_debug``, ``medium_debug``, ``hard_debug``.
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
        self._simulator = CodeSimulator(self._scenario)
        self._current_code = self._simulator.buggy_code

        max_steps = task_def["max_steps"]

        self._state = DebugState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=task_def["difficulty"],
            code=self._current_code,
            visible_tests=self._simulator.visible_tests,
            hidden_tests=self._simulator.hidden_tests,
            execution_history=[],
            last_output="",
            issue_submitted="",
            issue_correct=False,
            issue_similarity=0.0,
            fixes_attempted=0,
            fixes_passed=0,
            cumulative_reward=0.0,
            grader_score=0.0,
        )

        return DebugObservation(
            done=False,
            reward=0.0,
            code=self._simulator.format_code(),
            visible_tests=self._simulator.visible_tests,
            execution_output="",
            test_results="",
            step_number=0,
            max_steps=max_steps,
            task_id=task_id,
            difficulty=task_def["difficulty"],
            action_feedback="Debugging session started. Examine the code and tests, then find and fix the bug.",
        )

    def step(self, action: DebugAction | dict, **kwargs: Any) -> DebugObservation:
        """Process one agent action and return the observation + reward."""
        if isinstance(action, dict):
            action = DebugAction(**action)

        if self._simulator is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        action_dict = action.model_dump()
        max_steps = self._task_def["max_steps"]

        # ── Truncation check ────────────────────────────────────
        if self._state.step_count > max_steps:
            penalty = truncation_penalty()
            self._state.cumulative_reward += penalty
            self._finalize_grading()
            return self._make_observation(
                execution_output="Episode truncated — max steps exceeded.",
                test_results="",
                reward=penalty,
                done=True,
                action_feedback="Max steps exceeded. Episode ended.",
            )

        # ── Action validation ───────────────────────────────────
        atype = action.action_type
        if atype not in VALID_ACTIONS:
            return self._make_observation(
                execution_output="",
                test_results="",
                reward=-0.05,
                done=False,
                action_feedback=(
                    f"Invalid action: '{atype}'. "
                    f"Valid: {sorted(VALID_ACTIONS)}"
                ),
            )

        # ── Action dispatch ─────────────────────────────────────
        if atype == "run_code":
            return self._handle_run_code(action, action_dict)
        elif atype == "run_tests":
            return self._handle_run_tests(action, action_dict)
        elif atype == "create_issue":
            return self._handle_create_issue(action, action_dict)
        elif atype == "suggest_fix":
            return self._handle_suggest_fix(action, action_dict)
        elif atype == "request_changes":
            return self._handle_request_changes(action, action_dict)

        return self._make_observation(
            execution_output="", test_results="",
            reward=0.0, done=False,
            action_feedback="No-op action.",
        )

    @property
    def state(self) -> DebugState:
        return self._state

    # ── Action Handlers ──────────────────────────────────────────

    def _handle_run_code(
        self, action: DebugAction, action_dict: dict
    ) -> DebugObservation:
        """Execute a code snippet in the sandbox."""
        assert self._simulator is not None

        result: ExecutionResult = self._executor.run_snippet(
            snippet=action.code,
            context_code=self._current_code,
        )

        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr

        reward = compute_run_code_reward(
            action_dict,
            output,
            self._simulator.bug_keywords,
            self._state.execution_history,
        )
        self._state.cumulative_reward += reward
        self._state.execution_history.append({**action_dict, "output": output[:500]})
        self._state.last_output = output

        return self._make_observation(
            execution_output=output,
            test_results="",
            reward=reward,
            done=False,
            action_feedback=f"Code executed. Exit code: {result.exit_code}",
        )

    def _handle_run_tests(
        self, action: DebugAction, action_dict: dict
    ) -> DebugObservation:
        """Run the visible test suite against the current code."""
        assert self._simulator is not None

        result: ExecutionResult = self._executor.run_tests(
            source_code=self._current_code,
            test_sources=self._simulator.visible_tests,
        )

        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr

        reward = compute_run_tests_reward(
            test_passed=result.passed,
            tests_run=result.tests_run,
            tests_passed=result.tests_passed,
            history=self._state.execution_history,
        )
        self._state.cumulative_reward += reward
        self._state.execution_history.append({
            **action_dict,
            "tests_run": result.tests_run,
            "tests_passed": result.tests_passed,
            "passed": result.passed,
        })
        self._state.last_output = output

        return self._make_observation(
            execution_output="",
            test_results=output,
            reward=reward,
            done=False,
            tests_passed=result.passed,
            action_feedback=f"Tests run: {result.tests_run}, passed: {result.tests_passed}",
        )

    def _handle_create_issue(
        self, action: DebugAction, action_dict: dict
    ) -> DebugObservation:
        """Submit a bug description."""
        assert self._simulator is not None

        from .tasks import _keyword_overlap

        similarity = _keyword_overlap(
            action.issue_description, self._simulator.bug_keywords
        )
        self._state.issue_submitted = action.issue_description
        self._state.issue_similarity = round(similarity, 4)
        self._state.issue_correct = similarity >= 0.5

        reward = compute_create_issue_reward(
            action.issue_description, self._simulator.bug_keywords
        )
        self._state.cumulative_reward += reward
        self._state.execution_history.append({
            **action_dict,
            "similarity": round(similarity, 4),
        })

        feedback = f"Issue submitted (confidence: {similarity:.0%}). "
        if self._state.issue_correct:
            feedback += "Bug identification looks accurate!"
        else:
            feedback += "Consider running more code to better understand the bug."

        return self._make_observation(
            execution_output="",
            test_results="",
            reward=reward,
            done=False,
            action_feedback=feedback,
        )

    def _handle_suggest_fix(
        self, action: DebugAction, action_dict: dict
    ) -> DebugObservation:
        """Submit a patched version of the source code."""
        assert self._simulator is not None

        self._state.fixes_attempted += 1

        # Run ALL tests (visible + hidden) against the patched code
        all_tests = self._simulator.all_tests
        result: ExecutionResult = self._executor.run_tests(
            source_code=action.patch_code,
            test_sources=all_tests,
        )

        some_passed = result.tests_passed > 0 and not result.passed
        reward = compute_suggest_fix_reward(
            test_passed=result.passed,
            some_passed=some_passed,
            timed_out=result.timed_out,
        )
        self._state.cumulative_reward += reward

        if result.passed:
            self._state.fixes_passed += 1
            # Update current code to the fixed version
            self._current_code = action.patch_code

        self._state.execution_history.append({
            **action_dict,
            "tests_run": result.tests_run,
            "tests_passed": result.tests_passed,
            "all_passed": result.passed,
        })

        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr

        feedback = f"Fix applied. Tests: {result.tests_passed}/{result.tests_run} passed."
        if result.passed:
            feedback += " ✅ All tests pass!"
        elif some_passed:
            feedback += " Some tests still failing."
        else:
            feedback += " ❌ Tests failed."

        return self._make_observation(
            execution_output="",
            test_results=output,
            reward=reward,
            done=False,
            tests_passed=result.passed,
            action_feedback=feedback,
        )

    def _handle_request_changes(
        self, action: DebugAction, action_dict: dict
    ) -> DebugObservation:
        """Finalize the debugging session."""
        assert self._simulator is not None

        reward = compute_request_changes_reward(
            fixes_attempted=self._state.fixes_attempted,
            fixes_passed=self._state.fixes_passed,
        )
        self._state.cumulative_reward += reward
        self._state.execution_history.append(action_dict)

        self._finalize_grading()

        return self._make_observation(
            execution_output="",
            test_results="",
            reward=reward,
            done=True,
            action_feedback=(
                f"Session complete. Grader score: {self._state.grader_score:.4f}"
            ),
        )

    # ── Helpers ──────────────────────────────────────────────────

    def _finalize_grading(self) -> None:
        """Run the deterministic grader and store the score."""
        assert self._simulator is not None
        max_steps = self._task_def["max_steps"]
        self._state.grader_score = grade_episode(
            issue_submitted=self._state.issue_submitted,
            bug_keywords=self._simulator.bug_keywords,
            execution_history=self._state.execution_history,
            fixes_attempted=self._state.fixes_attempted,
            fixes_passed=self._state.fixes_passed,
            total_steps_used=self._state.step_count,
            max_total_steps=max_steps,
        )

    def _make_observation(
        self,
        execution_output: str,
        test_results: str,
        reward: float,
        done: bool,
        action_feedback: str = "",
        tests_passed: bool | None = None,
    ) -> DebugObservation:
        assert self._simulator is not None
        max_steps = self._task_def["max_steps"]
        return DebugObservation(
            done=done,
            reward=reward,
            code=self._simulator.format_code(self._current_code),
            visible_tests=self._simulator.visible_tests,
            execution_output=execution_output,
            test_results=test_results,
            tests_passed=tests_passed,
            step_number=self._state.step_count,
            max_steps=max_steps,
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            action_feedback=action_feedback,
        )
