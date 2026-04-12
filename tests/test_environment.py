"""Tests for the core DebugEnvironment."""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.debug_environment import DebugEnvironment
from models import DebugAction


class TestReset:
    def test_reset_returns_observation(self):
        env = DebugEnvironment()
        obs = env.reset(task_id="easy_debug", seed=42)
        assert obs.done is False
        assert obs.code != ""
        assert len(obs.visible_tests) > 0
        assert obs.step_number == 0

    def test_reset_different_tasks(self):
        env = DebugEnvironment()
        for task_id in ["easy_debug", "medium_debug", "hard_debug"]:
            obs = env.reset(task_id=task_id, seed=0)
            assert obs.task_id == task_id
            assert obs.done is False

    def test_reset_reproducible_with_seed(self):
        env = DebugEnvironment()
        obs1 = env.reset(task_id="easy_debug", seed=123)
        code1 = obs1.code
        obs2 = env.reset(task_id="easy_debug", seed=123)
        code2 = obs2.code
        assert code1 == code2


class TestActions:
    def _make_env(self):
        env = DebugEnvironment()
        env.reset(task_id="easy_debug", seed=42)
        return env

    def test_run_code(self):
        env = self._make_env()
        obs = env.step(DebugAction(
            action_type="run_code",
            code="from code import *\nprint('hello')",
        ))
        assert obs.done is False
        assert "hello" in obs.execution_output

    def test_run_tests(self):
        env = self._make_env()
        obs = env.step(DebugAction(action_type="run_tests"))
        assert obs.done is False
        # Should have some test output (pass or fail)
        assert obs.test_results != "" or obs.action_feedback != ""

    def test_create_issue(self):
        env = self._make_env()
        obs = env.step(DebugAction(
            action_type="create_issue",
            issue_description="There is a bug in the range() call",
        ))
        assert obs.done is False
        assert env.state.issue_submitted != ""

    def test_suggest_fix(self):
        env = self._make_env()
        # Get the ground truth fix
        fix = env._simulator.ground_truth_fix
        obs = env.step(DebugAction(
            action_type="suggest_fix",
            patch_code=fix,
        ))
        assert obs.done is False
        assert obs.tests_passed is True
        assert obs.reward > 0

    def test_request_changes_ends_episode(self):
        env = self._make_env()
        obs = env.step(DebugAction(
            action_type="request_changes",
            message="Done",
        ))
        assert obs.done is True

    def test_invalid_action_type(self):
        env = self._make_env()
        obs = env.step(DebugAction(action_type="fly_to_moon"))
        assert obs.reward < 0  # penalty


class TestFixAndFinalize:
    def test_fix_then_finalize(self):
        env = DebugEnvironment()
        env.reset(task_id="easy_debug", seed=42)
        fix = env._simulator.ground_truth_fix
        env.step(DebugAction(
            action_type="suggest_fix",
            patch_code=fix,
        ))
        obs = env.step(DebugAction(
            action_type="request_changes",
            message="Fixed",
        ))
        assert obs.done is True
        assert env.state.grader_score > 0

    def test_grader_score_in_range(self):
        env = DebugEnvironment()
        env.reset(task_id="easy_debug", seed=42)
        fix = env._simulator.ground_truth_fix
        env.step(DebugAction(
            action_type="create_issue",
            issue_description="off by one range index skip first",
        ))
        env.step(DebugAction(
            action_type="suggest_fix",
            patch_code=fix,
        ))
        env.step(DebugAction(
            action_type="request_changes",
            message="Fixed",
        ))
        assert 0.0 <= env.state.grader_score <= 1.0


class TestTruncation:
    def test_max_steps_truncates(self):
        env = DebugEnvironment()
        env.reset(task_id="easy_debug", seed=42)
        for _ in range(20):
            obs = env.step(DebugAction(
                action_type="run_code",
                code="print('probe')",
            ))
            if obs.done:
                break
        assert obs.done is True
        assert env.state.grader_score >= 0.0
