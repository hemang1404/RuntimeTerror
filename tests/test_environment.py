"""Tests for the core IncidentEnvironment."""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.incident_environment import IncidentEnvironment
from models import IncidentAction


class TestReset:
    def test_reset_returns_observation(self):
        env = IncidentEnvironment()
        obs = env.reset(task_id="easy_triage", seed=42)
        assert obs.done is False
        assert obs.phase == "investigation"
        assert obs.alert_title != ""
        assert len(obs.available_services) > 0
        assert len(obs.available_files) > 0

    def test_reset_different_tasks(self):
        env = IncidentEnvironment()
        for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
            obs = env.reset(task_id=task_id, seed=0)
            assert obs.task_id == task_id
            assert obs.done is False

    def test_reset_reproducible_with_seed(self):
        env = IncidentEnvironment()
        obs1 = env.reset(task_id="easy_triage", seed=123)
        state1 = env.state
        obs2 = env.reset(task_id="easy_triage", seed=123)
        state2 = env.state
        assert state1.incident_id == state2.incident_id


class TestInvestigation:
    def _make_env(self):
        env = IncidentEnvironment()
        env.reset(task_id="easy_triage", seed=42)
        return env

    def test_query_logs(self):
        env = self._make_env()
        obs = env.step(IncidentAction(
            action_type="query_logs",
            service=env.state.queries_made[0]["service"]
            if env.state.queries_made
            else env._simulator.available_services[0],
        ))
        # Should not crash, observation should have output
        assert obs.output_type == "logs"
        assert obs.done is False

    def test_inspect_code(self):
        env = self._make_env()
        files = env._simulator.available_files
        obs = env.step(IncidentAction(
            action_type="inspect_code", file=files[0]
        ))
        assert obs.output_type == "code"
        assert len(obs.output) > 0

    def test_invalid_action_type(self):
        env = self._make_env()
        obs = env.step(IncidentAction(action_type="fly_to_moon"))
        assert obs.reward < 0  # penalty

    def test_submit_root_cause_transitions_phase(self):
        env = self._make_env()
        obs = env.step(IncidentAction(
            action_type="submit_root_cause",
            root_cause="The database connections are leaking because get_user never releases the connection",
        ))
        assert env.state.phase == "remediation"
        assert obs.done is False


class TestRemediation:
    def _make_env_in_remediation(self):
        env = IncidentEnvironment()
        env.reset(task_id="easy_triage", seed=42)
        # Get the buggy file content and solution for fixing
        env.step(IncidentAction(
            action_type="submit_root_cause",
            root_cause="connection leak in get_user",
        ))
        return env

    def test_suggest_fix_runs_tests(self):
        env = self._make_env_in_remediation()
        # Submit the known-good solution
        solution = env._simulator.solution_patch
        file_name = list(solution.keys())[0]
        obs = env.step(IncidentAction(
            action_type="suggest_fix",
            file=file_name,
            patch_code=solution[file_name],
        ))
        assert obs.output_type == "test_result"
        assert obs.tests_passed is True
        assert obs.reward > 0

    def test_submit_resolution_ends_episode(self):
        env = self._make_env_in_remediation()
        obs = env.step(IncidentAction(action_type="submit_resolution"))
        assert obs.done is True

    def test_grader_score_in_range(self):
        env = self._make_env_in_remediation()
        solution = env._simulator.solution_patch
        file_name = list(solution.keys())[0]
        env.step(IncidentAction(
            action_type="suggest_fix",
            file=file_name,
            patch_code=solution[file_name],
        ))
        env.step(IncidentAction(action_type="submit_resolution"))
        assert 0.0 <= env.state.grader_score <= 1.0


class TestTruncation:
    def test_max_steps_truncates(self):
        env = IncidentEnvironment()
        env.reset(task_id="easy_triage", seed=42)
        # Step many times past the limit
        for _ in range(20):
            obs = env.step(IncidentAction(
                action_type="query_logs",
                service=env._simulator.available_services[0],
            ))
            if obs.done:
                break
        assert obs.done is True
        assert env.state.grader_score >= 0.0
