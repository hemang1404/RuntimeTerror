"""Tests for the reward engine."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.rewards import (
    compute_investigation_reward,
    compute_remediation_reward,
    truncation_penalty,
)
from server.simulator import IncidentSimulator
from server.tasks import load_scenarios


def _make_simulator(task="easy_triage"):
    scenarios = load_scenarios(task)
    return IncidentSimulator(scenarios[0])


class TestInvestigationRewards:
    def test_relevant_log_query(self):
        sim = _make_simulator()
        action = {
            "action_type": "query_logs",
            "service": sim.available_services[0],
            "keyword": "error",
        }
        r = compute_investigation_reward(action, sim, [])
        assert r >= 0.0

    def test_unknown_service_penalty(self):
        sim = _make_simulator()
        action = {
            "action_type": "query_logs",
            "service": "nonexistent-service",
            "keyword": "",
        }
        r = compute_investigation_reward(action, sim, [])
        assert r < 0

    def test_inspect_buggy_file_bonus(self):
        sim = _make_simulator()
        action = {
            "action_type": "inspect_code",
            "file": sim.buggy_file,
        }
        r = compute_investigation_reward(action, sim, [])
        assert r == 0.1

    def test_duplicate_query_penalty(self):
        sim = _make_simulator()
        action = {
            "action_type": "query_logs",
            "service": sim.available_services[0],
        }
        r = compute_investigation_reward(action, sim, [action])
        assert r == -0.1

    def test_correct_root_cause_reward(self):
        sim = _make_simulator()
        keywords = " ".join(sim.root_cause_keywords)
        action = {
            "action_type": "submit_root_cause",
            "root_cause": keywords,
        }
        r = compute_investigation_reward(action, sim, [])
        assert r >= 0.2


class TestRemediationRewards:
    def test_fix_passed_reward(self):
        r = compute_remediation_reward(
            {"action_type": "suggest_fix"}, test_passed=True
        )
        assert r == 0.5

    def test_fix_failed_penalty(self):
        r = compute_remediation_reward(
            {"action_type": "suggest_fix"}, test_passed=False
        )
        assert r == -0.1

    def test_fix_some_passed(self):
        r = compute_remediation_reward(
            {"action_type": "suggest_fix"}, test_passed=False, some_passed=True
        )
        assert r == 0.2

    def test_truncation_penalty(self):
        assert truncation_penalty() < 0
