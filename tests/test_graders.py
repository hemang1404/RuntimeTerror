"""Tests for the deterministic graders."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.tasks import grade_episode, load_scenarios, TASK_DEFINITIONS


class TestGrader:
    def test_perfect_agent_scores_high(self):
        """Agent that finds everything and fixes it should score > 0.8."""
        score = grade_episode(
            root_cause_submitted="connection leak release close pool exhausted get_user",
            root_cause_keywords=["connection", "leak", "release", "close", "pool", "exhausted", "get_user"],
            queries_made=[
                {"action_type": "inspect_code", "file": "db/pool.py"},
                {"action_type": "submit_root_cause"},
            ],
            buggy_file="db/pool.py",
            fixes_attempted=1,
            fixes_passed=1,
            max_investigation_steps=10,
            investigation_steps_used=2,
            total_steps_used=4,
            max_total_steps=15,
        )
        assert score >= 0.8, f"Perfect agent scored {score}"

    def test_no_action_scores_zero(self):
        """Agent that does nothing should score 0."""
        score = grade_episode(
            root_cause_submitted="",
            root_cause_keywords=["connection", "leak"],
            queries_made=[],
            buggy_file="db/pool.py",
            fixes_attempted=0,
            fixes_passed=0,
            max_investigation_steps=10,
            investigation_steps_used=0,
            total_steps_used=0,
            max_total_steps=15,
        )
        assert score <= 0.15, f"No-action agent scored {score}"

    def test_partial_agent_scores_between(self):
        """Agent that finds root cause but doesn't fix should score 0.3-0.6."""
        score = grade_episode(
            root_cause_submitted="connection pool exhaustion",
            root_cause_keywords=["connection", "leak", "release", "close", "pool", "exhausted"],
            queries_made=[{"action_type": "inspect_code", "file": "db/pool.py"}],
            buggy_file="db/pool.py",
            fixes_attempted=0,
            fixes_passed=0,
            max_investigation_steps=10,
            investigation_steps_used=3,
            total_steps_used=4,
            max_total_steps=15,
        )
        assert 0.1 <= score <= 0.7, f"Partial agent scored {score}"

    def test_score_always_in_range(self):
        """Grader must return a value in [0.0, 1.0]."""
        for task_id in TASK_DEFINITIONS:
            scenarios = load_scenarios(task_id)
            for s in scenarios:
                gt = s["ground_truth"]
                score = grade_episode(
                    root_cause_submitted=gt["root_cause"],
                    root_cause_keywords=gt["root_cause_keywords"],
                    queries_made=[
                        {"action_type": "inspect_code", "file": gt["buggy_file"]}
                    ],
                    buggy_file=gt["buggy_file"],
                    fixes_attempted=1,
                    fixes_passed=1,
                    max_investigation_steps=10,
                    investigation_steps_used=5,
                    total_steps_used=7,
                    max_total_steps=15,
                )
                assert 0.0 <= score <= 1.0, f"{s['id']}: {score}"
