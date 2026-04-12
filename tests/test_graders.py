"""Tests for the deterministic graders."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.tasks import grade_episode, load_scenarios, TASK_DEFINITIONS


class TestGrader:
    def test_perfect_agent_scores_high(self):
        """Agent that finds everything and fixes it should score > 0.8."""
        score = grade_episode(
            issue_submitted="off-by-one range index 0 skip first 1",
            bug_keywords=["range", "index", "0", "off-by-one", "first", "skip", "1"],
            execution_history=[
                {"action_type": "run_tests"},
                {"action_type": "run_code", "code": "print(test)"},
                {"action_type": "create_issue"},
                {"action_type": "suggest_fix", "all_passed": True},
            ],
            fixes_attempted=1,
            fixes_passed=1,
            total_steps_used=5,
            max_total_steps=15,
        )
        assert score >= 0.7, f"Perfect agent scored {score}"

    def test_no_action_scores_zero(self):
        """Agent that does nothing should score ~0."""
        score = grade_episode(
            issue_submitted="",
            bug_keywords=["range", "index"],
            execution_history=[],
            fixes_attempted=0,
            fixes_passed=0,
            total_steps_used=0,
            max_total_steps=15,
        )
        assert score <= 0.15, f"No-action agent scored {score}"

    def test_partial_agent_scores_between(self):
        """Agent that finds bug but doesn't fix should score moderate."""
        score = grade_episode(
            issue_submitted="off-by-one range issue",
            bug_keywords=["range", "index", "0", "off-by-one", "first", "skip"],
            execution_history=[
                {"action_type": "run_tests"},
                {"action_type": "run_code"},
            ],
            fixes_attempted=0,
            fixes_passed=0,
            total_steps_used=3,
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
                    issue_submitted=gt["bug_description"],
                    bug_keywords=gt["bug_keywords"],
                    execution_history=[
                        {"action_type": "run_tests"},
                        {"action_type": "suggest_fix"},
                    ],
                    fixes_attempted=1,
                    fixes_passed=1,
                    total_steps_used=5,
                    max_total_steps=15,
                )
                assert 0.0 <= score <= 1.0, f"{s['id']}: {score}"
