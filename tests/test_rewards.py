"""Tests for the reward engine."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.rewards import (
    compute_run_code_reward,
    compute_run_tests_reward,
    compute_create_issue_reward,
    compute_suggest_fix_reward,
    compute_request_changes_reward,
    truncation_penalty,
)


class TestRunCodeRewards:
    def test_useful_output_rewarded(self):
        r = compute_run_code_reward(
            {"action_type": "run_code", "code": "print(test)"},
            "Error: ZeroDivisionError",
            ["zero", "division"],
            [],
        )
        assert r >= 0.0

    def test_duplicate_penalized(self):
        prev = {"action_type": "run_code", "code": "print(1)"}
        r = compute_run_code_reward(
            {"action_type": "run_code", "code": "print(1)"},
            "1",
            [],
            [prev],
        )
        assert r == -0.1


class TestRunTestsRewards:
    def test_failure_is_useful(self):
        r = compute_run_tests_reward(
            test_passed=False,
            tests_run=3,
            tests_passed=1,
            history=[],
        )
        assert r >= 0.0

    def test_too_many_test_runs(self):
        history = [{"action_type": "run_tests"}] * 4
        r = compute_run_tests_reward(
            test_passed=False,
            tests_run=2,
            tests_passed=0,
            history=history,
        )
        assert r < 0


class TestCreateIssueRewards:
    def test_correct_issue_rewarded(self):
        r = compute_create_issue_reward(
            "off-by-one range index skip first",
            ["range", "index", "0", "off-by-one", "first", "skip"],
        )
        assert r >= 0.15

    def test_wrong_issue_penalized(self):
        r = compute_create_issue_reward(
            "the code is broken somehow",
            ["range", "index", "0", "off-by-one"],
        )
        assert r < 0


class TestSuggestFixRewards:
    def test_fix_passed_reward(self):
        r = compute_suggest_fix_reward(test_passed=True)
        assert r == 0.5

    def test_fix_failed_penalty(self):
        r = compute_suggest_fix_reward(test_passed=False)
        assert r == -0.3

    def test_fix_some_passed(self):
        r = compute_suggest_fix_reward(test_passed=False, some_passed=True)
        assert r == 0.2


class TestRequestChangesRewards:
    def test_with_passed_fix(self):
        r = compute_request_changes_reward(fixes_attempted=1, fixes_passed=1)
        assert r == 1.0

    def test_without_fix_attempt(self):
        r = compute_request_changes_reward(fixes_attempted=0, fixes_passed=0)
        assert r == -0.3


class TestTruncation:
    def test_truncation_penalty(self):
        assert truncation_penalty() < 0
