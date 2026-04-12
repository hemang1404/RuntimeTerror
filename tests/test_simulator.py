"""Tests for the CodeSimulator."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.code_simulator import CodeSimulator
from server.tasks import load_scenarios


def _make_sim(task="easy_debug", idx=0):
    return CodeSimulator(load_scenarios(task)[idx])


class TestSimulator:
    def test_buggy_code_not_empty(self):
        sim = _make_sim()
        assert len(sim.buggy_code) > 0

    def test_visible_tests_present(self):
        sim = _make_sim()
        assert len(sim.visible_tests) > 0

    def test_hidden_tests_present(self):
        sim = _make_sim()
        assert len(sim.hidden_tests) > 0

    def test_all_tests_is_union(self):
        sim = _make_sim()
        assert len(sim.all_tests) == len(sim.visible_tests) + len(sim.hidden_tests)

    def test_ground_truth_present(self):
        sim = _make_sim()
        assert len(sim.bug_description) > 0
        assert len(sim.bug_keywords) > 0
        assert len(sim.ground_truth_fix) > 0

    def test_format_code_has_line_numbers(self):
        sim = _make_sim()
        formatted = sim.format_code()
        assert "|" in formatted

    def test_format_visible_tests(self):
        sim = _make_sim()
        formatted = sim.format_visible_tests()
        assert "Test 1" in formatted

    def test_all_scenarios_loadable(self):
        for task in ["easy_debug", "medium_debug", "hard_debug"]:
            scenarios = load_scenarios(task)
            assert len(scenarios) == 5, f"{task} should have 5 scenarios"
            for s in scenarios:
                sim = CodeSimulator(s)
                assert len(sim.buggy_code) > 0
                assert len(sim.visible_tests) > 0
                assert len(sim.hidden_tests) > 0
                assert len(sim.ground_truth_fix) > 0
