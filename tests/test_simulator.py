"""Tests for the IncidentSimulator."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.simulator import IncidentSimulator
from server.tasks import load_scenarios


def _make_sim(task="easy_triage", idx=0):
    return IncidentSimulator(load_scenarios(task)[idx])


class TestSimulator:
    def test_query_logs_known_service(self):
        sim = _make_sim()
        service = sim.available_services[0]
        output = sim.query_logs(service)
        assert len(output) > 0, "Should return log entries"

    def test_query_logs_unknown_service(self):
        sim = _make_sim()
        output = sim.query_logs("nonexistent-service")
        assert "ERROR" in output

    def test_query_logs_with_keyword(self):
        sim = _make_sim()
        service = sim.available_services[0]
        output = sim.query_logs(service, keyword="error")
        # Should either find entries or return no-match message
        assert len(output) > 0

    def test_query_metrics(self):
        sim = _make_sim()
        metric = sim.available_metrics[0]
        output = sim.query_metrics(metric)
        assert "Metric:" in output

    def test_inspect_code_known_file(self):
        sim = _make_sim()
        fname = sim.available_files[0]
        output = sim.inspect_code(fname)
        assert "|" in output  # line numbers

    def test_inspect_code_unknown_file(self):
        sim = _make_sim()
        output = sim.inspect_code("does/not/exist.py")
        assert "ERROR" in output

    def test_run_diagnostic(self):
        sim = _make_sim()
        cmd = sim.available_commands[0]
        output = sim.run_diagnostic(cmd)
        assert len(output) > 0

    def test_run_diagnostic_unknown(self):
        sim = _make_sim()
        output = sim.run_diagnostic("rm -rf /")
        assert "ERROR" in output

    def test_ground_truth_accessible(self):
        sim = _make_sim()
        assert len(sim.root_cause_keywords) > 0
        assert sim.buggy_file != ""

    def test_all_scenarios_loadable(self):
        for task in ["easy_triage", "medium_triage", "hard_triage"]:
            scenarios = load_scenarios(task)
            assert len(scenarios) == 5, f"{task} should have 5 scenarios"
            for s in scenarios:
                sim = IncidentSimulator(s)
                assert len(sim.available_services) > 0
                assert len(sim.available_files) > 0
