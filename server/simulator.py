"""
IncidentEnv — Incident Simulator.

Serves pre-computed log, metric, config, and source-file data for a loaded
incident scenario.  The agent queries this to investigate.
"""

from __future__ import annotations

import json
from typing import Any


class IncidentSimulator:
    """Serves diagnostic data for one incident scenario."""

    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self._logs: dict[str, list[dict]] = scenario.get("logs", {})
        self._metrics: dict[str, dict] = scenario.get("metrics", {})
        self._files: dict[str, str] = scenario.get("source_files", {})
        self._configs: dict[str, str] = scenario.get("configs", {})
        self._ground_truth: dict[str, Any] = scenario.get("ground_truth", {})

    # ── Query helpers ──────────────────────────────────────────────

    def query_logs(self, service: str, keyword: str = "") -> str:
        """Return log entries for *service*, optionally filtered by *keyword*."""
        entries = self._logs.get(service)
        if entries is None:
            return f"[ERROR] Unknown service: '{service}'. Available: {self.available_services}"

        if keyword:
            entries = [
                e for e in entries
                if keyword.lower() in e.get("msg", "").lower()
                or keyword.lower() in e.get("level", "").lower()
            ]

        if not entries:
            return f"[INFO] No log entries matching keyword='{keyword}' for service='{service}'."

        lines: list[str] = []
        for e in entries:
            lines.append(f"[{e.get('ts', '')}] {e.get('level', 'INFO'):5s} | {e.get('msg', '')}")
        return "\n".join(lines)

    def query_metrics(self, metric: str, time_range: str = "5m") -> str:
        """Return formatted time-series values for *metric*."""
        m = self._metrics.get(metric)
        if m is None:
            return f"[ERROR] Unknown metric: '{metric}'. Available: {self.available_metrics}"

        unit = m.get("unit", "")
        values = m.get("values", [])

        # Simulate time-range trimming (just slice the tail)
        slices = {"1m": 2, "5m": 5, "15m": 8, "1h": 10}
        n = slices.get(time_range, 5)
        values = values[-n:]

        lines = [f"Metric: {metric} ({unit})  |  Range: last {time_range}"]
        for i, v in enumerate(values):
            bar = "█" * min(int(v / max(max(m.get("values", [1])), 1) * 20), 20)
            lines.append(f"  t-{len(values)-1-i:>2}: {v:>10} {unit}  {bar}")
        return "\n".join(lines)

    def inspect_code(self, filename: str) -> str:
        """Return source file contents."""
        content = self._files.get(filename)
        if content is None:
            return f"[ERROR] File not found: '{filename}'. Available: {self.available_files}"
        # Add line numbers
        numbered = []
        for i, line in enumerate(content.split("\n"), 1):
            numbered.append(f"{i:>4} | {line}")
        return "\n".join(numbered)

    def run_diagnostic(self, command: str) -> str:
        """Return pre-computed output for a supported diagnostic command."""
        output = self._configs.get(command)
        if output is None:
            available = list(self._configs.keys())
            return (
                f"[ERROR] Unknown command: '{command}'.\n"
                f"Available commands: {available}"
            )
        return output

    # ── Discovery properties ──────────────────────────────────────

    @property
    def available_services(self) -> list[str]:
        return list(self._logs.keys())

    @property
    def available_files(self) -> list[str]:
        return list(self._files.keys())

    @property
    def available_metrics(self) -> list[str]:
        return list(self._metrics.keys())

    @property
    def available_commands(self) -> list[str]:
        return list(self._configs.keys())

    # ── Ground-truth helpers (used by grader, not exposed to agent) ─

    @property
    def root_cause_keywords(self) -> list[str]:
        return self._ground_truth.get("root_cause_keywords", [])

    @property
    def buggy_file(self) -> str:
        return self._ground_truth.get("buggy_file", "")

    @property
    def ground_truth_root_cause(self) -> str:
        return self._ground_truth.get("root_cause", "")

    @property
    def test_code(self) -> str:
        return self.scenario.get("test_code", "")

    @property
    def solution_patch(self) -> dict[str, str]:
        return self.scenario.get("solution_patch", {})

    @property
    def original_files(self) -> dict[str, str]:
        return dict(self._files)
