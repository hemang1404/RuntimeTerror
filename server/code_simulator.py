"""
RuntimeTerror — Code Simulator.

Serves buggy code, test suites, and ground-truth data for a loaded debugging
task scenario.  The agent queries this to inspect code and understand bugs.
"""

from __future__ import annotations

from typing import Any


class CodeSimulator:
    """Serves debugging data for one task scenario."""

    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self._ground_truth: dict[str, Any] = scenario.get("ground_truth", {})

    # ── Source code ────────────────────────────────────────────────

    @property
    def buggy_code(self) -> str:
        return self.scenario.get("buggy_code", "")

    @property
    def visible_tests(self) -> list[str]:
        return self.scenario.get("visible_tests", [])

    @property
    def hidden_tests(self) -> list[str]:
        return self.scenario.get("hidden_tests", [])

    @property
    def all_tests(self) -> list[str]:
        """All tests (visible + hidden), used for final grading."""
        return self.visible_tests + self.hidden_tests

    # ── Ground-truth helpers (used by grader, not exposed to agent) ─

    @property
    def bug_description(self) -> str:
        return self._ground_truth.get("bug_description", "")

    @property
    def bug_keywords(self) -> list[str]:
        return self._ground_truth.get("bug_keywords", [])

    @property
    def ground_truth_fix(self) -> str:
        return self._ground_truth.get("fix", "")

    # ── Display helpers ───────────────────────────────────────────

    def format_code(self, code: str | None = None) -> str:
        """Return code with line numbers for display."""
        source = code or self.buggy_code
        numbered = []
        for i, line in enumerate(source.split("\n"), 1):
            numbered.append(f"{i:>4} | {line}")
        return "\n".join(numbered)

    def format_visible_tests(self) -> str:
        """Return visible tests formatted for display."""
        parts = []
        for i, test in enumerate(self.visible_tests, 1):
            parts.append(f"--- Test {i} ---\n{test}")
        return "\n".join(parts)
