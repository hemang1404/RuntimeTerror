"""
NitpickAI -- PR Debugging Environment.

Wraps a real GitHub pull request as a debugging environment.
The agent can read the PR diff, run the tests, investigate the code,
and suggest fixes -- all against real repository code.

Uses the same action space as DebugEnvironment but operates on
actual PR data instead of synthetic scenarios.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from typing import Any

from models import DebugAction, DebugObservation, DebugState

from .executor import CodeExecutor
from .github_fetcher import PRData, fetch_pr


# ── Constants ────────────────────────────────────────────────────

MAX_STEPS = 20

VALID_ACTIONS = {
    "run_code",
    "run_tests",
    "create_issue",
    "suggest_fix",
    "request_changes",
}


class PREnvironment:
    """Debugging environment backed by a real GitHub pull request.

    The PR's changed source files become the "buggy code" and the
    repository's test files become the test suite.
    """

    def __init__(self) -> None:
        self._state = DebugState()
        self._executor = CodeExecutor()
        self._pr: PRData | None = None
        self._workspace: str = ""
        self._current_source: dict[str, str] = {}  # filename -> content
        self._test_sources: dict[str, str] = {}
        self._base_test_results: str = ""  # test output before any changes
        self._deps_installed: bool = False  # cache: only install once
        self._deps_workspace: str = ""     # reusable workspace with deps

    def reset(self, pr_url: str, **kwargs: Any) -> DebugObservation:
        """Start a debugging session from a GitHub PR URL.

        Parameters
        ----------
        pr_url : str
            Full or shorthand GitHub PR URL.
        """
        # Fetch PR data
        self._pr = fetch_pr(pr_url)

        if not self._pr.source_files and not self._pr.test_files:
            raise ValueError(
                f"PR #{self._pr.pr_number} has no Python source files changed. "
                "NitpickAI only works with Python repositories."
            )

        # Set up workspace
        self._workspace = tempfile.mkdtemp(prefix="nitpick_pr_")
        self._current_source = dict(self._pr.source_files)

        # Combine test files: those changed in PR + those from repo
        # Limit to 3 test files max to keep execution fast
        self._test_sources = {}
        self._test_sources.update(self._pr.repo_test_files)
        self._test_sources.update(self._pr.test_files)  # PR tests override
        # Keep only the 3 most relevant test files
        if len(self._test_sources) > 3:
            items = list(self._test_sources.items())[:3]
            self._test_sources = dict(items)

        # Build combined code display
        code_display = self._format_source_code()
        diff_display = self._format_diff()
        test_list = list(self._test_sources.values())

        # Initialize state
        self._state = DebugState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=f"pr_{self._pr.owner}/{self._pr.repo}#{self._pr.pr_number}",
            difficulty="real",
            code=code_display,
            visible_tests=test_list[:5],  # Show up to 5 test files
            hidden_tests=test_list[5:],
            execution_history=[],
            last_output="",
            issue_submitted="",
            issue_correct=False,
            issue_similarity=0.0,
            fixes_attempted=0,
            fixes_passed=0,
            cumulative_reward=0.0,
            grader_score=0.0,
        )

        feedback_parts = [
            f"PR #{self._pr.pr_number}: {self._pr.title}",
            f"Repository: {self._pr.owner}/{self._pr.repo}",
            f"Branch: {self._pr.head_branch} -> {self._pr.base_branch}",
            f"Changed files: {len(self._pr.changed_files)}",
            f"Test files found: {len(self._test_sources)}",
        ]

        return DebugObservation(
            done=False,
            reward=0.0,
            code=code_display,
            visible_tests=test_list[:5],
            execution_output=diff_display,  # Show the diff as initial context
            test_results="",
            step_number=0,
            max_steps=MAX_STEPS,
            task_id=self._state.task_id,
            difficulty="real",
            action_feedback="\n".join(feedback_parts),
        )

    def step(self, action: DebugAction | dict, **kwargs: Any) -> DebugObservation:
        """Process one agent action."""
        if isinstance(action, dict):
            action = DebugAction(**action)

        if self._pr is None:
            raise RuntimeError("Call reset() with a PR URL before step().")

        self._state.step_count += 1

        # Truncation
        if self._state.step_count > MAX_STEPS:
            self._state.cumulative_reward -= 0.5
            return self._make_obs(
                execution_output="Episode truncated.",
                test_results="",
                reward=-0.5,
                done=True,
                action_feedback="Max steps exceeded.",
            )

        atype = action.action_type
        if atype not in VALID_ACTIONS:
            return self._make_obs(
                execution_output="",
                test_results="",
                reward=-0.05,
                done=False,
                action_feedback=f"Invalid action: '{atype}'",
            )

        if atype == "run_code":
            return self._handle_run_code(action)
        elif atype == "run_tests":
            return self._handle_run_tests()
        elif atype == "create_issue":
            return self._handle_create_issue(action)
        elif atype == "suggest_fix":
            return self._handle_suggest_fix(action)
        elif atype == "request_changes":
            return self._handle_request_changes(action)

        return self._make_obs("", "", 0.0, False, "Unknown action.")

    @property
    def state(self) -> DebugState:
        return self._state

    # ── Action Handlers ──────────────────────────────────────────

    def _handle_run_code(self, action: DebugAction) -> DebugObservation:
        """Run a code snippet with all source files available."""
        # Write all current source files to a temp dir and run
        combined_source = "\n\n".join(
            f"# === {fname} ===\n{content}"
            for fname, content in self._current_source.items()
        )

        result = self._executor.run_snippet(
            snippet=action.code,
            context_code=combined_source,
        )

        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr

        reward = 0.05 if output.strip() else 0.0
        self._state.cumulative_reward += reward
        self._state.execution_history.append({
            "action_type": "run_code",
            "code": action.code[:200],
            "output": output[:500],
        })

        return self._make_obs(
            execution_output=output,
            test_results="",
            reward=reward,
            done=False,
            action_feedback=f"Code executed. Exit code: {result.exit_code}",
        )
    def _handle_run_tests(self) -> DebugObservation:
        """Run the test suite against the current source code."""
        # Reuse a persistent workspace to avoid re-installing deps every time
        if not self._deps_workspace:
            self._deps_workspace = tempfile.mkdtemp(prefix="nitpick_ws_")
        workspace = self._deps_workspace

        try:
            # Write source files
            for fname, content in self._current_source.items():
                fpath = os.path.join(workspace, os.path.basename(fname))
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)

            # Also write as source.py for simple import
            if self._current_source:
                combined = "\n\n".join(self._current_source.values())
                with open(os.path.join(workspace, "source.py"), "w", encoding="utf-8") as f:
                    f.write(combined)

            # Write test files
            for fname, content in self._test_sources.items():
                test_path = os.path.join(workspace, f"test_{os.path.basename(fname)}")
                if not test_path.endswith(".py"):
                    test_path += ".py"
                # Avoid duplicate names
                if os.path.exists(test_path):
                    test_path = os.path.join(workspace, f"t_{hash(fname) % 10000}_{os.path.basename(fname)}")
                with open(test_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # (Pip install skipped to enforce zero-dependency standard library isolation)

            # Run pytest
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", workspace, "-v", "--tb=short", "--no-header"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=workspace,
                    env={**os.environ, "PYTHONPATH": workspace},
                )
                output = proc.stdout
                if proc.stderr:
                    output += "\n" + proc.stderr
                passed = proc.returncode == 0
            except subprocess.TimeoutExpired:
                output = "Tests timed out after 60 seconds."
                passed = False

        finally:
            # Cleanup handled by OS temp cleanup
            pass

        reward = 0.05 if not passed else 0.1  # Higher reward if tests pass
        self._state.cumulative_reward += reward
        self._state.execution_history.append({
            "action_type": "run_tests",
            "passed": passed,
        })

        return self._make_obs(
            execution_output="",
            test_results=output[:3000],  # Truncate for sanity
            reward=reward,
            done=False,
            tests_passed=passed,
            action_feedback=f"Tests {'PASSED' if passed else 'FAILED'}",
        )

    def _handle_create_issue(self, action: DebugAction) -> DebugObservation:
        """Submit a bug description."""
        self._state.issue_submitted = action.issue_description
        # For real PRs we can't automatically grade the issue accuracy
        # Give a small positive reward for any substantive description
        reward = 0.15 if len(action.issue_description) > 20 else 0.0
        self._state.cumulative_reward += reward
        self._state.execution_history.append({
            "action_type": "create_issue",
            "description": action.issue_description[:200],
        })

        return self._make_obs(
            execution_output="",
            test_results="",
            reward=reward,
            done=False,
            action_feedback=f"Issue submitted: {action.issue_description[:100]}",
        )

    def _handle_suggest_fix(self, action: DebugAction) -> DebugObservation:
        """Submit a fix and validate against the test suite."""
        self._state.fixes_attempted += 1

        # Determine which file to patch
        # If patch_code looks like a complete file, replace the first source file
        if self._current_source:
            first_file = list(self._current_source.keys())[0]
            old_source = dict(self._current_source)
            self._current_source[first_file] = action.patch_code

        # Run tests with the patched code
        workspace = tempfile.mkdtemp(prefix="nitpick_fix_")
        try:
            for fname, content in self._current_source.items():
                fpath = os.path.join(workspace, os.path.basename(fname))
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)

            combined = "\n\n".join(self._current_source.values())
            with open(os.path.join(workspace, "source.py"), "w", encoding="utf-8") as f:
                f.write(combined)

            for fname, content in self._test_sources.items():
                test_path = os.path.join(workspace, f"test_{os.path.basename(fname)}")
                if not test_path.endswith(".py"):
                    test_path += ".py"
                with open(test_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # (Pip install skipped to enforce zero-dependency standard library isolation)

            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", workspace, "-v", "--tb=short", "--no-header"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=workspace,
                    env={**os.environ, "PYTHONPATH": workspace},
                )
                output = proc.stdout
                if proc.stderr:
                    output += "\n" + proc.stderr
                passed = proc.returncode == 0
            except subprocess.TimeoutExpired:
                output = "Tests timed out."
                passed = False

        finally:
            pass

        if passed:
            self._state.fixes_passed += 1
            reward = 0.5
        else:
            # Revert if tests failed
            self._current_source = old_source
            reward = -0.2

        self._state.cumulative_reward += reward
        self._state.execution_history.append({
            "action_type": "suggest_fix",
            "passed": passed,
        })

        feedback = f"Fix applied. Tests {'PASSED' if passed else 'FAILED'}."
        return self._make_obs(
            execution_output="",
            test_results=output[:3000],
            reward=reward,
            done=False,
            tests_passed=passed,
            action_feedback=feedback,
        )

    def _handle_request_changes(self, action: DebugAction) -> DebugObservation:
        """Finalize the session."""
        if self._state.fixes_passed > 0:
            reward = 1.0
        elif self._state.fixes_attempted > 0:
            reward = 0.0
        else:
            reward = -0.3

        self._state.cumulative_reward += reward

        # Calculate a simple grader score for real PRs
        score = 0.0
        if self._state.issue_submitted:
            score += 0.25
        if self._state.fixes_attempted > 0:
            score += 0.25
        if self._state.fixes_passed > 0:
            score += 0.50
        self._state.grader_score = score

        return self._make_obs(
            execution_output="",
            test_results="",
            reward=reward,
            done=True,
            action_feedback=f"Session complete. Score: {score:.2f}",
        )

    # ── Helpers ──────────────────────────────────────────────────

    def _make_obs(
        self,
        execution_output: str,
        test_results: str,
        reward: float,
        done: bool,
        action_feedback: str = "",
        tests_passed: bool | None = None,
    ) -> DebugObservation:
        """Build a DebugObservation from the current state."""
        code_display = self._format_source_code()
        test_list = list(self._test_sources.values())
        return DebugObservation(
            done=done,
            reward=reward,
            code=code_display,
            visible_tests=test_list[:5],
            execution_output=execution_output,
            test_results=test_results,
            tests_passed=tests_passed,
            step_number=self._state.step_count,
            max_steps=MAX_STEPS,
            task_id=self._state.task_id,
            difficulty="real",
            action_feedback=action_feedback,
        )

    def _format_source_code(self) -> str:
        """Format all changed source files with headers and line numbers."""
        parts = []
        for fname, content in self._current_source.items():
            lines = content.split("\n")
            numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
            parts.append(f"# === {fname} ===\n" + "\n".join(numbered))
        return "\n\n".join(parts)

    def _format_diff(self) -> str:
        """Format the PR diff for display."""
        parts = [f"PR #{self._pr.pr_number} Diff:\n"]
        for f in self._pr.changed_files:
            parts.append(f"--- {f.filename} ({f.status}, +{f.additions}/-{f.deletions})")
            if f.patch:
                parts.append(f.patch[:1000])  # Truncate large diffs
            parts.append("")
        return "\n".join(parts)




# ── Convenience function for Gradio ──────────────────────────────

def analyze_pr(pr_url: str) -> dict[str, Any]:
    """Quick analysis of a PR without running the full environment loop.

    Returns a dict with summary information for display in the UI.
    """
    pr = fetch_pr(pr_url)

    summary = {
        "title": pr.title,
        "repo": f"{pr.owner}/{pr.repo}",
        "pr_number": pr.pr_number,
        "state": pr.state,
        "branch": f"{pr.head_branch} -> {pr.base_branch}",
        "changed_files": [
            {"name": f.filename, "status": f.status, "additions": f.additions, "deletions": f.deletions}
            for f in pr.changed_files
        ],
        "source_files": list(pr.source_files.keys()),
        "test_files": list(pr.test_files.keys()) + list(pr.repo_test_files.keys()),
        "has_requirements": bool(pr.requirements),
        "diff": "\n".join(
            f"{f.filename} ({f.status})\n{f.patch[:500]}"
            for f in pr.changed_files
            if f.patch
        ),
    }
    return summary
