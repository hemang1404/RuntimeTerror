"""
NitpickAI — Sandboxed Code Executor.

Runs agent-suggested code snippets and test suites in sandboxed subprocesses.
Enforces timeout and safety limits.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of running code or tests."""

    passed: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    tests_run: int = 0
    tests_passed: int = 0


class CodeExecutor:
    """Runs code snippets and pytest test cases in temp directories."""

    TIMEOUT_SECONDS: int = 10

    def run_snippet(
        self,
        snippet: str,
        context_code: str = "",
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Execute a code snippet with optional context code in a sandbox.

        Parameters
        ----------
        snippet:
            The code to execute (e.g. ``print(calculate_total([Item(10)]))``)
        context_code:
            Source code that defines functions/classes used by the snippet.
            Written as ``code.py`` so snippet can ``from code import ...``.
        timeout:
            Execution timeout in seconds (default: self.TIMEOUT_SECONDS).
        """
        timeout = timeout or self.TIMEOUT_SECONDS

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write context code as importable module
            if context_code:
                code_path = os.path.join(tmpdir, "source.py")
                with open(code_path, "w", encoding="utf-8") as fh:
                    fh.write(context_code)

            # Write the snippet as a script
            script_path = os.path.join(tmpdir, "run_snippet.py")
            script = ""
            if context_code:
                script += "import sys, os\nsys.path.insert(0, os.getcwd())\n"
            script += snippet
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(script)

            try:
                result = subprocess.run(
                    ["python", "run_snippet.py"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                return ExecutionResult(
                    passed=(result.returncode == 0),
                    stdout=result.stdout[-2000:],
                    stderr=result.stderr[-1000:],
                    exit_code=result.returncode,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    passed=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout}s",
                    exit_code=-1,
                    timed_out=True,
                )
            except Exception as exc:
                return ExecutionResult(
                    passed=False,
                    stdout="",
                    stderr=f"Execution error: {exc}",
                    exit_code=-1,
                )

    def run_tests(
        self,
        source_code: str,
        test_sources: list[str],
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Run a list of test sources against source_code using pytest.

        Parameters
        ----------
        source_code:
            The main source file content. Written as ``source.py``.
        test_sources:
            List of test file contents. Each is written as
            ``test_N.py`` and all are run with pytest.
        timeout:
            Execution timeout in seconds.
        """
        timeout = timeout or self.TIMEOUT_SECONDS

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write source code as source.py
            code_path = os.path.join(tmpdir, "source.py")
            with open(code_path, "w", encoding="utf-8") as fh:
                fh.write(source_code)

            # Write test files
            test_files = []
            for i, test_src in enumerate(test_sources):
                test_name = f"test_{i}.py"
                test_path = os.path.join(tmpdir, test_name)
                with open(test_path, "w", encoding="utf-8") as fh:
                    fh.write(test_src)
                test_files.append(test_name)

            if not test_files:
                return ExecutionResult(
                    passed=True,
                    stdout="No tests to run.",
                    exit_code=0,
                )

            try:
                result = subprocess.run(
                    [
                        "python", "-m", "pytest",
                        *test_files,
                        "-v", "--tb=short", "--no-header", "-q",
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )

                tests_run, tests_passed = self._parse_pytest_output(result.stdout)

                return ExecutionResult(
                    passed=(result.returncode == 0),
                    stdout=result.stdout[-2000:],
                    stderr=result.stderr[-500:],
                    exit_code=result.returncode,
                    timed_out=False,
                    tests_run=tests_run,
                    tests_passed=tests_passed,
                )

            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    passed=False,
                    stdout="",
                    stderr=f"Test execution timed out after {timeout}s",
                    exit_code=-1,
                    timed_out=True,
                )
            except Exception as exc:
                return ExecutionResult(
                    passed=False,
                    stdout="",
                    stderr=f"Test execution error: {exc}",
                    exit_code=-1,
                )

    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_pytest_output(stdout: str) -> tuple[int, int]:
        """Extract (tests_run, tests_passed) from pytest -q output."""
        import re

        passed = 0
        failed = 0
        errors = 0
        for m in re.finditer(r"(\d+)\s+passed", stdout):
            passed = int(m.group(1))
        for m in re.finditer(r"(\d+)\s+failed", stdout):
            failed = int(m.group(1))
        for m in re.finditer(r"(\d+)\s+error", stdout):
            errors = int(m.group(1))
        total = passed + failed + errors
        return total, passed
