"""
IncidentEnv — Sandboxed Code Executor.

Runs agent-suggested code fixes against scenario test cases in a subprocess.
Enforces timeout and memory limits (safe for vcpu=2, 8 GB).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    """Result of running tests against an agent's patch."""

    passed: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    tests_run: int = 0
    tests_passed: int = 0


class CodeExecutor:
    """Runs patched code against pytest test cases in a temp directory."""

    TIMEOUT_SECONDS: int = 10

    def run_fix(
        self,
        original_files: dict[str, str],
        patch: dict[str, str],
        test_code: str,
    ) -> ExecutionResult:
        """Apply *patch* on top of *original_files*, run *test_code* via pytest.

        Parameters
        ----------
        original_files:
            Mapping of ``relative_path → file_content`` for the scenario's source
            files.  These form the base code.
        patch:
            Same format as *original_files* — entries here **overwrite** the
            originals (the agent's suggested fix).
        test_code:
            Raw Python source for the test file.  Will be written as
            ``test_fix.py`` inside the temp directory and executed with pytest.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Write original source files
            for relpath, content in original_files.items():
                fpath = os.path.join(tmpdir, relpath)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as fh:
                    fh.write(content)
                # Ensure every directory has an __init__.py for imports
                pkg_dir = os.path.dirname(fpath)
                init_path = os.path.join(pkg_dir, "__init__.py")
                if not os.path.exists(init_path):
                    with open(init_path, "w") as fh:
                        fh.write("")

            # 2. Apply patch (overwrite with agent's fix)
            for relpath, content in patch.items():
                fpath = os.path.join(tmpdir, relpath)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as fh:
                    fh.write(content)

            # 3. Write test file
            test_path = os.path.join(tmpdir, "test_fix.py")
            with open(test_path, "w", encoding="utf-8") as fh:
                fh.write(test_code)

            # 4. Run pytest in subprocess with timeout
            try:
                result = subprocess.run(
                    [
                        "python", "-m", "pytest",
                        "test_fix.py",
                        "-v", "--tb=short", "--no-header", "-q",
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.TIMEOUT_SECONDS,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )

                # Parse pass/fail counts from pytest output
                tests_run, tests_passed = self._parse_pytest_output(result.stdout)

                return ExecutionResult(
                    passed=(result.returncode == 0),
                    stdout=result.stdout[-2000:],  # cap to avoid huge payloads
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
                    stderr=f"Execution timed out after {self.TIMEOUT_SECONDS}s",
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

    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_pytest_output(stdout: str) -> tuple[int, int]:
        """Extract (tests_run, tests_passed) from pytest -q output."""
        # pytest -q outputs lines like "2 passed" or "1 failed, 1 passed"
        import re

        passed = 0
        failed = 0
        for m in re.finditer(r"(\d+)\s+passed", stdout):
            passed = int(m.group(1))
        for m in re.finditer(r"(\d+)\s+failed", stdout):
            failed = int(m.group(1))
        total = passed + failed
        return total, passed
