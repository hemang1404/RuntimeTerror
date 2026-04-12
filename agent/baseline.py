"""
RuntimeTerror — Baseline Agent.

A simple rule-based agent that:
1. Runs tests to observe failures
2. Runs code to probe the buggy function
3. Creates an issue describing the bug
4. Attempts a naive fix based on common patterns
5. Runs tests again to verify
6. Finalizes with request_changes

No LLM required. Serves as a scoring baseline for the benchmark.
"""

from __future__ import annotations

import re
from typing import Any


class BaselineAgent:
    """Rule-based baseline agent for the RuntimeTerror debugging environment."""

    def __init__(self) -> None:
        self._step = 0
        self._phase = "explore"  # explore → diagnose → fix → finalize
        self._test_output = ""
        self._code = ""
        self._tests = []
        self._error_info = ""
        self._fix_attempted = False
        self._tests_passed = False

    def act(self, observation: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Choose the next action based on the current observation and state."""
        self._step += 1
        code = observation.get("code", "")
        tests = observation.get("visible_tests", [])
        exec_output = observation.get("execution_output", "")
        test_results = observation.get("test_results", "")
        feedback = observation.get("action_feedback", "")

        if code:
            self._code = code
        if tests:
            self._tests = tests

        if test_results:
            self._test_output = test_results
        if exec_output:
            self._error_info += exec_output + "\n"

        # Phase 1: Run tests to see what fails
        if self._phase == "explore" and self._step == 1:
            return {"action_type": "run_tests"}

        # Phase 2: Run code to probe
        if self._phase == "explore" and self._step == 2:
            self._phase = "probe"
            snippet = self._generate_probe_snippet()
            return {"action_type": "run_code", "code": snippet}

        # Phase 3: Create issue based on observed errors
        if self._phase == "probe" or (self._phase == "explore" and self._step >= 3):
            self._phase = "diagnose"
            issue = self._generate_issue_description()
            return {"action_type": "create_issue", "issue_description": issue}

        # Phase 4: Attempt fix
        if self._phase == "diagnose":
            self._phase = "fix"
            self._fix_attempted = True
            fix = self._generate_fix()
            return {"action_type": "suggest_fix", "patch_code": fix}

        # Phase 5: Check if fix worked, try again or finalize
        if self._phase == "fix":
            tests_passed = observation.get("tests_passed", False)
            if tests_passed:
                self._tests_passed = True
                self._phase = "finalize"
                return {
                    "action_type": "request_changes",
                    "message": "Fix applied and all tests pass.",
                }
            elif not self._tests_passed and self._step <= 8:
                # Try an alternative fix
                fix = self._generate_alternative_fix()
                return {"action_type": "suggest_fix", "patch_code": fix}
            else:
                self._phase = "finalize"
                return {
                    "action_type": "request_changes",
                    "message": "Best effort fix attempted.",
                }

        # Default: finalize
        return {
            "action_type": "request_changes",
            "message": "Session complete.",
        }

    def _generate_probe_snippet(self) -> str:
        """Generate a code snippet to probe the buggy function."""
        # Extract function names from visible tests
        func_names = set()
        for test in self._tests:
            for match in re.finditer(r"from source import (.+)", test):
                imports = match.group(1)
                for name in imports.split(","):
                    name = name.strip()
                    if name:
                        func_names.add(name)

        if not func_names:
            return "from source import *\nprint(dir())"

        # Try to call the first imported function
        lines = ["from source import *"]
        for fn in list(func_names)[:2]:
            lines.append(f"try:")
            lines.append(f"    print(f'{fn}: {{repr({fn})}}')")
            lines.append(f"except Exception as e:")
            lines.append(f"    print(f'{fn} error: {{e}}')")
        return "\n".join(lines)

    def _generate_issue_description(self) -> str:
        """Generate a bug description based on observed errors."""
        parts = []

        # Parse test failures
        if self._test_output:
            if "AssertionError" in self._test_output or "FAILED" in self._test_output:
                parts.append("Test failures detected")
            if "Error" in self._test_output:
                # Extract error types
                for match in re.finditer(r"(\w+Error)", self._test_output):
                    parts.append(f"Error type: {match.group(1)}")

        # Parse execution output
        if self._error_info:
            if "None" in self._error_info:
                parts.append("function may return None unexpectedly")
            if "Error" in self._error_info:
                for match in re.finditer(r"(\w+Error): (.+?)(?:\n|$)", self._error_info):
                    parts.append(f"{match.group(1)}: {match.group(2)}")

        if not parts:
            parts.append("Bug detected in the source code based on test failures")

        return ". ".join(parts)

    def _generate_fix(self) -> str:
        """Attempt a naive fix based on common bug patterns."""
        # Strip line numbers from displayed code
        raw_code = self._strip_line_numbers(self._code)

        # Pattern: off-by-one in range(1, ...)
        if "range(1," in raw_code:
            raw_code = raw_code.replace("range(1,", "range(0,", 1)
            # More general: range(1, len -> range(len
            raw_code = re.sub(r"range\(1,\s*len\(", "range(len(", raw_code)

        # Pattern: missing return statement
        lines = raw_code.split("\n")
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            # If a line creates a variable and next line is unindented/end
            if re.match(r"\s+\w+\s*=\s*.+\(", line) and not line.strip().startswith("return"):
                stripped = line.lstrip()
                indent = line[:len(line) - len(stripped)]
                var_name = stripped.split("=")[0].strip()
                # Check if next line is end of function or less indented
                if i + 1 >= len(lines) or (
                    i + 1 < len(lines) and
                    not lines[i + 1].strip() and
                    "return" not in "\n".join(lines[max(0, i-3):i+1])
                ):
                    new_lines.append(f"{indent}return {var_name}")
        raw_code = "\n".join(new_lines)

        # Pattern: > should be >=
        if "> " in raw_code and ">=" not in raw_code:
            # Only fix comparison operators, not other uses
            raw_code = re.sub(r"(\w+)\s*>\s*(\w+)", r"\1 >= \2", raw_code, count=1)

        # Pattern: division without zero check
        if "/ len(" in raw_code and "if not" not in raw_code:
            raw_code = raw_code.replace(
                "/ len(",
                "/ len(",
            )
            # Add a guard
            lines = raw_code.split("\n")
            new_lines = []
            for line in lines:
                if "/ len(" in line and "if" not in line:
                    indent = line[:len(line) - len(line.lstrip())]
                    # Find the variable being divided
                    match = re.search(r"(\w+)\s*/\s*len\((\w+)\)", line)
                    if match:
                        var_name = match.group(2)
                        new_lines.append(f"{indent}if not {var_name}:")
                        new_lines.append(f"{indent}    return 0.0")
                new_lines.append(line)
            raw_code = "\n".join(new_lines)

        # Pattern: ' '.join with empty strings
        if "' '.join(" in raw_code:
            raw_code = re.sub(
                r"'\s'\.join\((\w+)\)",
                r"' '.join(p for p in \1 if p)",
                raw_code,
            )

        return raw_code

    def _generate_alternative_fix(self) -> str:
        """Try a different fix strategy."""
        raw_code = self._strip_line_numbers(self._code)

        # Try the ground truth approach: just add safety checks everywhere
        lines = raw_code.split("\n")
        new_lines = []
        for line in lines:
            new_lines.append(line)
            # Add None checks after dict access
            if ".get(" in line and "if" not in line:
                indent = line[:len(line) - len(line.lstrip())]
                var_match = re.match(r"\s*(\w+)\s*=.*\.get\(", line)
                if var_match:
                    var = var_match.group(1)
                    new_lines.append(f"{indent}if {var} is None:")
                    new_lines.append(f"{indent}    {var} = ''")

        return "\n".join(new_lines)

    @staticmethod
    def _strip_line_numbers(code: str) -> str:
        """Strip line number prefixes from formatted code display."""
        lines = code.split("\n")
        cleaned = []
        for line in lines:
            # Match pattern: "   1 | def foo():"
            match = re.match(r"\s*\d+\s*\|\s?(.*)", line)
            if match:
                cleaned.append(match.group(1))
            else:
                cleaned.append(line)
        return "\n".join(cleaned)


def main():
    """Run the baseline agent locally against the environment."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from server.debug_environment import DebugEnvironment

    tasks = ["easy_debug", "medium_debug", "hard_debug"]
    agent = BaselineAgent()

    for task_id in tasks:
        env = DebugEnvironment()
        obs = env.reset(task_id=task_id, seed=42)
        agent = BaselineAgent()  # fresh agent per task

        print(f"\n{'='*50}")
        print(f"Task: {task_id} | {obs.difficulty}")
        print(f"{'='*50}")

        while not obs.done:
            action = agent.act(obs.model_dump(), env.state.model_dump())
            obs = env.step(action)
            print(f"  Step {obs.step_number}: {action['action_type']:20s} -> reward={obs.reward:+.2f}")

        state = env.state
        print(f"  Score: {state.grader_score:.4f} | Cumulative: {state.cumulative_reward:.2f}")


if __name__ == "__main__":
    main()
