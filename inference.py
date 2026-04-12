#!/usr/bin/env python3
"""
RuntimeTerror — Submission Inference Script.

MANDATORY ENV VARS
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT
    [START] task=<task_name> env=runtime_terror model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, List, Optional

from openai import OpenAI
import requests

# ── Environment Configuration ────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "missing-token"))
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

BENCHMARK = "runtime_terror"
TASKS = ["easy_debug", "medium_debug", "hard_debug"]
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.3"))

# ── Structured Logging ───────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ── Environment Client (HTTP) ───────────────────────────────────


class EnvClient:
    """Minimal HTTP client for the RuntimeTerror environment server."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None
        self._http = requests.Session()

    def reset(self, task_id: str, seed: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        resp = self._http.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data.get("session_id")
        return data.get("observation", data)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        assert self.session_id, "Call reset() first"
        resp = self._http.post(
            f"{self.base_url}/step/{self.session_id}", json=action, timeout=60
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict[str, Any]:
        assert self.session_id, "Call reset() first"
        resp = self._http.get(
            f"{self.base_url}/state/{self.session_id}", timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._http.close()


# ── LLM Agent ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Python debugging agent. You are given buggy Python code and test cases. Your goal is to find and fix the bug.

You have these actions available. Respond with a JSON object for each action:

1. **run_code** — Execute a code snippet to investigate the bug.
   {"action_type": "run_code", "code": "<python code to run>"}
   NOTE: The source module is called 'source'. Use 'from source import *' to access the buggy code's functions.

2. **run_tests** — Run the visible test suite against the current code.
   {"action_type": "run_tests"}

3. **create_issue** — Describe the bug you've identified.
   {"action_type": "create_issue", "issue_description": "<description of the bug>"}

4. **suggest_fix** — Submit the complete fixed source code.
   {"action_type": "suggest_fix", "patch_code": "<complete fixed python source code>"}

5. **request_changes** — Finalize and end the session.
   {"action_type": "request_changes", "message": "<summary>"}

## Strategy
Follow this debugging workflow:
1. First, run_tests to see what's failing
2. Then, run_code with targeted snippets to understand the bug
3. create_issue with a clear description of the root cause
4. suggest_fix with the complete corrected source code (not just the changed line — the ENTIRE fixed file)
5. request_changes to finalize

## Rules
- Respond with ONLY a valid JSON object. No markdown, no explanation, no code blocks.
- The patch_code in suggest_fix must be the COMPLETE source file, not a diff.
- Be precise in your issue descriptions — mention the specific function, line, and error type.
- You have limited steps, so be efficient."""


def parse_action(raw_response: str) -> dict[str, Any]:
    """Extract a JSON action from the LLM response."""
    text = raw_response.strip()

    # Strip markdown code fences if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Find outermost { ... }
    brace_start = text.find("{")
    if brace_start == -1:
        return {"action_type": "run_tests"}

    depth = 0
    brace_end = -1
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    if brace_end == -1:
        return {"action_type": "run_tests"}

    json_str = text[brace_start:brace_end]

    try:
        action = json.loads(json_str)
        if "action_type" not in action:
            return {"action_type": "run_tests"}
        return action
    except json.JSONDecodeError:
        json_str = json_str.replace("'", '"')
        try:
            action = json.loads(json_str)
            if "action_type" not in action:
                return {"action_type": "run_tests"}
            return action
        except json.JSONDecodeError:
            return {"action_type": "run_tests"}


def format_action_for_log(action: dict[str, Any]) -> str:
    """Format an action dict as a compact string for [STEP] logging."""
    atype = action.get("action_type", "unknown")
    if atype == "run_code":
        code = action.get("code", "")[:60].replace("\n", "\\n")
        return f"run_code('{code}')"
    elif atype == "run_tests":
        return "run_tests()"
    elif atype == "create_issue":
        desc = action.get("issue_description", "")[:60]
        return f"create_issue('{desc}')"
    elif atype == "suggest_fix":
        n = len(action.get("patch_code", ""))
        return f"suggest_fix({n}_chars)"
    elif atype == "request_changes":
        msg = action.get("message", "")[:40]
        return f"request_changes('{msg}')"
    return json.dumps(action, separators=(",", ":"))


def build_user_message(obs: dict[str, Any], step: int, has_issued: bool, has_fixed: bool) -> str:
    """Build the user prompt from an observation."""
    parts = []
    max_steps = obs.get("max_steps", 20)
    parts.append(f"[Step {step}/{max_steps}]")

    # Show code on first step
    code = obs.get("code", "")
    if step == 1 and code:
        max_code = 3000
        code_display = code[:max_code] + (f"\n... [truncated {len(code)-max_code} chars]" if len(code) > max_code else "")
        parts.append(f"\n## Source Code:\n```python\n{code_display}\n```")

    # Show visible tests on first step
    tests = obs.get("visible_tests", [])
    if step == 1 and tests:
        parts.append("\n## Test Cases (sample):")
        for i, test in enumerate(tests[:2], 1):
            t = test[:1000] + ("..." if len(test) > 1000 else "")
            parts.append(f"### Test {i}:\n```python\n{t}\n```")

    # Execution output
    exec_output = obs.get("execution_output", "")
    if exec_output:
        out = exec_output[:1500] + ("..." if len(exec_output) > 1500 else "")
        parts.append(f"\n## Execution Output:\n```\n{out}\n```")

    # Test results
    test_results = obs.get("test_results", "")
    if test_results:
        res = test_results[:1500] + ("..." if len(test_results) > 1500 else "")
        parts.append(f"\n## Test Results:\n```\n{res}\n```")

    # Feedback
    feedback = obs.get("action_feedback", "")
    if feedback:
        parts.append(f"\n## Feedback: {feedback}")

    # Progress nudging
    if step >= 3 and not has_issued:
        parts.append("\n>> IMPORTANT: You've explored enough. Now use create_issue to describe the bug you found.")
    elif has_issued and not has_fixed:
        if code:
            max_fix = 2500
            code_for_fix = code[:max_fix] + ("\n..." if len(code) > max_fix else "")
            parts.append(f"\n## Current Source Code (write your fix based on this):\n```python\n{code_for_fix}\n```")
        parts.append(
            '\n>> IMPORTANT: You already identified the bug. Your NEXT action MUST be suggest_fix.'
            '\n>> Respond with: {"action_type": "suggest_fix", "patch_code": "<entire fixed source code>"}'
            '\n>> The patch_code must contain the COMPLETE fixed file, not just the changed lines.'
        )
    elif has_fixed:
        tests_passed = obs.get("tests_passed", False)
        if tests_passed:
            parts.append("\n>> All tests passed! Use request_changes to finalize.")
        else:
            parts.append("\n>> Fix didn't pass all tests. Try suggest_fix again with a different fix, or request_changes to finalize.")

    parts.append("\nWhat is your next action? Respond with a JSON object only.")
    return "\n".join(parts)


def force_generate_fix(client: OpenAI, code: str) -> dict[str, Any]:
    """Make a dedicated LLM call to generate a fix when stuck."""
    # Strip line numbers
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        m = re.match(r"\s*\d+\s*\|\s?(.*)", line)
        cleaned.append(m.group(1) if m else line)
    raw_code = "\n".join(cleaned)

    fix_prompt = [
        {"role": "system", "content": "You are a Python bug fixer. You will be given buggy code. Output ONLY the complete fixed Python code. No explanations, no markdown, no code fences. Just the raw Python code."},
        {"role": "user", "content": f"Fix this buggy Python code:\n\n{raw_code}\n\nOutput the complete fixed code:"},
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=fix_prompt,
            temperature=0.1,
            max_tokens=2048,
        )
        fixed = (resp.choices[0].message.content or "").strip()
        if fixed.startswith("```"):
            m = re.search(r"```(?:python)?\s*\n?(.*?)```", fixed, re.DOTALL)
            if m:
                fixed = m.group(1).strip()
        if fixed and len(fixed) > 10:
            return {"action_type": "suggest_fix", "patch_code": fixed}
    except Exception as e:
        pass

    return {"action_type": "request_changes", "message": "Could not generate fix."}


# ── Episode Runner ───────────────────────────────────────────────


def run_episode(client: OpenAI, env_url: str, task_id: str, seed: int = 42) -> float:
    """Run one debugging episode. Returns the grader score in [0, 1]."""
    env = EnvClient(env_url)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    actions_taken: list[str] = []
    has_issued = False
    has_fixed = False

    step_no = 0
    rewards: list[float] = []
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id, seed=seed)
        max_steps = obs.get("max_steps", 20)

        while not obs.get("done", False) and step_no < max_steps:
            step_no += 1

            # Build user prompt and add to conversation
            user_msg = build_user_message(obs, step_no, has_issued, has_fixed)
            messages.append({"role": "user", "content": user_msg})

            # Call LLM
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048,
                )
                raw = resp.choices[0].message.content or '{"action_type": "run_tests"}'
            except Exception as e:
                raw = '{"action_type": "run_tests"}'

            # Parse the action
            action = parse_action(raw)

            # Loop detection: if LLM repeats same action 2+ times, force progression
            atype = action.get("action_type", "")
            recent = actions_taken[-2:] if len(actions_taken) >= 2 else []
            if len(recent) == 2 and all(a == atype for a in recent):
                if not has_issued:
                    action = {"action_type": "create_issue", "issue_description": "Bug detected based on test failures and code analysis."}
                elif not has_fixed:
                    code = obs.get("code", "")
                    action = force_generate_fix(client, code)

            # Track progress
            actions_taken.append(action.get("action_type", ""))
            if action.get("action_type") == "create_issue":
                has_issued = True
            if action.get("action_type") == "suggest_fix":
                has_fixed = True

            # Add assistant response to history
            messages.append({"role": "assistant", "content": json.dumps(action)})

            # Trim conversation if too long
            if len(messages) > 20:
                messages = messages[:1] + messages[-14:]

            # Step the environment
            result = env.step(action)
            obs_data = result.get("observation", result)
            reward = float(result.get("reward", 0.0) or 0.0)
            done = bool(obs_data.get("done", False) if isinstance(obs_data, dict) else result.get("done", False))
            error = None

            rewards.append(reward)
            action_str = format_action_for_log(action)
            log_step(step=step_no, action=action_str, reward=reward, done=done, error=error)

            obs = obs_data if isinstance(obs_data, dict) else {}

            if done:
                break

        # Auto-finalize if not done
        if not obs.get("done", False):
            action = {"action_type": "request_changes", "message": "auto-finalize"}
            result = env.step(action)
            step_no += 1
            reward = float(result.get("reward", 0.0) or 0.0)
            obs_data = result.get("observation", result)
            done = True
            rewards.append(reward)
            log_step(step=step_no, action="request_changes('auto-finalize')", reward=reward, done=done, error=None)

        # Get final state
        final_state = env.state()
        score = clamp01(float(final_state.get("grader_score", 0.0) or 0.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        score = 0.0
        success = False

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=step_no, score=score, rewards=rewards)

    return score


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    if not API_KEY or API_KEY == "missing-token":
        print("[WARN] API_KEY not set — LLM calls may fail", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: dict[str, float] = {}
    for task_id in TASKS:
        score = run_episode(client, ENV_URL, task_id, seed=42)
        all_scores[task_id] = score

    # Save results
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "inference_scores.json"), "w", encoding="utf-8") as f:
        json.dump(
            {tid: {"score": round(s, 4), "success": s >= SUCCESS_SCORE_THRESHOLD} for tid, s in all_scores.items()},
            f, indent=2,
        )


if __name__ == "__main__":
    main()
