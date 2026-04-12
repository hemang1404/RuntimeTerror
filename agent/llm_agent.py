"""
RuntimeTerror — LLM-Powered Debugging Agent.

Uses an OpenAI-compatible API (Ollama locally, LiteLLM proxy in hackathon)
to drive the debugging loop. The LLM reads code, reasons about bugs,
runs experiments, and submits fixes.

Local:      API_BASE_URL=http://localhost:11434/v1  MODEL_NAME=qwen2.5-coder:7b
Hackathon:  API_BASE_URL and API_KEY are injected by the platform
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

# ── Configuration ────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "ollama")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5-coder:7b")

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


def create_client():
    """Create an OpenAI-compatible client."""
    try:
        from openai import OpenAI
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except ImportError:
        # Fallback to raw HTTP if openai package not installed
        return None


def call_llm_openai(client, messages: list[dict], temperature: float = 0.2) -> str:
    """Call the LLM via the OpenAI client."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


def call_llm_http(messages: list[dict], temperature: float = 0.2) -> str:
    """Call the LLM via raw HTTP (fallback if openai not installed)."""
    import requests

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    resp = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


def parse_action(raw_response: str) -> dict[str, Any]:
    """Extract a JSON action from the LLM response.

    Handles common LLM quirks: markdown code fences, extra text, etc.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Try to find JSON object in the text
    # Look for the outermost { ... }
    brace_start = text.find("{")
    if brace_start == -1:
        return {"action_type": "run_tests"}  # safe fallback

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
        # Try to fix common JSON issues
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        try:
            action = json.loads(json_str)
            if "action_type" not in action:
                return {"action_type": "run_tests"}
            return action
        except json.JSONDecodeError:
            return {"action_type": "run_tests"}


class LLMAgent:
    """LLM-powered debugging agent for RuntimeTerror."""

    def __init__(self) -> None:
        self._client = create_client()
        self._messages: list[dict[str, str]] = []
        self._step = 0
        self._actions_taken: list[str] = []
        self._has_issued = False
        self._has_fixed = False

    def reset(self) -> None:
        """Reset the agent for a new episode."""
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._step = 0
        self._actions_taken = []
        self._has_issued = False
        self._has_fixed = False

    def act(self, observation: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Choose the next action using the LLM."""
        self._step += 1

        # Build the user message from the observation
        user_msg = self._format_observation(observation)
        self._messages.append({"role": "user", "content": user_msg})

        # Call the LLM
        try:
            if self._client is not None:
                raw = call_llm_openai(self._client, self._messages)
            else:
                raw = call_llm_http(self._messages)
        except Exception as e:
            print(f"  [LLM ERROR] {e}", flush=True)
            raw = '{"action_type": "run_tests"}'

        # Parse the action
        action = parse_action(raw)

        # Loop detection: if the LLM repeats the same action 2+ times, force progression
        atype = action.get("action_type", "")
        recent = self._actions_taken[-2:] if len(self._actions_taken) >= 2 else []
        if len(recent) == 2 and all(a == atype for a in recent):
            if not self._has_issued:
                action = {"action_type": "create_issue", "issue_description": "Bug detected based on test failures and code analysis."}
            elif not self._has_fixed:
                # Make a dedicated LLM call to generate the fix
                action = self._force_generate_fix(observation)

        # Track progress
        self._actions_taken.append(action.get("action_type", ""))
        if action.get("action_type") == "create_issue":
            self._has_issued = True
        if action.get("action_type") == "suggest_fix":
            self._has_fixed = True

        # Add assistant response to history (for multi-turn context)
        self._messages.append({"role": "assistant", "content": json.dumps(action)})

        # Keep conversation manageable -- trim if too long
        if len(self._messages) > 20:
            # Keep system prompt + last 14 messages
            self._messages = self._messages[:1] + self._messages[-14:]

        return action

    def _force_generate_fix(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Make a dedicated LLM call to generate a fix when the agent is stuck."""
        code = observation.get("code", "")
        # Strip line numbers
        import re as _re
        lines = code.split("\n")
        cleaned = []
        for line in lines:
            m = _re.match(r"\s*\d+\s*\|\s?(.*)", line)
            cleaned.append(m.group(1) if m else line)
        raw_code = "\n".join(cleaned)

        fix_prompt = [
            {"role": "system", "content": "You are a Python bug fixer. You will be given buggy code. Output ONLY the complete fixed Python code. No explanations, no markdown, no code fences. Just the raw Python code."},
            {"role": "user", "content": f"Fix this buggy Python code:\n\n{raw_code}\n\nOutput the complete fixed code:"},
        ]

        try:
            if self._client is not None:
                fixed = call_llm_openai(self._client, fix_prompt, temperature=0.1)
            else:
                fixed = call_llm_http(fix_prompt, temperature=0.1)

            # Clean up: strip code fences if the model added them
            fixed = fixed.strip()
            if fixed.startswith("```"):
                m = _re.search(r"```(?:python)?\s*\n?(.*?)```", fixed, _re.DOTALL)
                if m:
                    fixed = m.group(1).strip()

            if fixed and len(fixed) > 10:
                return {"action_type": "suggest_fix", "patch_code": fixed}
        except Exception as e:
            print(f"  [FIX GEN ERROR] {e}", flush=True)

        # Ultimate fallback: submit what we have
        return {"action_type": "request_changes", "message": "Could not generate fix."}

    def _format_observation(self, obs: dict[str, Any]) -> str:
        """Format the observation into a clear prompt for the LLM."""
        parts = []

        step = obs.get("step_number", self._step)
        max_steps = obs.get("max_steps", 20)
        parts.append(f"[Step {step}/{max_steps}]")

        # Show code on first step — truncate large files for LLM speed
        code = obs.get("code", "")
        if self._step == 1 and code:
            MAX_CODE = 3000
            if len(code) > MAX_CODE:
                code_display = code[:MAX_CODE] + f"\n... [truncated {len(code)-MAX_CODE} chars] ..."
            else:
                code_display = code
            parts.append(f"\n## Source Code:\n```python\n{code_display}\n```")

        # Show visible tests on first step — limit to 2 files, 1000 chars each
        tests = obs.get("visible_tests", [])
        if self._step == 1 and tests:
            parts.append("\n## Test Cases (sample):")
            for i, test in enumerate(tests[:2], 1):
                t = test[:1000] + ("..." if len(test) > 1000 else "")
                parts.append(f"### Test {i}:\n```python\n{t}\n```")

        # Show execution output (truncated)
        exec_output = obs.get("execution_output", "")
        if exec_output:
            out = exec_output[:1500] + ("..." if len(exec_output) > 1500 else "")
            parts.append(f"\n## Execution Output:\n```\n{out}\n```")

        # Show test results (truncated)
        test_results = obs.get("test_results", "")
        if test_results:
            res = test_results[:1500] + ("..." if len(test_results) > 1500 else "")
            parts.append(f"\n## Test Results:\n```\n{res}\n```")

        # Show feedback
        feedback = obs.get("action_feedback", "")
        if feedback:
            parts.append(f"\n## Feedback: {feedback}")

        # Show reward
        reward = obs.get("reward")
        if reward is not None and self._step > 1:
            parts.append(f"Reward: {reward:+.2f}")

        # Progress nudging to prevent loops
        code = obs.get("code", "")
        if self._step >= 3 and not self._has_issued:
            parts.append("\n>> IMPORTANT: You've explored enough. Now use create_issue to describe the bug you found.")
        elif self._has_issued and not self._has_fixed:
            # Re-show the code so the LLM can write the fix — truncated
            if code:
                MAX_FIX_CODE = 2500
                code_for_fix = code[:MAX_FIX_CODE] + ("\n..." if len(code) > MAX_FIX_CODE else "")
                parts.append(f"\n## Current Source Code (write your fix based on this):\n```python\n{code_for_fix}\n```")
            parts.append(
                "\n>> IMPORTANT: You already identified the bug. Your NEXT action MUST be suggest_fix."
                "\n>> Respond with: {\"action_type\": \"suggest_fix\", \"patch_code\": \"<entire fixed source code>\"}"
                "\n>> The patch_code must contain the COMPLETE fixed file, not just the changed lines."
            )
        elif self._has_fixed:
            tests_passed = obs.get("tests_passed", False)
            if tests_passed:
                parts.append("\n>> All tests passed! Use request_changes to finalize.")
            else:
                parts.append("\n>> Fix didn't pass all tests. Try suggest_fix again with a different fix, or request_changes to finalize.")

        parts.append("\nWhat is your next action? Respond with a JSON object only.")

        return "\n".join(parts)


# ── Main: Run the LLM agent against the environment ─────────────

def run_episode(task_id: str, seed: int | None = None, verbose: bool = True, pr_url: str | None = None) -> dict[str, Any]:
    """Run one episode with the LLM agent."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    if pr_url:
        from server.pr_environment import PREnvironment
        env = PREnvironment()
        obs = env.reset(pr_url=pr_url)
    else:
        from server.debug_environment import DebugEnvironment
        env = DebugEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)

    agent = LLMAgent()
    agent.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task_id} | Difficulty: {obs.difficulty}")
        print(f"  Model: {MODEL_NAME}")
        print(f"  API: {API_BASE_URL}")
        print(f"{'='*60}")

    rewards = []
    step = 0

    while not obs.done and step < env._task_def["max_steps"]:
        t0 = time.time()
        action = agent.act(obs.model_dump(), env.state.model_dump())
        t1 = time.time()

        obs = env.step(action)
        step += 1
        reward = obs.reward or 0.0
        rewards.append(reward)

        if verbose:
            atype = action.get("action_type", "?")
            print(f"  Step {step}: {atype:20s} -> reward={reward:+.2f}  ({t1-t0:.1f}s)")
            if atype == "run_code":
                print(f"         code: {action.get('code', '')[:80]}")
            elif atype == "create_issue":
                print(f"         issue: {action.get('issue_description', '')[:80]}")
            elif atype == "suggest_fix":
                print(f"         fix: {len(action.get('patch_code', ''))} chars")

    # Auto-finalize if not done
    if not obs.done:
        obs = env.step({"action_type": "request_changes", "message": "auto-finalize"})
        step += 1
        rewards.append(obs.reward or 0.0)

    state = env.state
    if verbose:
        print(f"\n  {'-'*40}")
        print(f"  Score:        {state.grader_score:.4f}")
        print(f"  Cumulative:   {state.cumulative_reward:.2f}")
        print(f"  Steps:        {step}")
        print(f"  Issue correct: {'Yes' if state.issue_correct else 'No'} ({state.issue_similarity:.0%})")
        print(f"  Fixes passed:  {state.fixes_passed}/{state.fixes_attempted}")
        print(f"  {'-'*40}")

    return {
        "score": round(state.grader_score, 4),
        "cumulative_reward": round(state.cumulative_reward, 4),
        "steps": step,
        "success": state.grader_score >= 0.3,
        "issue_correct": state.issue_correct,
        "issue_similarity": round(state.issue_similarity, 4),
        "fixes_attempted": state.fixes_attempted,
        "fixes_passed": state.fixes_passed,
    }


def main():
    """Run the LLM agent on a task."""
    import argparse

    parser = argparse.ArgumentParser(description="RuntimeTerror LLM Agent")
    parser.add_argument("--task", default="easy_debug",
                        choices=["easy_debug", "medium_debug", "hard_debug"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--pr", type=str, help="GitHub PR URL to test against")
    args = parser.parse_args()

    print(f"Config: model={MODEL_NAME}, api={API_BASE_URL}")

    if args.pr:
        result = run_episode(f"pr_{args.pr.split('/')[-1]}", seed=args.seed, pr_url=args.pr)
        results = {args.pr: result}
    elif args.all:
        tasks = ["easy_debug", "medium_debug", "hard_debug"]
        results = {}
        for task_id in tasks:
            result = run_episode(task_id, seed=args.seed)
            results[task_id] = result
    else:
        tasks = [args.task]
        results = {}
        for task_id in tasks:
            result = run_episode(task_id, seed=args.seed)
            results[task_id] = result

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Task':<20} {'Score':<10} {'Reward':<10} {'Fix?':<6}")
        print(f"  {'-'*46}")
        for tid, r in results.items():
            fix = "Yes" if r["fixes_passed"] > 0 else "No"
            print(f"  {tid:<20} {r['score']:<10.4f} {r['cumulative_reward']:<10.2f} {fix:<6}")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/llm_agent_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to outputs/llm_agent_scores.json")


if __name__ == "__main__":
    main()
