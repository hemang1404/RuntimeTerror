#!/usr/bin/env python3
"""
IncidentEnv — Baseline Inference Script.

This script demonstrates a two-phase AI agent that:
  1. Investigates a production incident using diagnostic tools.
  2. Suggests a code fix that is validated by test execution.

Required environment variables
------------------------------
    API_BASE_URL   LLM API endpoint (e.g. https://api-inference.huggingface.co/v1)
    MODEL_NAME     Model identifier  (e.g. mistralai/Mistral-7B-Instruct-v0.3)
    HF_TOKEN       API key           (e.g. hf_xxxxxxxxxxxxxxxx)

Usage
-----
    python inference.py                         # run all 3 tasks, 1 episode each
    python inference.py --task easy_triage      # run just easy
    python inference.py --episodes 3            # 3 episodes per task
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import requests
from openai import OpenAI

# ── Required env vars ────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

TASKS = ["easy_triage", "medium_triage", "hard_triage"]

# ── Prompts ──────────────────────────────────────────────────────

INVESTIGATION_SYSTEM = (
    "You are an expert on-call engineer. You are investigating a production "
    "incident. Analyse the information provided and choose the BEST next "
    "diagnostic action. Respond ONLY with a JSON object."
)

INVESTIGATION_PROMPT = """\
## Incident Alert
**{alert_title}**
{alert_description}
Severity: {severity} | Affected: {affected_service}

## Available Resources
- Services: {services}
- Metrics:  {metrics}
- Files:    {files}
- Commands: {commands}

## Investigation History (step {step}/{max_steps})
{history}

## Your Task
Choose ONE action. Respond with valid JSON matching one of these schemas:
- {{"action_type":"query_logs","service":"<service>","keyword":"<optional>"}}
- {{"action_type":"query_metrics","metric":"<metric>","time_range":"5m"}}
- {{"action_type":"inspect_code","file":"<filename>"}}
- {{"action_type":"run_diagnostic","command":"<command>"}}
- {{"action_type":"submit_root_cause","root_cause":"<detailed explanation>"}}

If you have enough information, submit_root_cause. Otherwise keep investigating.
"""

FIX_PROMPT = """\
## Root Cause
{root_cause}

## Buggy File: {file}
```
{code}
```

## Test Output (from previous attempt, if any)
{test_output}

Write a COMPLETE fixed version of this file. Respond ONLY with JSON:
{{"action_type":"suggest_fix","file":"{file}","patch_code":"<complete fixed file>"}}
"""

# ── LLM Client ──────────────────────────────────────────────────

def get_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client."""
    if not API_BASE_URL or not HF_TOKEN:
        print("ERROR: API_BASE_URL and HF_TOKEN must be set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def call_llm(client: OpenAI, prompt: str, system: str = "") -> dict[str, Any]:
    """Call the LLM and parse the JSON response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
            content = resp.choices[0].message.content or "{}"
            # Try to extract JSON from the response
            content = content.strip()
            if content.startswith("```"):
                # Strip markdown code fences
                lines = content.split("\n")
                content = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                )
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            if attempt == 2:
                print(f"  LLM parse error after 3 attempts: {e}", file=sys.stderr)
                return {"action_type": "submit_root_cause", "root_cause": "Unable to determine"}
            time.sleep(1)
    return {}


# ── Environment Client ───────────────────────────────────────────

class EnvClient:
    """Simple HTTP client for IncidentEnv."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None
        self._http = requests.Session()

    def reset(self, **kwargs) -> dict:
        resp = self._http.post(f"{self.base_url}/reset", json=kwargs)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data.get("session_id")
        return data.get("observation", data)

    def step(self, action: dict) -> dict:
        assert self.session_id
        resp = self._http.post(
            f"{self.base_url}/step/{self.session_id}", json=action
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        assert self.session_id
        resp = self._http.get(f"{self.base_url}/state/{self.session_id}")
        resp.raise_for_status()
        return resp.json()


# ── Episode Runner ───────────────────────────────────────────────

def run_episode(
    llm: OpenAI, env: EnvClient, task_id: str
) -> float:
    """Run one full episode (investigate → fix). Return grader score."""
    obs = env.reset(task_id=task_id)
    history_lines: list[str] = []
    root_cause = ""
    inspected_code: dict[str, str] = {}

    # ── Phase 1: Investigation ─────────────────────────────────
    while obs.get("phase") == "investigation" and not obs.get("done"):
        prompt = INVESTIGATION_PROMPT.format(
            alert_title=obs.get("alert_title", ""),
            alert_description=obs.get("alert_description", ""),
            severity=obs.get("severity_level", ""),
            affected_service=obs.get("affected_service", ""),
            services=", ".join(obs.get("available_services", [])),
            metrics=", ".join(obs.get("available_metrics", [])),
            files=", ".join(obs.get("available_files", [])),
            commands=", ".join(obs.get("available_commands", [])),
            step=obs.get("step_number", 0),
            max_steps=obs.get("max_steps", 15),
            history="\n".join(history_lines[-10:]) or "(none yet)",
        )
        action = call_llm(llm, prompt, INVESTIGATION_SYSTEM)

        result = env.step(action)
        obs_data = result.get("observation", result)

        output = obs_data.get("output", "")
        atype = action.get("action_type", "")
        history_lines.append(f"[{atype}] → {output[:200]}")

        if atype == "submit_root_cause":
            root_cause = action.get("root_cause", "")

        if atype == "inspect_code" and "ERROR" not in output:
            inspected_code[action.get("file", "")] = output

        obs = obs_data

    # ── Phase 2: Remediation ──────────────────────────────────
    test_output = ""
    fix_attempts = 0
    while not obs.get("done") and fix_attempts < 3:
        # Determine which file to fix
        files = obs.get("available_files", [])
        target_file = files[0] if files else ""

        # Get code for the file
        code = inspected_code.get(target_file, "(not inspected — inspect it first)")

        prompt = FIX_PROMPT.format(
            root_cause=root_cause,
            file=target_file,
            code=code,
            test_output=test_output or "(no previous attempt)",
        )
        action = call_llm(llm, prompt, "You are an expert software engineer. Fix the bug. Respond ONLY with JSON.")

        if action.get("action_type") != "suggest_fix":
            action["action_type"] = "suggest_fix"
            action["file"] = target_file

        result = env.step(action)
        obs_data = result.get("observation", result)
        test_output = obs_data.get("test_output", "")
        fix_attempts += 1

        obs = obs_data

        if obs_data.get("tests_passed"):
            break

    # ── Submit resolution ────────────────────────────────────
    if not obs.get("done"):
        result = env.step({"action_type": "submit_resolution"})
        obs = result.get("observation", result)

    # Get final state
    final_state = env.state()
    return final_state.get("grader_score", 0.0)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IncidentEnv Baseline Inference")
    parser.add_argument(
        "--task", default="all",
        choices=["easy_triage", "medium_triage", "hard_triage", "all"],
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Episodes per task (default: 1)",
    )
    parser.add_argument(
        "--env-url", default=None,
        help=f"Environment URL (default: {ENV_URL})",
    )
    args = parser.parse_args()

    env_url = args.env_url or ENV_URL
    tasks = TASKS if args.task == "all" else [args.task]
    llm = get_llm_client()
    env = EnvClient(env_url)

    print(f"IncidentEnv Inference")
    print(f"  LLM:      {API_BASE_URL} / {MODEL_NAME}")
    print(f"  Env:      {env_url}")
    print(f"  Tasks:    {tasks}")
    print(f"  Episodes: {args.episodes}")
    print()

    start = time.time()
    all_results: dict[str, Any] = {}

    for task_id in tasks:
        scores = []
        for ep in range(args.episodes):
            print(f"[{task_id}] Episode {ep+1}/{args.episodes} ...", end=" ", flush=True)
            try:
                score = run_episode(llm, env, task_id)
                scores.append(score)
                print(f"score={score:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                scores.append(0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        all_results[task_id] = {"scores": scores, "avg": round(avg, 4)}
        print(f"  → {task_id} average: {avg:.4f}\n")

    elapsed = time.time() - start
    print(f"Total runtime: {elapsed:.1f}s")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    results_path = os.path.join("outputs", "baseline_scores.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
