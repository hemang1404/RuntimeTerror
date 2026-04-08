#!/usr/bin/env python3
"""
Submission-safe IncidentEnv inference script.

Stdout emits only:
- [START] ...
- [STEP] ...
- [END] ...
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

# Required env vars for submission
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = os.getenv("BENCHMARK", "incident_env")
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

TASKS = ["easy_triage", "medium_triage", "hard_triage"]
VALID_INVESTIGATION_ACTIONS = {
    "query_logs",
    "query_metrics",
    "inspect_code",
    "run_diagnostic",
    "submit_root_cause",
}

INVESTIGATION_SYSTEM = (
    "You are an expert on-call engineer. Investigate the incident and choose the "
    "single best next diagnostic action. Respond ONLY with strict JSON."
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

Choose ONE action and output JSON only:
- {{"action_type":"query_logs","service":"<service>","keyword":"<optional>"}}
- {{"action_type":"query_metrics","metric":"<metric>","time_range":"5m"}}
- {{"action_type":"inspect_code","file":"<filename>"}}
- {{"action_type":"run_diagnostic","command":"<command>"}}
- {{"action_type":"submit_root_cause","root_cause":"<detailed explanation>"}}
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

Write a COMPLETE fixed version of this file.
Constraints:
- Preserve existing function/class names and signatures.
- Make the smallest safe change needed to pass tests.
- Return strict JSON only, no markdown.

{{"action_type":"suggest_fix","file":"{file}","patch_code":"<complete fixed file>"}}
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    done_val = "true" if done else "false"
    err_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def get_llm_client() -> OpenAI:
    if not HF_TOKEN:
        # Keep OpenAI client usage while allowing fallback policy when token is missing.
        return OpenAI(base_url=API_BASE_URL, api_key="missing-token")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def call_llm(client: OpenAI, prompt: str, system: str = "") -> dict[str, Any]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            content = (resp.choices[0].message.content or "{}").strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(l for l in lines if not l.strip().startswith("```"))
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start : end + 1]
            return json.loads(content)
        except Exception as e:
            if attempt == 1:
                return {}
            time.sleep(0.5)
    return {}


class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None
        self._use_session_routes = True
        self._http = requests.Session()

    def reset(self, **kwargs) -> dict[str, Any]:
        resp = self._http.post(f"{self.base_url}/reset", json=kwargs, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data.get("session_id")
        self._use_session_routes = bool(self.session_id)
        return data.get("observation", data)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._use_session_routes:
            assert self.session_id
            resp = self._http.post(
                f"{self.base_url}/step/{self.session_id}", json=action, timeout=30
            )
        else:
            resp = self._http.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict[str, Any]:
        if self._use_session_routes:
            assert self.session_id
            resp = self._http.get(f"{self.base_url}/state/{self.session_id}", timeout=30)
        else:
            resp = self._http.get(f"{self.base_url}/state", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._http.close()


def _action_sig(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("action_type", ""),
        action.get("service", ""),
        action.get("keyword", ""),
        action.get("metric", ""),
        action.get("time_range", ""),
        action.get("file", ""),
        action.get("command", ""),
    )


def choose_fallback_investigation_action(obs: dict[str, Any], seen: set[tuple[Any, ...]]) -> dict[str, Any]:
    services = obs.get("available_services", [])
    metrics = obs.get("available_metrics", [])
    files = obs.get("available_files", [])
    commands = obs.get("available_commands", [])

    candidates: list[dict[str, Any]] = []
    if services:
        preferred_service = obs.get("affected_service") if obs.get("affected_service") in services else services[0]
        candidates.append({"action_type": "query_logs", "service": preferred_service})
    if files:
        candidates.append({"action_type": "inspect_code", "file": files[0]})
    if metrics:
        candidates.append({"action_type": "query_metrics", "metric": metrics[0], "time_range": "5m"})
    if commands:
        candidates.append({"action_type": "run_diagnostic", "command": commands[0]})

    for cand in candidates:
        if _action_sig(cand) not in seen:
            return cand

    return {
        "action_type": "submit_root_cause",
        "root_cause": "Likely bug in the affected service based on diagnostics and code inspection.",
    }


def sanitize_investigation_action(
    action: dict[str, Any],
    obs: dict[str, Any],
    seen: set[tuple[Any, ...]],
    force_submit: bool,
) -> dict[str, Any]:
    if force_submit:
        return {
            "action_type": "submit_root_cause",
            "root_cause": action.get("root_cause") or "Likely bug in the affected service based on diagnostics and code inspection.",
        }

    atype = action.get("action_type", "")
    if atype not in VALID_INVESTIGATION_ACTIONS:
        return choose_fallback_investigation_action(obs, seen)

    services = obs.get("available_services", [])
    metrics = obs.get("available_metrics", [])
    files = obs.get("available_files", [])
    commands = obs.get("available_commands", [])

    clean: dict[str, Any] = {"action_type": atype}
    if atype == "query_logs":
        clean["service"] = action.get("service") if action.get("service") in services else (obs.get("affected_service") if obs.get("affected_service") in services else (services[0] if services else ""))
        if action.get("keyword"):
            clean["keyword"] = action.get("keyword")
    elif atype == "query_metrics":
        clean["metric"] = action.get("metric") if action.get("metric") in metrics else (metrics[0] if metrics else "")
        clean["time_range"] = action.get("time_range", "5m")
    elif atype == "inspect_code":
        clean["file"] = action.get("file") if action.get("file") in files else (files[0] if files else "")
    elif atype == "run_diagnostic":
        clean["command"] = action.get("command") if action.get("command") in commands else (commands[0] if commands else "")
    elif atype == "submit_root_cause":
        clean["root_cause"] = action.get("root_cause", "Likely bug in the affected service.")

    if _action_sig(clean) in seen:
        return choose_fallback_investigation_action(obs, seen)

    return clean


def run_episode(llm: OpenAI, env_url: str, task_id: str) -> float:
    env = EnvClient(env_url)
    history_lines: list[str] = []
    root_cause = ""
    inspected_code: dict[str, str] = {}
    seen_actions: set[tuple[Any, ...]] = set()

    step_no = 0
    rewards: list[float] = []
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id)

        did_open_logs = False
        did_open_inspect = False

        while obs.get("phase") == "investigation" and not obs.get("done"):
            if not did_open_logs:
                action = choose_fallback_investigation_action(obs, seen_actions)
                if action.get("action_type") == "query_logs":
                    did_open_logs = True
            elif not did_open_inspect and obs.get("available_files", []):
                action = {"action_type": "inspect_code", "file": obs.get("available_files", [""])[0]}
                did_open_inspect = True
            else:
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
                llm_action = call_llm(llm, prompt, INVESTIGATION_SYSTEM)
                step_num = int(obs.get("step_number", 0))
                max_steps = int(obs.get("max_steps", 15))
                force_submit = step_num >= max_steps - 2
                action = sanitize_investigation_action(llm_action, obs, seen_actions, force_submit)

            seen_actions.add(_action_sig(action))
            result = env.step(action)
            step_no += 1
            reward = float(result.get("reward", 0.0) or 0.0)
            done = bool(result.get("done", False))
            obs_data = result.get("observation", result)
            error = obs_data.get("last_action_error") if isinstance(obs_data, dict) else None
            log_step(step_no, json.dumps(action, separators=(",", ":")), reward, done, error)
            rewards.append(reward)

            output = obs_data.get("output", "") if isinstance(obs_data, dict) else ""
            atype = action.get("action_type", "")
            history_lines.append(f"[{atype}] -> {output[:200]}")

            if atype == "submit_root_cause":
                root_cause = action.get("root_cause", "")
            if atype == "inspect_code" and isinstance(obs_data, dict) and "ERROR" not in output:
                inspected_code[action.get("file", "")] = output

            obs = obs_data if isinstance(obs_data, dict) else {}

        test_output = ""
        fix_attempts = 0
        while not obs.get("done") and fix_attempts < 3:
            files = obs.get("available_files", [])
            target_file = files[0] if files else ""
            code = inspected_code.get(target_file, "")

            action: dict[str, Any]
            if code:
                prompt = FIX_PROMPT.format(
                    root_cause=root_cause or "Likely bug in affected service.",
                    file=target_file,
                    code=code,
                    test_output=test_output or "(no previous attempt)",
                )
                action = call_llm(
                    llm,
                    prompt,
                    "You are an expert software engineer. Fix the bug and respond ONLY with strict JSON.",
                )
            else:
                action = {}

            if action.get("action_type") != "suggest_fix":
                action = {
                    "action_type": "suggest_fix",
                    "file": target_file,
                    "patch_code": code or "# no-op fallback patch",
                }
            else:
                action["file"] = action.get("file") or target_file

            result = env.step(action)
            step_no += 1
            reward = float(result.get("reward", 0.0) or 0.0)
            done = bool(result.get("done", False))
            obs_data = result.get("observation", result)
            error = obs_data.get("last_action_error") if isinstance(obs_data, dict) else None
            log_step(step_no, json.dumps(action, separators=(",", ":")), reward, done, error)
            rewards.append(reward)

            if isinstance(obs_data, dict):
                test_output = obs_data.get("test_output", "")
            fix_attempts += 1
            obs = obs_data if isinstance(obs_data, dict) else {}

            if isinstance(obs_data, dict) and obs_data.get("tests_passed"):
                break

        if not obs.get("done"):
            action = {"action_type": "submit_resolution"}
            result = env.step(action)
            step_no += 1
            reward = float(result.get("reward", 0.0) or 0.0)
            done = bool(result.get("done", False))
            obs_data = result.get("observation", result)
            error = obs_data.get("last_action_error") if isinstance(obs_data, dict) else None
            log_step(step_no, json.dumps(action, separators=(",", ":")), reward, done, error)
            rewards.append(reward)

        final_state = env.state()
        score = clamp01(float(final_state.get("grader_score", 0.0) or 0.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score

    except Exception as e:
        score = 0.0
        success = False
        return 0.0

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=step_no, score=score, rewards=rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description="IncidentEnv Submission Inference")
    parser.add_argument(
        "--task",
        default="all",
        choices=["easy_triage", "medium_triage", "hard_triage", "all"],
        help="Task to run (default: all)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument("--env-url", default=None, help=f"Environment URL (default: {ENV_URL})")
    args = parser.parse_args()

    env_url = args.env_url or ENV_URL
    tasks = TASKS if args.task == "all" else [args.task]
    llm = get_llm_client()

    all_results: dict[str, Any] = {}
    for task_id in tasks:
        scores = []
        for _ in range(args.episodes):
            score = run_episode(llm, env_url, task_id)
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.0
        all_results[task_id] = {"scores": [round(s, 4) for s in scores], "avg": round(avg, 4)}

    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "baseline_scores.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
