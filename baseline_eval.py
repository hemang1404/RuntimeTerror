#!/usr/bin/env python3
"""
RuntimeTerror — Baseline Evaluation Script.

Runs the baseline agent across all tasks and produces reproducible metrics:
- Average reward per task
- Success rate (grader score >= threshold)
- Per-episode detailed scores

Usage:
    python baseline_eval.py                     # all tasks
    python baseline_eval.py --task easy_debug   # specific task
    python baseline_eval.py --episodes 5        # multiple episodes per task
    python baseline_eval.py --env-url http://localhost:7860  # remote server

Outputs results to outputs/baseline_scores.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

# -- Configuration ------------------------------------------------

BENCHMARK = "runtime_terror"
TASKS = ["easy_debug", "medium_debug", "hard_debug"]
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.3"))


# -- Logging ------------------------------------------------------

def log_start(task: str, mode: str) -> None:
    print(f"[START] task={task} benchmark={BENCHMARK} mode={mode}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool) -> None:
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(f"[END] success={success_val} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# -- Agent Runner (Local Mode) -----------------------------------

def run_episode_local(task_id: str, seed: int | None = None) -> dict[str, Any]:
    """Run the baseline agent locally (in-process, no server required)."""
    sys.path.insert(0, os.path.dirname(__file__))

    from server.debug_environment import DebugEnvironment
    from agent.baseline import BaselineAgent

    env = DebugEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    agent = BaselineAgent()

    step_no = 0
    rewards: list[float] = []

    log_start(task=task_id, mode="local")

    while not obs.done and step_no < env._task_def["max_steps"]:
        action = agent.act(obs.model_dump(), env.state.model_dump())
        obs = env.step(action)
        step_no += 1
        reward = obs.reward or 0.0
        rewards.append(reward)
        log_step(step_no, action.get("action_type", "unknown"), reward, obs.done)

    # Finalize if not done
    if not obs.done:
        obs = env.step({"action_type": "request_changes", "message": "auto-finalize"})
        step_no += 1
        rewards.append(obs.reward or 0.0)
        log_step(step_no, "request_changes", obs.reward or 0.0, obs.done)

    state = env.state
    score = clamp01(state.grader_score)
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=step_no, score=score, rewards=rewards)

    return {
        "score": round(score, 4),
        "cumulative_reward": round(state.cumulative_reward, 4),
        "steps": step_no,
        "success": success,
        "issue_correct": state.issue_correct,
        "issue_similarity": round(state.issue_similarity, 4),
        "fixes_attempted": state.fixes_attempted,
        "fixes_passed": state.fixes_passed,
    }


# -- Agent Runner (Remote Mode) ----------------------------------

def run_episode_remote(
    env_url: str, task_id: str, seed: int | None = None
) -> dict[str, Any]:
    """Run the baseline agent against a remote RuntimeTerror server."""
    import requests

    sys.path.insert(0, os.path.dirname(__file__))
    from agent.baseline import BaselineAgent

    http = requests.Session()
    agent = BaselineAgent()

    # Reset
    reset_payload = {"task_id": task_id}
    if seed is not None:
        reset_payload["seed"] = seed
    resp = http.post(f"{env_url}/reset", json=reset_payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]
    obs = data["observation"]

    step_no = 0
    rewards: list[float] = []

    log_start(task=task_id, mode="remote")

    while not obs.get("done", False) and step_no < 30:
        action = agent.act(obs, {})
        resp = http.post(
            f"{env_url}/step/{session_id}", json=action, timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
        obs = result.get("observation", result)
        step_no += 1
        reward = float(result.get("reward", 0.0) or 0.0)
        rewards.append(reward)
        log_step(step_no, action.get("action_type", "unknown"), reward, obs.get("done", False))

    # Finalize if not done
    if not obs.get("done", False):
        action = {"action_type": "request_changes", "message": "auto-finalize"}
        resp = http.post(f"{env_url}/step/{session_id}", json=action, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        step_no += 1
        rewards.append(float(result.get("reward", 0.0) or 0.0))

    # Get final state
    resp = http.get(f"{env_url}/state/{session_id}", timeout=30)
    resp.raise_for_status()
    state = resp.json()
    http.close()

    score = clamp01(float(state.get("grader_score", 0.0)))
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=step_no, score=score, rewards=rewards)

    return {
        "score": round(score, 4),
        "cumulative_reward": round(float(state.get("cumulative_reward", 0.0)), 4),
        "steps": step_no,
        "success": success,
        "issue_correct": state.get("issue_correct", False),
        "issue_similarity": round(float(state.get("issue_similarity", 0.0)), 4),
        "fixes_attempted": state.get("fixes_attempted", 0),
        "fixes_passed": state.get("fixes_passed", 0),
    }


# -- Main ---------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RuntimeTerror Baseline Evaluation — Reproducible benchmark metrics"
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=["easy_debug", "medium_debug", "hard_debug", "all"],
        help="Task to evaluate (default: all)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Episodes per task (use >1 for variance estimation)",
    )
    parser.add_argument(
        "--env-url", default=None,
        help="Remote server URL (default: run locally in-process)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base seed for reproducibility (each episode uses seed+i)",
    )
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]

    all_results: dict[str, Any] = {}
    print(f"\n{'=' * 60}")
    print(f"  RuntimeTerror Baseline Evaluation")
    print(f"  Tasks: {tasks}")
    print(f"  Episodes per task: {args.episodes}")
    print(f"  Mode: {'remote (' + args.env_url + ')' if args.env_url else 'local'}")
    print(f"  Success threshold: {SUCCESS_SCORE_THRESHOLD}")
    print(f"{'=' * 60}\n")

    for task_id in tasks:
        episodes = []
        for i in range(args.episodes):
            seed = (args.seed + i) if args.seed is not None else None
            print(f"\n-- {task_id} episode {i+1}/{args.episodes} (seed={seed}) --")

            if args.env_url:
                result = run_episode_remote(args.env_url, task_id, seed=seed)
            else:
                result = run_episode_local(task_id, seed=seed)

            episodes.append(result)

        scores = [ep["score"] for ep in episodes]
        rewards = [ep["cumulative_reward"] for ep in episodes]
        successes = sum(1 for ep in episodes if ep["success"])

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success_rate = successes / len(episodes) if episodes else 0.0

        all_results[task_id] = {
            "avg_score": round(avg_score, 4),
            "avg_reward": round(avg_reward, 4),
            "success_rate": round(success_rate, 4),
            "episodes": episodes,
        }

    # -- Summary ----------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Task':<20} {'Avg Score':<12} {'Avg Reward':<12} {'Success Rate':<12}")
    print(f"{'-' * 56}")
    for task_id, data in all_results.items():
        print(
            f"{task_id:<20} {data['avg_score']:<12.4f} "
            f"{data['avg_reward']:<12.4f} {data['success_rate']:<12.1%}"
        )
    print(f"{'=' * 60}\n")

    # -- Save results -----------------------------------------
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "baseline_scores.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
