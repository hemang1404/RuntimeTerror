"""
NitpickAI — FastAPI + Gradio Server Application.

Creates the HTTP server for the debugging environment, compatible with
the OpenEnv client ecosystem. Includes a Gradio-based UI with:
  1. Interactive mode — humans can manually debug code
  2. Agent demo mode — watch the baseline agent step through a task
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse

from .debug_environment import DebugEnvironment

app = FastAPI(title="NitpickAI", version="0.1.0")

# Session storage (one env per session)
_sessions: dict[str, DebugEnvironment] = {}

# ── HTTP endpoints (OpenEnv-compatible) ──────────────────────────


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})


@app.get("/schema")
async def schema():
    from models import DebugAction, DebugObservation, DebugState
    return JSONResponse({
        "action": DebugAction.model_json_schema(),
        "observation": DebugObservation.model_json_schema(),
        "state": DebugState.model_json_schema(),
    })


# ── REST endpoints for agent clients ────────────────────────────


@app.post("/reset")
async def reset_endpoint(body: dict[str, Any] | None = None):
    body = body or {}
    env = DebugEnvironment()
    session_id = str(uuid.uuid4())
    _sessions[session_id] = env
    obs = env.reset(**body)
    return JSONResponse({
        "session_id": session_id,
        "observation": obs.model_dump(),
    })


@app.post("/step/{session_id}")
async def step_endpoint(session_id: str, body: dict[str, Any]):
    env = _sessions.get(session_id)
    if env is None:
        return JSONResponse({"error": "Unknown session"}, status_code=404)
    obs = env.step(body)
    resp: dict[str, Any] = {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }
    if obs.done:
        resp["state"] = env.state.model_dump()
    return JSONResponse(resp)


@app.get("/state/{session_id}")
async def state_endpoint(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        return JSONResponse({"error": "Unknown session"}, status_code=404)
    return JSONResponse(env.state.model_dump())


# ── WebSocket endpoint ───────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    env = DebugEnvironment()
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "reset":
                data = msg.get("data", {})
                obs = env.reset(**data)
                await ws.send_json({
                    "type": "observation",
                    "data": obs.model_dump(),
                })

            elif msg_type == "step":
                data = msg.get("data", {})
                obs = env.step(data)
                await ws.send_json({
                    "type": "observation",
                    "data": obs.model_dump(),
                })

            elif msg_type == "state":
                await ws.send_json({
                    "type": "state",
                    "data": env.state.model_dump(),
                })

            elif msg_type == "close":
                await ws.close()
                break

            else:
                await ws.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown type: {msg_type}"},
                })

    except WebSocketDisconnect:
        pass


# ── Gradio UI ────────────────────────────────────────────────────

def build_gradio_app():
    """Build the Gradio interface with Interactive + Agent Demo tabs."""
    import gradio as gr

    # ── Interactive Mode State ─────────────────────────────────
    interactive_env = [None]  # mutable container for environment reference
    interactive_rewards = [None]  # track reward history

    def interactive_reset(task_id, seed):
        env = DebugEnvironment()
        seed_val = int(seed) if seed else None
        obs = env.reset(task_id=task_id, seed=seed_val)
        interactive_env[0] = env
        interactive_rewards[0] = []
        return (
            obs.code,
            "\n\n".join(obs.visible_tests),
            "",
            "",
            obs.action_feedback,
            f"Step: 0/{obs.max_steps}  |  Reward: 0.0",
            "",
        )

    def interactive_run_code(code_snippet):
        env = interactive_env[0]
        if env is None:
            return ("", "", "Reset the environment first!", "", "")
        obs = env.step({"action_type": "run_code", "code": code_snippet})
        interactive_rewards[0].append(obs.reward or 0.0)
        reward_log = " → ".join(f"{r:+.2f}" for r in interactive_rewards[0])
        return (
            obs.execution_output,
            obs.test_results,
            obs.action_feedback,
            f"Step: {obs.step_number}/{obs.max_steps}  |  Cumulative: {env.state.cumulative_reward:.2f}",
            f"Rewards: {reward_log}",
        )

    def interactive_run_tests():
        env = interactive_env[0]
        if env is None:
            return ("", "", "Reset the environment first!", "", "")
        obs = env.step({"action_type": "run_tests"})
        interactive_rewards[0].append(obs.reward or 0.0)
        reward_log = " → ".join(f"{r:+.2f}" for r in interactive_rewards[0])
        return (
            obs.execution_output,
            obs.test_results,
            obs.action_feedback,
            f"Step: {obs.step_number}/{obs.max_steps}  |  Cumulative: {env.state.cumulative_reward:.2f}",
            f"Rewards: {reward_log}",
        )

    def interactive_create_issue(issue_desc):
        env = interactive_env[0]
        if env is None:
            return ("", "", "Reset the environment first!", "", "")
        obs = env.step({"action_type": "create_issue", "issue_description": issue_desc})
        interactive_rewards[0].append(obs.reward or 0.0)
        reward_log = " → ".join(f"{r:+.2f}" for r in interactive_rewards[0])
        return (
            obs.execution_output,
            obs.test_results,
            obs.action_feedback,
            f"Step: {obs.step_number}/{obs.max_steps}  |  Cumulative: {env.state.cumulative_reward:.2f}",
            f"Rewards: {reward_log}",
        )

    def interactive_suggest_fix(patch_code):
        env = interactive_env[0]
        if env is None:
            return ("", "", "", "Reset the environment first!", "", "")
        obs = env.step({"action_type": "suggest_fix", "patch_code": patch_code})
        interactive_rewards[0].append(obs.reward or 0.0)
        reward_log = " → ".join(f"{r:+.2f}" for r in interactive_rewards[0])
        return (
            obs.code,
            obs.execution_output,
            obs.test_results,
            obs.action_feedback,
            f"Step: {obs.step_number}/{obs.max_steps}  |  Cumulative: {env.state.cumulative_reward:.2f}",
            f"Rewards: {reward_log}",
        )

    def interactive_request_changes(message):
        env = interactive_env[0]
        if env is None:
            return ("", "", "Reset the environment first!", "", "")
        obs = env.step({"action_type": "request_changes", "message": message})
        interactive_rewards[0].append(obs.reward or 0.0)
        reward_log = " → ".join(f"{r:+.2f}" for r in interactive_rewards[0])
        return (
            obs.execution_output,
            obs.test_results,
            obs.action_feedback,
            f"Step: {obs.step_number}/{obs.max_steps}  |  Score: {env.state.grader_score:.4f}  |  Cumulative: {env.state.cumulative_reward:.2f}",
            f"Rewards: {reward_log}",
        )

    # ── Agent Demo Mode ────────────────────────────────────────

    def run_agent_demo(task_id, seed):
        """Run the baseline agent and yield step-by-step results."""
        from agent.baseline import BaselineAgent

        env = DebugEnvironment()
        seed_val = int(seed) if seed else None
        obs = env.reset(task_id=task_id, seed=seed_val)

        agent = BaselineAgent()
        log_lines = []
        rewards = []
        step = 0

        log_lines.append(f"🔧 Task: {task_id} | Difficulty: {obs.difficulty}")
        log_lines.append(f"📝 Code:\n{obs.code[:500]}...\n")

        while not obs.done and step < env._task_def["max_steps"]:
            action = agent.act(obs.model_dump(), env.state.model_dump())
            atype = action.get("action_type", "unknown")
            obs = env.step(action)
            step += 1
            reward = obs.reward or 0.0
            rewards.append(reward)

            log_lines.append(f"─── Step {step} ───")
            log_lines.append(f"Action: {atype}")
            if atype == "run_code":
                log_lines.append(f"  Code: {action.get('code', '')[:100]}")
            elif atype == "create_issue":
                log_lines.append(f"  Issue: {action.get('issue_description', '')[:100]}")
            elif atype == "suggest_fix":
                log_lines.append(f"  Fix submitted ({len(action.get('patch_code', ''))} chars)")
            log_lines.append(f"  Reward: {reward:+.2f} | Cumulative: {env.state.cumulative_reward:.2f}")
            if obs.action_feedback:
                log_lines.append(f"  Feedback: {obs.action_feedback}")
            log_lines.append("")

        # Final summary
        state = env.state
        reward_str = " → ".join(f"{r:+.2f}" for r in rewards)
        summary = (
            f"\n{'═' * 40}\n"
            f"🏆 FINAL RESULTS\n"
            f"{'═' * 40}\n"
            f"Grader Score: {state.grader_score:.4f}\n"
            f"Cumulative Reward: {state.cumulative_reward:.2f}\n"
            f"Steps Used: {state.step_count}/{env._task_def['max_steps']}\n"
            f"Issue Correct: {'✅' if state.issue_correct else '❌'} ({state.issue_similarity:.0%})\n"
            f"Fixes Attempted: {state.fixes_attempted}\n"
            f"Fixes Passed: {state.fixes_passed}\n"
            f"Reward Progression: {reward_str}\n"
        )
        log_lines.append(summary)

        return "\n".join(log_lines), reward_str, f"{state.grader_score:.4f}"

    # ── Build Gradio Interface ─────────────────────────────────

    with gr.Blocks(
        title="NitpickAI — Interactive Debugging Benchmark",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .code-display textarea { font-family: 'Fira Code', 'Cascadia Code', monospace !important; font-size: 13px !important; }
        .status-bar { font-size: 14px; font-weight: 600; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🔍 NitpickAI — Interactive Debugging Benchmark
            **Debug real Python code. Identify bugs. Submit fixes. Get graded.**

            An OpenEnv-compliant environment for evaluating AI agents on interactive code debugging.
            """
        )

        with gr.Tabs():
            # ── Tab 1: Interactive Mode ────────────────────────
            with gr.Tab("🎮 Interactive Mode"):
                gr.Markdown("### Debug code manually — explore the environment yourself")

                with gr.Row():
                    task_select = gr.Dropdown(
                        choices=["easy_debug", "medium_debug", "hard_debug"],
                        value="easy_debug",
                        label="Task",
                    )
                    seed_input = gr.Textbox(
                        value="42", label="Seed (optional)",
                        max_lines=1,
                    )
                    reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

                status_bar = gr.Textbox(
                    label="Status", interactive=False,
                    elem_classes=["status-bar"],
                )
                reward_bar = gr.Textbox(
                    label="Reward Progression", interactive=False,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        code_display = gr.Textbox(
                            label="📄 Source Code",
                            lines=18, interactive=False,
                            elem_classes=["code-display"],
                        )
                        tests_display = gr.Textbox(
                            label="🧪 Visible Tests",
                            lines=10, interactive=False,
                            elem_classes=["code-display"],
                        )

                    with gr.Column(scale=1):
                        exec_output = gr.Textbox(
                            label="💻 Execution Output",
                            lines=8, interactive=False,
                            elem_classes=["code-display"],
                        )
                        test_results = gr.Textbox(
                            label="📊 Test Results",
                            lines=8, interactive=False,
                            elem_classes=["code-display"],
                        )
                        feedback = gr.Textbox(
                            label="🔔 Action Feedback",
                            lines=3, interactive=False,
                        )

                gr.Markdown("### Actions")

                with gr.Accordion("▶️ Run Code", open=False):
                    code_input = gr.Textbox(
                        label="Code Snippet",
                        lines=4,
                        placeholder="from code import *\nprint(calculate_total([Item(10)]))",
                        elem_classes=["code-display"],
                    )
                    run_code_btn = gr.Button("▶️ Run Code", variant="secondary")

                with gr.Accordion("🧪 Run Tests", open=False):
                    run_tests_btn = gr.Button(
                        "🧪 Run Visible Tests", variant="secondary",
                    )

                with gr.Accordion("🐛 Create Issue", open=False):
                    issue_input = gr.Textbox(
                        label="Bug Description",
                        lines=3,
                        placeholder="Describe the bug you found...",
                    )
                    create_issue_btn = gr.Button("🐛 Submit Issue", variant="secondary")

                with gr.Accordion("🔧 Suggest Fix", open=False):
                    fix_input = gr.Textbox(
                        label="Fixed Source Code",
                        lines=12,
                        placeholder="Paste the complete fixed source code here...",
                        elem_classes=["code-display"],
                    )
                    suggest_fix_btn = gr.Button("🔧 Submit Fix", variant="secondary")

                with gr.Accordion("✅ Finalize", open=False):
                    message_input = gr.Textbox(
                        label="Final Notes",
                        lines=2,
                        placeholder="Summary of changes...",
                    )
                    finalize_btn = gr.Button("✅ Request Changes", variant="primary")

                # ── Wire up interactive callbacks ──────────────
                reset_btn.click(
                    interactive_reset,
                    inputs=[task_select, seed_input],
                    outputs=[code_display, tests_display, exec_output, test_results, feedback, status_bar, reward_bar],
                )
                run_code_btn.click(
                    interactive_run_code,
                    inputs=[code_input],
                    outputs=[exec_output, test_results, feedback, status_bar, reward_bar],
                )
                run_tests_btn.click(
                    interactive_run_tests,
                    inputs=[],
                    outputs=[exec_output, test_results, feedback, status_bar, reward_bar],
                )
                create_issue_btn.click(
                    interactive_create_issue,
                    inputs=[issue_input],
                    outputs=[exec_output, test_results, feedback, status_bar, reward_bar],
                )
                suggest_fix_btn.click(
                    interactive_suggest_fix,
                    inputs=[fix_input],
                    outputs=[code_display, exec_output, test_results, feedback, status_bar, reward_bar],
                )
                finalize_btn.click(
                    interactive_request_changes,
                    inputs=[message_input],
                    outputs=[exec_output, test_results, feedback, status_bar, reward_bar],
                )

            # ── Tab 2: Agent Demo Mode ─────────────────────────
            with gr.Tab("🤖 Agent Demo"):
                gr.Markdown("### Watch the baseline agent debug code step-by-step")

                with gr.Row():
                    agent_task = gr.Dropdown(
                        choices=["easy_debug", "medium_debug", "hard_debug"],
                        value="easy_debug",
                        label="Task",
                    )
                    agent_seed = gr.Textbox(
                        value="42", label="Seed",
                        max_lines=1,
                    )
                    agent_run_btn = gr.Button("🚀 Run Agent", variant="primary")

                with gr.Row():
                    with gr.Column(scale=2):
                        agent_log = gr.Textbox(
                            label="📋 Agent Action Log",
                            lines=30,
                            interactive=False,
                            elem_classes=["code-display"],
                        )
                    with gr.Column(scale=1):
                        agent_rewards = gr.Textbox(
                            label="📈 Reward Progression",
                            interactive=False,
                        )
                        agent_score = gr.Textbox(
                            label="🏆 Final Score",
                            interactive=False,
                        )

                agent_run_btn.click(
                    run_agent_demo,
                    inputs=[agent_task, agent_seed],
                    outputs=[agent_log, agent_rewards, agent_score],
                )

            # ── Tab 3: GitHub PR Mode ─────────────────────────
            with gr.Tab("🐙 GitHub PR Mode"):
                gr.Markdown(
                    """
                    ### Debug a real GitHub Pull Request
                    Select a curated Zero-Dependency PR below, or paste any GitHub PR URL to let the AI agent analyze and debug real code.
                    """
                )

                with gr.Row():
                    pr_url_input = gr.Dropdown(
                        label="Curated Zero-Dependency PRs (Or paste your own URL)",
                        choices=[
                            "https://github.com/pallets/click/pull/2224",
                            "https://github.com/pallets/flask/pull/4496",
                            "https://github.com/pydantic/pydantic/pull/5730",
                            "https://github.com/marshmallow-code/marshmallow/pull/2069"
                        ],
                        allow_custom_value=True,
                        value="https://github.com/pallets/click/pull/2224",
                        scale=3,
                    )
                    pr_fetch_btn = gr.Button("Fetch PR", variant="primary", scale=1)

                pr_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    with gr.Column(scale=1):
                        pr_info = gr.Textbox(
                            label="PR Info",
                            lines=6, interactive=False,
                        )
                        pr_diff = gr.Textbox(
                            label="PR Diff",
                            lines=12, interactive=False,
                            elem_classes=["code-display"],
                        )
                    with gr.Column(scale=1):
                        pr_code = gr.Textbox(
                            label="Source Code",
                            lines=15, interactive=False,
                            elem_classes=["code-display"],
                        )
                        pr_tests_display = gr.Textbox(
                            label="Test Files",
                            lines=8, interactive=False,
                            elem_classes=["code-display"],
                        )

                gr.Markdown("### Actions")
                with gr.Row():
                    pr_run_tests_btn = gr.Button("Run Tests", variant="secondary")
                    pr_run_agent_btn = gr.Button("Auto-Debug with AI", variant="primary")

                with gr.Accordion("Run Code Snippet", open=False):
                    pr_code_input = gr.Textbox(
                        label="Code to run",
                        lines=4,
                        elem_classes=["code-display"],
                    )
                    pr_run_code_btn = gr.Button("Run Code", variant="secondary")

                with gr.Accordion("Submit Fix", open=False):
                    pr_fix_input = gr.Textbox(
                        label="Fixed source code",
                        lines=10,
                        elem_classes=["code-display"],
                    )
                    pr_fix_btn = gr.Button("Submit Fix", variant="secondary")

                pr_output = gr.Textbox(
                    label="Output / Agent Log",
                    lines=15, interactive=False,
                    elem_classes=["code-display"],
                )
                pr_score = gr.Textbox(
                    label="Score", interactive=False,
                )

                # ── PR Mode State & Callbacks ─────────────────
                pr_env_ref = [None]  # mutable container

                def pr_fetch(url):
                    try:
                        from .pr_environment import PREnvironment
                        env = PREnvironment()
                        obs = env.reset(pr_url=url)
                        pr_env_ref[0] = env

                        # Format test files summary
                        test_names = []
                        for fname in list(env._test_sources.keys())[:8]:
                            test_names.append(fname)
                        tests_summary = "\n".join(test_names) if test_names else "No test files found"

                        return (
                            f"PR fetched successfully!",
                            obs.action_feedback,
                            obs.execution_output,  # diff
                            obs.code,
                            tests_summary,
                            "",
                            "",
                        )
                    except Exception as e:
                        return (
                            f"Error: {e}",
                            "", "", "", "", "", "",
                        )

                def pr_run_tests():
                    env = pr_env_ref[0]
                    if env is None:
                        return "Fetch a PR first!", ""
                    obs = env.step({"action_type": "run_tests"})
                    return obs.test_results or obs.execution_output, obs.action_feedback

                def pr_run_code(code):
                    env = pr_env_ref[0]
                    if env is None:
                        return "Fetch a PR first!", ""
                    obs = env.step({"action_type": "run_code", "code": code})
                    return obs.execution_output, obs.action_feedback

                def pr_submit_fix(fix_code):
                    env = pr_env_ref[0]
                    if env is None:
                        return "Fetch a PR first!", ""
                    obs = env.step({"action_type": "suggest_fix", "patch_code": fix_code})
                    return obs.test_results or obs.execution_output, obs.action_feedback

                def pr_run_llm_agent(url):
                    """Run the LLM agent against a real PR."""
                    try:
                        from .pr_environment import PREnvironment
                        from agent.llm_agent import LLMAgent

                        env = PREnvironment()
                        obs = env.reset(pr_url=url)
                        pr_env_ref[0] = env

                        agent = LLMAgent()
                        agent.reset()

                        log_lines = []
                        log_lines.append(f"PR: {obs.action_feedback}")
                        log_lines.append(f"Model: {os.environ.get('MODEL_NAME', 'qwen2.5-coder:7b')}")
                        log_lines.append("")

                        step = 0
                        print(f"Starting auto-debug on PR #{env._pr.pr_number}...")
                        while not obs.done and step < 15:
                            print(f"[Step {step+1}/15] Agent is thinking...")
                            action = agent.act(obs.model_dump())
                            
                            atype = action.get("action_type", "?")
                            print(f"[Step {step+1}/15] Action: {atype}")
                            
                            obs = env.step(action)
                            step += 1
                            log_lines.append(f"Step {step}: {atype} -> reward={obs.reward:+.2f}")
                            if atype == "run_code":
                                log_lines.append(f"  code: {action.get('code', '')[:80]}")
                            elif atype == "create_issue":
                                log_lines.append(f"  issue: {action.get('issue_description', '')[:120]}")
                            elif atype == "suggest_fix":
                                log_lines.append(f"  fix: {len(action.get('patch_code', ''))} chars")
                                if obs.tests_passed:
                                    log_lines.append("  >>> ALL TESTS PASS!")
                            log_lines.append("")

                        # Auto-finalize
                        if not obs.done:
                            obs = env.step({"action_type": "request_changes", "message": "Auto-debug complete."})

                        state = env.state
                        log_lines.append(f"\n{'='*40}")
                        log_lines.append(f"Score: {state.grader_score:.2f}")
                        log_lines.append(f"Cumulative reward: {state.cumulative_reward:.2f}")
                        log_lines.append(f"Fixes: {state.fixes_passed}/{state.fixes_attempted}")

                        return "\n".join(log_lines), f"Score: {state.grader_score:.2f}"

                    except Exception as e:
                        import traceback
                        return f"Error: {e}\n{traceback.format_exc()}", "Error"

                # ── Wire up PR callbacks ──────────────────────
                pr_fetch_btn.click(
                    pr_fetch,
                    inputs=[pr_url_input],
                    outputs=[pr_status, pr_info, pr_diff, pr_code, pr_tests_display, pr_output, pr_score],
                )
                pr_run_tests_btn.click(
                    pr_run_tests,
                    inputs=[],
                    outputs=[pr_output, pr_score],
                )
                pr_run_code_btn.click(
                    pr_run_code,
                    inputs=[pr_code_input],
                    outputs=[pr_output, pr_score],
                )
                pr_fix_btn.click(
                    pr_submit_fix,
                    inputs=[pr_fix_input],
                    outputs=[pr_output, pr_score],
                )
                pr_run_agent_btn.click(
                    pr_run_llm_agent,
                    inputs=[pr_url_input],
                    outputs=[pr_output, pr_score],
                )

            # ── Tab 4: About ──────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    """
                    ## NitpickAI — Interactive Debugging Benchmark

                    ### Action Space
                    | Action | Description | Reward |
                    |--------|-------------|--------|
                    | `run_code` | Execute a code snippet in sandbox | +0.05 (useful output) |
                    | `run_tests` | Run visible test suite | +0.05 (reveals failures) |
                    | `create_issue` | Describe the identified bug | +0.3 (accurate) / -0.2 (poor) |
                    | `suggest_fix` | Submit patched source code | +0.5 (all pass) / -0.3 (fail) |
                    | `request_changes` | Finalize the session | +1.0 (after fix) / -0.3 (no fix) |

                    ### Grading Formula
                    ```
                    score = 0.25 × issue_accuracy
                          + 0.15 × code_execution_quality
                          + 0.40 × fix_quality
                          + 0.10 × efficiency
                          + 0.10 × decision_quality
                    ```

                    ### Tasks
                    - **easy_debug**: Obvious bugs (off-by-one, missing return, wrong operator)
                    - **medium_debug**: Requires execution (type coercion, mutable defaults, missing keys)
                    - **hard_debug**: Multi-step (closures, generator exhaustion, cache corruption)

                    ### OpenEnv Compliance
                    - REST API: `POST /reset`, `POST /step/{id}`, `GET /state/{id}`
                    - WebSocket: `/ws`
                    - Schema: `GET /schema`

                    Built for the Meta × PyTorch × Hugging Face Hackathon.
                    """
                )

    return demo


# ── Mount Gradio ─────────────────────────────────────────────────

try:
    import gradio as gr
    gradio_app = build_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/")
except ImportError:
    # Gradio not installed — serve a simple HTML page instead
    @app.get("/")
    async def root():
        return HTMLResponse(
            "<h1>NitpickAI</h1>"
            "<p>Interactive Debugging Benchmark for AI Agents.</p>"
            "<p>Install gradio for the full UI: <code>pip install gradio</code></p>"
            "<p>Connect via REST API at <code>/reset</code>, <code>/step/{id}</code>, <code>/state/{id}</code></p>"
        )


def main():
    """Entry point for ``uv run server`` or ``python -m server.app``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
