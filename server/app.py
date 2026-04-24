"""
RuntimeTerror — FastAPI + Gradio Server Application.

Creates the HTTP server for the debugging environment, compatible with
the OpenEnv client ecosystem. Includes a Gradio-based UI with:
  1. Interactive mode — humans can manually debug code
  2. Agent demo mode — watch the baseline agent step through a task
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse

from .debug_environment import DebugEnvironment

app = FastAPI(title="RuntimeTerror", version="0.1.0")

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


@app.post("/pr/fetch")
async def pr_fetch_endpoint(body: dict[str, Any] | None = None):
    body = body or {}
    pr_url = body.get("url")
    if not pr_url:
        return JSONResponse({"error": "Missing 'url' parameter"}, status_code=400)
        
    from .pr_environment import PREnvironment
    env = PREnvironment()
    session_id = str(uuid.uuid4())
    _sessions[session_id] = env
    try:
        obs = env.reset(pr_url=pr_url)
        return JSONResponse({
            "session_id": session_id,
            "observation": obs.model_dump(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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


@app.post("/ai_suggest/{session_id}")
async def ai_suggest_endpoint(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        return JSONResponse({"error": "Unknown session"}, status_code=404)
        
    last_output = getattr(env._state, "last_output", "") if hasattr(env, "_state") and hasattr(env._state, "last_output") else ""
    obs = env._make_observation(
        execution_output=last_output,
        test_results="",
        reward=0.0,
        done=False,
        action_feedback="AI compiling PR Review..."
    )
    
    from agent.llm_agent import LLMAgent
    try:
        agent = LLMAgent()
        max_internal_steps = 10
        curr_obs_dict = obs.model_dump()
        
        for i in range(max_internal_steps):
            action = agent.act(curr_obs_dict)
            
            # Map common hallucinations to correct action types
            atype = action.get("action_type", "unknown")
            if atype in ["fix_code", "patch_code", "submit_fix", "submit_pr", "submit_patch", "apply_fix", "fix", "patch", "submit"]:
                atype = "suggest_fix"
            elif atype in ["finish", "done", "complete", "end", "finalize"]:
                atype = "request_changes"
            
            if atype == "suggest_fix":
                patch = action.get("patch_code", "")
                if not patch and "fix" in action:
                    patch = action["fix"]
                
                # VALIDATION: Reject empty or tiny patches
                if not patch or len(patch.strip()) < 40:
                    curr_obs_dict["action_feedback"] = "ERROR: Your suggest_fix was empty or incomplete. You MUST provide the ENTIRE fixed Python source code for the file in 'patch_code'."
                    continue
                
                test_obs = env.step({"action_type": "suggest_fix", "patch_code": patch})
                passed = test_obs.tests_passed
                
                # If tests fail, give the AI another chance in the loop
                if not passed and i < max_internal_steps - 1:
                    curr_obs_dict = test_obs.model_dump()
                    curr_obs_dict["action_feedback"] = f"CRITICAL: Your last fix FAILED the tests. Regression detected. Please analyze the failure logs below and provide a NEW, corrected version of the full file.\n\nLogs:\n{test_obs.test_results}"
                    continue

                tests_summary = "✅ All tests successfully passed." if passed else "❌ Tests failed with this patch."
                issue_desc = env._state.issue_submitted if hasattr(env._state, "issue_submitted") and env._state.issue_submitted else "Identified logical defect based on automated codebase exploration."
                
                review_comment = {
                    "issue_found": issue_desc,
                    "suggested_fix": patch,
                    "test_results": f"{tests_summary}\n\nLog:\n{test_obs.test_results}",
                    "passed": passed
                }
                return JSONResponse({"suggestion": review_comment, "explanation": f"AI synthesized a fix after {i+1} steps of analysis."})
                
            elif atype in ["run_code", "run_tests", "create_issue"]:
                # Execute exploratory action and continue loop
                obs_obj = env.step(action)
                curr_obs_dict = obs_obj.model_dump()
                continue
            elif atype == "request_changes":
                return JSONResponse({"suggestion": None, "explanation": "AI ended session without providing a code patch."})
            else:
                return JSONResponse({"suggestion": None, "explanation": f"AI requested unknown action: {atype}"})

                
        return JSONResponse({"suggestion": None, "explanation": f"AI reached maximum exploration depth ({max_internal_steps} steps) without a conclusive fix."})
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"suggestion": None, "explanation": f"AI error: {str(e)}"}, status_code=500)


# ── SSE Streaming AI Analysis ────────────────────────────────────

@app.get("/ai_suggest_stream/{session_id}")
async def ai_suggest_stream_endpoint(session_id: str):
    """Stream AI analysis steps as Server-Sent Events.

    Each event has the format:
        data: {"step": 1, "action": "run_tests", "feedback": "...", "done": false}

    The final event contains the full suggestion:
        data: {"done": true, "suggestion": {...}, "explanation": "..."}
    """
    env = _sessions.get(session_id)
    if env is None:
        return JSONResponse({"error": "Unknown session"}, status_code=404)

    event_queue: queue.Queue = queue.Queue()

    def _run_agent_loop():
        """Execute the LLM agent loop in a background thread, pushing events to the queue."""
        try:
            last_output = getattr(env._state, "last_output", "") if hasattr(env, "_state") and hasattr(env._state, "last_output") else ""
            obs = env._make_observation(
                execution_output=last_output,
                test_results="",
                reward=0.0,
                done=False,
                action_feedback="AI compiling PR Review..."
            )

            from agent.llm_agent import LLMAgent
            agent = LLMAgent()
            max_internal_steps = 10
            curr_obs_dict = obs.model_dump()

            event_queue.put({
                "step": 0,
                "action": "init",
                "feedback": "AI agent initialized. Starting analysis...",
                "done": False,
            })

            for i in range(max_internal_steps):
                step_num = i + 1
                t0 = time.time()
                action = agent.act(curr_obs_dict)
                elapsed = round(time.time() - t0, 1)

                # Map common hallucinations to correct action types
                atype = action.get("action_type", "unknown")
                if atype in ["fix_code", "patch_code", "submit_fix", "submit_pr", "submit_patch", "apply_fix", "fix", "patch", "submit"]:
                    atype = "suggest_fix"
                elif atype in ["finish", "done", "complete", "end", "finalize"]:
                    atype = "request_changes"

                if atype == "suggest_fix":
                    patch = action.get("patch_code", "")
                    if not patch and "fix" in action:
                        patch = action["fix"]

                    # VALIDATION: Reject empty or tiny patches
                    if not patch or len(patch.strip()) < 40:
                        event_queue.put({
                            "step": step_num,
                            "action": "suggest_fix",
                            "feedback": f"⚠️ Fix rejected (too short). Retrying... ({elapsed}s)",
                            "done": False,
                        })
                        curr_obs_dict["action_feedback"] = "ERROR: Your suggest_fix was empty or incomplete. You MUST provide the ENTIRE fixed Python source code for the file in 'patch_code'."
                        continue

                    event_queue.put({
                        "step": step_num,
                        "action": "suggest_fix",
                        "feedback": f"🔧 Applying fix ({len(patch)} chars) and running tests... ({elapsed}s)",
                        "done": False,
                    })

                    test_obs = env.step({"action_type": "suggest_fix", "patch_code": patch})
                    passed = test_obs.tests_passed

                    # If tests fail, give the AI another chance in the loop
                    if not passed and i < max_internal_steps - 1:
                        event_queue.put({
                            "step": step_num,
                            "action": "fix_failed",
                            "feedback": "❌ Fix failed tests. Retrying with new approach...",
                            "done": False,
                        })
                        curr_obs_dict = test_obs.model_dump()
                        curr_obs_dict["action_feedback"] = f"CRITICAL: Your last fix FAILED the tests. Regression detected. Please analyze the failure logs below and provide a NEW, corrected version of the full file.\n\nLogs:\n{test_obs.test_results}"
                        continue

                    tests_summary = "✅ All tests successfully passed." if passed else "❌ Tests failed with this patch."
                    issue_desc = env._state.issue_submitted if hasattr(env._state, "issue_submitted") and env._state.issue_submitted else "Identified logical defect based on automated codebase exploration."

                    review_comment = {
                        "issue_found": issue_desc,
                        "suggested_fix": patch,
                        "test_results": f"{tests_summary}\n\nLog:\n{test_obs.test_results}",
                        "passed": passed,
                    }
                    event_queue.put({
                        "done": True,
                        "suggestion": review_comment,
                        "explanation": f"AI synthesized a fix after {step_num} steps of analysis.",
                    })
                    return

                elif atype in ["run_code", "run_tests", "create_issue"]:
                    label = {
                        "run_code": "🔍 Running code snippet to investigate...",
                        "run_tests": "🧪 Running test suite...",
                        "create_issue": "📝 Documenting identified bug...",
                    }[atype]
                    event_queue.put({
                        "step": step_num,
                        "action": atype,
                        "feedback": f"{label} ({elapsed}s)",
                        "done": False,
                    })
                    obs_obj = env.step(action)
                    curr_obs_dict = obs_obj.model_dump()
                    continue

                elif atype == "request_changes":
                    event_queue.put({
                        "done": True,
                        "suggestion": None,
                        "explanation": "AI ended session without providing a code patch.",
                    })
                    return
                else:
                    event_queue.put({
                        "done": True,
                        "suggestion": None,
                        "explanation": f"AI requested unknown action: {atype}",
                    })
                    return

            # Max steps reached
            event_queue.put({
                "done": True,
                "suggestion": None,
                "explanation": f"AI reached maximum exploration depth ({max_internal_steps} steps) without a conclusive fix.",
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            event_queue.put({
                "done": True,
                "suggestion": None,
                "explanation": f"AI error: {str(e)}",
            })

    # Start the agent loop in a background thread
    thread = threading.Thread(target=_run_agent_loop, daemon=True)
    thread.start()

    async def event_stream():
        """Async generator that yields SSE events from the queue."""
        while True:
            try:
                data = await asyncio.to_thread(event_queue.get, timeout=180)
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("done"):
                    break
            except Exception:
                # Timeout — send keepalive and continue
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/pr/post_comment/{session_id}")
async def pr_post_comment_endpoint(session_id: str, body: dict[str, Any]):
    env = _sessions.get(session_id)
    if env is None or env._pr is None:
        return JSONResponse({"error": "Unknown or invalid PR session."})
        
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        # Return error if no token is available to authenticate the API request
        return JSONResponse({"error": "GITHUB_TOKEN environment variable is not set. Please set it in your terminal before starting the server."})
        
    comment_body = body.get("comment", "")
    
    owner = env._pr.owner
    repo = env._pr.repo
    pr_num = env._pr.pr_number
    
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_num}/comments"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    import requests
    resp = requests.post(url, headers=headers, json={"body": comment_body})
    
    if resp.status_code == 201:
        return JSONResponse({"success": True})
    else:
        return JSONResponse({"error": f"GitHub API error: {resp.text}"})

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


# ── Frontend HTML UI ──────────────────────────────────────────────

@app.get("/")
async def root():
    import os
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>RuntimeTerror</h1>"
            "<p>Error: templates/index.html not found. Please ensure the frontend bundle is built.</p>",
            status_code=404
        )

def main():
    """Entry point for ``uv run server`` or ``python -m server.app``."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
