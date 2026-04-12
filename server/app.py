"""
RuntimeTerror — FastAPI + Gradio Server Application.

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
        
    from agent.pr_environment import PREnvironment
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
        initial_action = agent.act(obs.model_dump())
        atype = initial_action.get("action_type", "unknown")
        
        if atype == "suggest_fix":
            patch = initial_action.get("patch_code", "")
            # Auto-apply the patch in the backend to grab test results
            test_obs = env.step({"action_type": "suggest_fix", "patch_code": patch})
            
            passed = test_obs.tests_passed
            tests_summary = "✅ All tests successfully passed." if passed else "❌ Tests failed with this patch."
            
            review_comment = {
                "issue_found": "Identified logical defect based on context execution and trace evaluation.",
                "suggested_fix": patch,
                "test_results": f"{tests_summary}\n\nLog:\n{test_obs.test_results}"
            }
            return JSONResponse({"suggestion": review_comment, "explanation": "AI generated a Code Review."})
            
        elif atype == "create_issue":
            review_comment = {
                "issue_found": initial_action.get("issue_description", "No explicit issue description."),
                "suggested_fix": "No patch available yet.",
                "test_results": "⚠️ N/A - requires exploratory fixes."
            }
            return JSONResponse({"suggestion": review_comment, "explanation": "AI identified an issue without a patch."})
        else:
            # Need exploratory execution? We simulate doing it internally or just return the output
            exec_obs = env.step(initial_action)
            return JSONResponse({"suggestion": None, "explanation": f"AI needs more context. It ran: {atype}. Internal state advanced."})
            
    except Exception as e:
        import traceback
        return JSONResponse({"suggestion": None, "explanation": f"AI error: {str(e)}"}, status_code=500)


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
