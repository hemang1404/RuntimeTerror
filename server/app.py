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
