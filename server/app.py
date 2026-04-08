"""
IncidentEnv — FastAPI Server Application.

Creates the HTTP + WebSocket server for the incident environment, compatible
with the OpenEnv client ecosystem (MCPToolClient, EnvClient, etc.).
"""

from __future__ import annotations

import os

# Support both in-repo (openenv) and standalone imports.
# The openenv-core HTTP server can run without per-client session isolation,
# which breaks this environment's stateful reset->step lifecycle on Spaces.
# Default to the explicit session-based fallback unless opted in.
USE_OPENENV_CORE_SERVER = os.getenv("USE_OPENENV_CORE_SERVER", "0") == "1"

if USE_OPENENV_CORE_SERVER:
    from openenv.core.env_server.http_server import create_app

    from .incident_environment import IncidentEnvironment
    from models import IncidentAction, IncidentObservation

    app = create_app(
        IncidentEnvironment,
        IncidentAction,
        IncidentObservation,
        env_name="incident_env",
    )

else:
    # ── Standalone FastAPI fallback (works without openenv-core) ──
    import json
    import uuid
    from typing import Any

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, HTMLResponse

    from .incident_environment import IncidentEnvironment

    app = FastAPI(title="IncidentEnv", version="0.1.0")

    # Session storage (one env per WebSocket session)
    _sessions: dict[str, IncidentEnvironment] = {}

    # ── HTTP endpoints ───────────────────────────────────────────

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "healthy"})

    @app.get("/")
    async def root():
        return HTMLResponse(
            "<h1>IncidentEnv</h1>"
            "<p>AI on-call incident response simulation.</p>"
            "<p>Connect via WebSocket at <code>/ws</code></p>"
        )

    @app.get("/schema")
    async def schema():
        from models import IncidentAction, IncidentObservation, IncidentState
        return JSONResponse({
            "action": IncidentAction.model_json_schema(),
            "observation": IncidentObservation.model_json_schema(),
            "state": IncidentState.model_json_schema(),
        })

    # ── REST endpoints for simple clients ────────────────────────

    @app.post("/reset")
    async def reset_endpoint(body: dict[str, Any] | None = None):
        body = body or {}
        env = IncidentEnvironment()
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

    # ── WebSocket endpoint ───────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        env = IncidentEnvironment()
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


def main():
    """Entry point for ``uv run server`` or ``python -m server.app``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
