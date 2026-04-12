"""
NitpickAI Client.

Thin wrapper around the OpenEnv EnvClient.  When openenv-core is not
installed the module defines a lightweight HTTP-based client instead.
"""

from __future__ import annotations

try:
    from openenv.core.mcp_client import MCPToolClient  # type: ignore[import-not-found]  # noqa: F401

    class NitpickEnv(MCPToolClient):  # type: ignore[misc]
        """Client for connecting to a NitpickAI server."""

        pass  # MCPToolClient provides reset / step / state / call_tool

except ImportError:  # openenv-core not installed → use standalone HTTP client
    # ── Standalone HTTP client fallback ───────────────────────────
    import json
    from typing import Any

    import requests

    class NitpickEnv:  # type: ignore[no-redef]
        """Minimal sync HTTP client for NitpickAI (no openenv-core)."""

        def __init__(self, base_url: str = "http://localhost:7860") -> None:
            self.base_url = base_url.rstrip("/")
            self.session_id: str | None = None
            self._http = requests.Session()

        def reset(self, **kwargs: Any) -> dict[str, Any]:
            resp = self._http.post(f"{self.base_url}/reset", json=kwargs)
            resp.raise_for_status()
            data = resp.json()
            self.session_id = data.get("session_id")
            return data.get("observation", data)

        def step(self, action: dict[str, Any] | Any) -> dict[str, Any]:
            if not isinstance(action, dict):
                action = action.model_dump()
            assert self.session_id, "Call reset() first"
            resp = self._http.post(
                f"{self.base_url}/step/{self.session_id}", json=action
            )
            resp.raise_for_status()
            return resp.json()

        def state(self) -> dict[str, Any]:
            assert self.session_id, "Call reset() first"
            resp = self._http.get(f"{self.base_url}/state/{self.session_id}")
            resp.raise_for_status()
            return resp.json()

        def close(self) -> None:
            self._http.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
