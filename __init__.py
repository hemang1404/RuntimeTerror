"""
IncidentEnv — AI on-call incident response simulation for OpenEnv.

An agent receives a production alert, investigates using diagnostic tools,
identifies the root cause, and submits a code fix validated by test execution.
"""

from .models import IncidentAction, IncidentObservation, IncidentState

try:
    from .client import IncidentEnv
except ImportError:
    IncidentEnv = None  # type: ignore[assignment, misc]

__all__ = [
    "IncidentAction",
    "IncidentObservation",
    "IncidentState",
    "IncidentEnv",
]
