"""
tongsim_offline

Dependency-free offline simulator with a Meta-action API:
  move / move2 / rotation / pick_up / put_down / open / close / turn_on / turn_off

Designed to mirror the "perception→reasoning→action loop" style:
  - `OfflineEnv.observe()` returns a JSON-serializable observation dict
  - `OfflineEnv.step(action)` applies an action and returns (obs, result)
"""

from .actions import (
    Action,
    ActionName,
    Close,
    Direction,
    Move,
    MoveToObject,
    Open,
    PickUp,
    PutDown,
    Rotate,
    TurnOff,
    TurnOn,
)
from .env import OfflineEnv, OfflineEnvConfig, Observation, StepResult
from .state import AgentState, ObjectState
from .vec import Vec3

__all__ = [
    "Action",
    "ActionName",
    "AgentState",
    "Close",
    "Direction",
    "Move",
    "MoveToObject",
    "ObjectState",
    "Observation",
    "OfflineEnv",
    "OfflineEnvConfig",
    "Open",
    "PickUp",
    "PutDown",
    "Rotate",
    "StepResult",
    "TurnOff",
    "TurnOn",
    "Vec3",
]

