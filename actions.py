from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

ActionName = Literal[
    "move",
    "move2",
    "rotation",
    "pick_up",
    "put_down",
    "open",
    "close",
    "turn_on",
    "turn_off",
]

Direction = Literal["forward", "backward", "left", "right"]


@dataclass(frozen=True, slots=True)
class Move:
    name: Literal["move"] = "move"
    direction: Direction = "forward"
    distance: float = 1.0


@dataclass(frozen=True, slots=True)
class MoveToObject:
    name: Literal["move2"] = "move2"
    target: str = ""


@dataclass(frozen=True, slots=True)
class Rotate:
    name: Literal["rotation"] = "rotation"
    yaw_deg: float = 120.0
    times: int = 3


@dataclass(frozen=True, slots=True)
class PickUp:
    name: Literal["pick_up"] = "pick_up"
    target: str = ""


@dataclass(frozen=True, slots=True)
class PutDown:
    name: Literal["put_down"] = "put_down"
    object_id: str = ""  # empty => drop last item


@dataclass(frozen=True, slots=True)
class Open:
    name: Literal["open"] = "open"
    target: str = ""


@dataclass(frozen=True, slots=True)
class Close:
    name: Literal["close"] = "close"
    target: str = ""


@dataclass(frozen=True, slots=True)
class TurnOn:
    name: Literal["turn_on"] = "turn_on"
    target: str = ""


@dataclass(frozen=True, slots=True)
class TurnOff:
    name: Literal["turn_off"] = "turn_off"
    target: str = ""


Action = Union[Move, MoveToObject, Rotate, PickUp, PutDown, Open, Close, TurnOn, TurnOff]

