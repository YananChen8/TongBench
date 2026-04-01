from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .actions import Action, Close, Direction, Move, MoveToObject, Open, PickUp, PutDown, Rotate, TurnOff, TurnOn
from .state import AgentState, ObjectState
from .vec import Vec3, angle_deg, clamp, wrap_degrees, yaw_to_forward_right


@dataclass(frozen=True, slots=True)
class OfflineEnvConfig:
    fov_deg: float = 90.0
    view_range: float = 500.0
    interact_range: float = 120.0
    drop_distance: float = 60.0
    carry_offset: Vec3 = Vec3(30.0, 0.0, 50.0)
    world_min: Vec3 = Vec3(-1000.0, -1000.0, -1000.0)
    world_max: Vec3 = Vec3(1000.0, 1000.0, 1000.0)


@dataclass(frozen=True, slots=True)
class StepResult:
    success: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class Observation:
    agent_initial_visible_ids: list[str]
    inventory_ids: list[str]
    agent_position: dict[str, float]
    agent_external_state: dict[str, Any]
    visible_object_ids: list[str]
    nearby_operable_ids: list[str]
    scene_description: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_initial_visible_ids": list(self.agent_initial_visible_ids),
            "inventory_ids": list(self.inventory_ids),
            "agent_position": dict(self.agent_position),
            "agent_external_state": dict(self.agent_external_state),
            "visible_object_ids": list(self.visible_object_ids),
            "nearby_operable_ids": list(self.nearby_operable_ids),
            "scene_description": list(self.scene_description),
        }


class OfflineEnv:
    def __init__(self, config: OfflineEnvConfig | None = None):
        self.config = config or OfflineEnvConfig()
        self.agent = AgentState()
        self.objects: dict[str, ObjectState] = {}
        self._initial_visible_ids: list[str] = []

    def reset(self, *, agent: AgentState | None = None, objects: list[ObjectState] | None = None) -> Observation:
        self.agent = agent or AgentState()
        self.objects = {o.object_id: o for o in (objects or [])}
        self._initial_visible_ids = self._visible_ids()
        return self.observe()

    # -----
    # View
    # -----

    def _in_range_xy(self, pos: Vec3, rng: float) -> bool:
        d = pos - self.agent.position
        return math.hypot(d.x, d.y) <= float(rng)

    def _is_in_view(self, obj: ObjectState) -> bool:
        if obj.held_by is not None:
            return False
        if not self._in_range_xy(obj.position, self.config.view_range):
            return False
        forward, _right = yaw_to_forward_right(self.agent.yaw_deg)
        return angle_deg(forward, obj.position - self.agent.position) <= (self.config.fov_deg / 2.0)

    def _visible_ids(self) -> list[str]:
        return [o.object_id for o in self.objects.values() if self._is_in_view(o)]

    def _nearby_operable_ids(self) -> list[str]:
        out: list[str] = []
        for o in self.objects.values():
            if o.held_by is not None:
                continue
            if not (o.openable or o.powerable):
                continue
            if self._in_range_xy(o.position, self.config.interact_range):
                out.append(o.object_id)
        return out

    def observe(self) -> Observation:
        visible_ids = self._visible_ids()
        nearby_operables = self._nearby_operable_ids()
        scene_desc: list[dict[str, Any]] = []
        for oid in visible_ids:
            o = self.objects[oid]
            aabb = o.aabb()
            scene_desc.append(
                {
                    "id": o.object_id,
                    "name": o.name,
                    "color": o.color,
                    "position": o.position.to_dict(),
                    "aabb": {"min": aabb["min"].to_dict(), "max": aabb["max"].to_dict()},
                    "pickable": bool(o.pickable),
                    "openable": bool(o.openable),
                    "powerable": bool(o.powerable),
                    "is_open": bool(o.is_open),
                    "is_on": bool(o.is_on),
                }
            )
        return Observation(
            agent_initial_visible_ids=list(self._initial_visible_ids),
            inventory_ids=list(self.agent.inventory_ids),
            agent_position=self.agent.position.to_dict(),
            agent_external_state={"position": self.agent.position.to_dict(), "yaw_deg": float(self.agent.yaw_deg)},
            visible_object_ids=visible_ids,
            nearby_operable_ids=nearby_operables,
            scene_description=scene_desc,
        )

    # ----
    # Step
    # ----

    def step(self, action: Action) -> tuple[Observation, StepResult]:
        result = self.apply(action)
        return self.observe(), result

    def apply(self, action: Action) -> StepResult:
        if isinstance(action, Move):
            return self._act_move(action.direction, action.distance)
        if isinstance(action, MoveToObject):
            return self._act_move2(action.target)
        if isinstance(action, Rotate):
            return self._act_rotation(action.yaw_deg, action.times)
        if isinstance(action, PickUp):
            return self._act_pick_up(action.target)
        if isinstance(action, PutDown):
            return self._act_put_down(action.object_id)
        if isinstance(action, Open):
            return self._act_open_close(action.target, open_=True)
        if isinstance(action, Close):
            return self._act_open_close(action.target, open_=False)
        if isinstance(action, TurnOn):
            return self._act_power(action.target, on=True)
        if isinstance(action, TurnOff):
            return self._act_power(action.target, on=False)
        return StepResult(False, f"Unsupported action: {getattr(action, 'name', None)!r}")

    # --------------
    # Action helpers
    # --------------

    def _sync_held(self) -> None:
        if not self.agent.inventory_ids:
            return
        forward, right = yaw_to_forward_right(self.agent.yaw_deg)
        carry_offset = (
            forward * self.config.carry_offset.x
            + right * self.config.carry_offset.y
            + Vec3(0.0, 0.0, self.config.carry_offset.z)
        )
        for oid in self.agent.inventory_ids:
            o = self.objects.get(oid)
            if o and o.held_by == self.agent.agent_id:
                o.position = self.agent.position + carry_offset

    def _act_move(self, direction: Direction, distance: float) -> StepResult:
        if float(distance) <= 0:
            return StepResult(False, "distance must be > 0")
        forward, right = yaw_to_forward_right(self.agent.yaw_deg)
        if direction == "forward":
            delta = forward * float(distance)
        elif direction == "backward":
            delta = forward * (-float(distance))
        elif direction == "left":
            delta = right * (-float(distance))
        else:
            delta = right * float(distance)
        next_pos = self.agent.position + delta
        self.agent.position = Vec3(
            clamp(next_pos.x, self.config.world_min.x, self.config.world_max.x),
            clamp(next_pos.y, self.config.world_min.y, self.config.world_max.y),
            clamp(next_pos.z, self.config.world_min.z, self.config.world_max.z),
        )
        self._sync_held()
        return StepResult(True, "moved")

    def _act_move2(self, target_id: str) -> StepResult:
        o = self.objects.get(target_id)
        if o is None:
            return StepResult(False, "target not found")
        self.agent.position = Vec3(o.position.x, o.position.y, self.agent.position.z)
        self._sync_held()
        return StepResult(True, "moved_to_object")

    def _act_rotation(self, yaw_deg: float, times: int) -> StepResult:
        if int(times) <= 0:
            return StepResult(False, "times must be > 0")
        self.agent.yaw_deg = wrap_degrees(self.agent.yaw_deg + float(yaw_deg) * int(times))
        self._sync_held()
        return StepResult(True, "rotated")

    def _require_target(self, target_id: str) -> ObjectState | None:
        o = self.objects.get(target_id)
        if o is None:
            return None
        if not self._is_in_view(o):
            return None
        if not self._in_range_xy(o.position, self.config.interact_range):
            return None
        return o

    def _act_pick_up(self, target_id: str) -> StepResult:
        o = self._require_target(target_id)
        if o is None:
            return StepResult(False, "target not in view/range")
        if not o.pickable:
            return StepResult(False, "not pickable")
        if o.held_by is not None:
            return StepResult(False, "already held")
        o.held_by = self.agent.agent_id
        if o.object_id not in self.agent.inventory_ids:
            self.agent.inventory_ids.append(o.object_id)
        self._sync_held()
        return StepResult(True, "picked_up")

    def _act_put_down(self, object_id: str) -> StepResult:
        if not self.agent.inventory_ids:
            return StepResult(False, "inventory empty")
        oid = object_id or self.agent.inventory_ids[-1]
        if oid not in self.agent.inventory_ids:
            return StepResult(False, "object not in inventory")
        o = self.objects.get(oid)
        if o is None:
            return StepResult(False, "object missing")
        if o.held_by != self.agent.agent_id:
            return StepResult(False, "object not held")
        o.held_by = None
        self.agent.inventory_ids = [x for x in self.agent.inventory_ids if x != oid]
        forward, _right = yaw_to_forward_right(self.agent.yaw_deg)
        o.position = self.agent.position + forward * self.config.drop_distance
        return StepResult(True, "put_down")

    def _act_open_close(self, target_id: str, *, open_: bool) -> StepResult:
        o = self._require_target(target_id)
        if o is None:
            return StepResult(False, "target not in view/range")
        if not o.openable:
            return StepResult(False, "not openable")
        o.is_open = bool(open_)
        return StepResult(True, "opened" if open_ else "closed")

    def _act_power(self, target_id: str, *, on: bool) -> StepResult:
        o = self._require_target(target_id)
        if o is None:
            return StepResult(False, "target not in view/range")
        if not o.powerable:
            return StepResult(False, "not powerable")
        o.is_on = bool(on)
        return StepResult(True, "turned_on" if on else "turned_off")

