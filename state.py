from __future__ import annotations

from dataclasses import dataclass, field

from .vec import Vec3


@dataclass(slots=True)
class AgentState:
    agent_id: str = "agent"
    position: Vec3 = field(default_factory=Vec3)
    yaw_deg: float = 0.0
    inventory_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ObjectState:
    object_id: str
    name: str = ""
    color: str = ""
    position: Vec3 = field(default_factory=Vec3)
    extent: Vec3 = field(default_factory=lambda: Vec3(10.0, 10.0, 10.0))

    pickable: bool = True
    openable: bool = False
    powerable: bool = False

    is_open: bool = False
    is_on: bool = False

    held_by: str | None = None

    def aabb(self) -> dict[str, Vec3]:
        return {"min": self.position - self.extent, "max": self.position + self.extent}

