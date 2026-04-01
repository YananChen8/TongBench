from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Vec3:
    """Minimal 3D vector for offline simulation (XY plane + Z for height)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        s = float(scalar)
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def xy_norm(self) -> float:
        return math.hypot(self.x, self.y)

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z)}

    @staticmethod
    def from_any(value: Any) -> "Vec3":
        if isinstance(value, Vec3):
            return value
        if isinstance(value, dict):
            return Vec3(
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            )
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            return Vec3(float(value[0]), float(value[1]), float(value[2]))
        raise TypeError(f"Cannot convert to Vec3: {type(value)}")


def yaw_to_forward_right(yaw_deg: float) -> tuple[Vec3, Vec3]:
    """UE-like convention: +X forward, +Y right, yaw around +Z."""
    rad = math.radians(float(yaw_deg))
    forward = Vec3(math.cos(rad), math.sin(rad), 0.0)
    right = Vec3(-math.sin(rad), math.cos(rad), 0.0)
    return forward, right


def angle_deg(a: Vec3, b: Vec3) -> float:
    """Angle between two XY vectors in degrees."""
    da = a.xy_norm()
    db = b.xy_norm()
    if da <= 1e-9 or db <= 1e-9:
        return 0.0 if db <= 1e-9 else 180.0
    cosv = (a.x * b.x + a.y * b.y) / (da * db)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def wrap_degrees(value: float) -> float:
    v = float(value) % 360.0
    return v + 360.0 if v < 0 else v


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

