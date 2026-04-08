from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure local TongSim SDK in ./tongsim/src can be imported.
import sys

sys.path.insert(0, str((Path(__file__).resolve().parent / "tongsim" / "src").resolve()))

import tongsim as ts
from tongsim.math import Transform, Vector3, euler_to_quaternion, quaternion_to_euler
from tongsim.type.rl_demo import RLDemoHandType, RLDemoOrientationMode
from tongsim_lite_protobuf import capture_pb2


DEFAULT_CAPTURE_PARAMS: dict[str, Any] = {
    "width": 960,
    "height": 540,
    "fov_degrees": 90.0,
    "qps": 5.0,
    "enable_depth": False,
    "color_source": capture_pb2.CaptureColorSource.COLOR_SOURCE_FINAL_COLOR_LDR,
    "color_format": capture_pb2.CaptureRenderTargetFormat.COLOR_FORMAT_RGBA8,
    "rgb_codec": capture_pb2.CaptureRgbCodec.CAPTURE_RGB_CODEC_NONE,
    "jpeg_quality": 95,
}


@dataclass
class RunnerConfig:
    grpc_endpoint: str
    subtasks_json: Path
    output_dir: Path
    agent_id: str | None
    move_speed: float
    move_tolerance: float
    rotation_yaw_deg: float
    rotation_times: int
    fov_yaw_deg: float
    interact_distance: float
    interact_clearance: float
    use_navmesh: bool
    strict_navmesh: bool
    nav_accept_radius: float
    drop_distance: float


class SubtaskTongSimRunner:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self._camera_id: bytes | None = None
        self._agent_id: str | None = None
        self._held_object_id: str | None = None

    async def _save_rgba_png(self, frame: dict[str, Any], path: Path) -> Path | None:
        rgba = frame.get("rgba8")
        if not rgba:
            return None
        width = int(frame.get("width", 0))
        height = int(frame.get("height", 0))
        if width <= 0 or height <= 0:
            return None

        try:
            from PIL import Image  # type: ignore

            img = Image.frombytes("RGBA", (width, height), bytes(rgba), "raw", "BGRA")
            out = path.with_suffix(".png")
            img.save(out)
            return out
        except Exception:
            return None

    async def _capture(self, conn: Any, tag: str) -> str | None:
        if not self._camera_id:
            return None
        if self._agent_id:
            tf = await ts.UnaryAPI.get_actor_transform(conn, self._agent_id)
            if tf is not None:
                # Simple chase camera: behind + above the agent, looking forward yaw.
                euler = quaternion_to_euler(tf.rotation, is_degree=True)
                yaw_rad = math.radians(float(euler.z))
                camera_loc = Vector3(
                    float(tf.location.x - 220.0 * math.cos(yaw_rad)),
                    float(tf.location.y - 220.0 * math.sin(yaw_rad)),
                    float(tf.location.z + 160.0),
                )
                await ts.CaptureAPI.set_camera_pose(
                    conn,
                    self._camera_id,
                    Transform(location=camera_loc, rotation=tf.rotation),
                )
        snap = await ts.CaptureAPI.capture_snapshot(
            conn,
            self._camera_id,
            include_color=True,
            include_depth=False,
            timeout_seconds=1.0,
        )
        if not snap:
            return None
        out = await self._save_rgba_png(snap, self.frames_dir / tag)
        return str(out) if out else None

    @staticmethod
    def _pick_agent_id(actors: list[dict[str, Any]], user_agent_id: str | None) -> str:
        if user_agent_id:
            return user_agent_id

        def score(a: dict[str, Any]) -> int:
            text = f"{a.get('name', '')} {a.get('class_path', '')} {a.get('tag', '')}".lower()
            s = 0
            for k, v in [("character", 4), ("pawn", 3), ("agent", 3), ("player", 2), ("mannequin", 2)]:
                if k in text:
                    s += v
            return s

        ranked = sorted(actors, key=score, reverse=True)
        if not ranked:
            raise RuntimeError("QueryState 返回空，无法选择可控 agent。")
        return ranked[0]["id"]

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(ch.lower() for ch in text if ch.isalnum() or ch == "_")

    def _resolve_target_actor(self, actors: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
        q = self._normalize(label)
        for actor in actors:
            hay = " ".join(
                [
                    str(actor.get("name", "")),
                    str(actor.get("tag", "")),
                    str(actor.get("class_path", "")),
                ]
            )
            if q and q in self._normalize(hay):
                return actor
        return None

    @staticmethod
    def _target_half_extent_xy(target_actor: dict[str, Any]) -> float:
        """
        Estimate target footprint radius in XY from AABB.
        """
        bbox = target_actor.get("bounding_box") or {}
        bmin = bbox.get("min") or {}
        bmax = bbox.get("max") or {}
        try:
            ex = max(0.0, float(bmax.get("x", 0.0)) - float(bmin.get("x", 0.0)))
            ey = max(0.0, float(bmax.get("y", 0.0)) - float(bmin.get("y", 0.0)))
        except Exception:
            return 0.0
        return 0.5 * math.hypot(ex, ey)

    async def _do_move2(self, conn: Any, target_actor: dict[str, Any]) -> dict[str, Any]:
        # "move2" => move to an interactable nearby point (not overlapping target center).
        agent_tf = await ts.UnaryAPI.get_actor_transform(conn, self._agent_id)
        if agent_tf is None:
            return {"success": False, "message": "get_actor_transform failed"}

        loc = target_actor["location"]
        tx, ty, tz = float(loc["x"]), float(loc["y"]), float(loc["z"])
        ax, ay = float(agent_tf.location.x), float(agent_tf.location.y)
        dx, dy = tx - ax, ty - ay
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            # Agent is already very close; pick a small offset to avoid zero-vector heading.
            nx, ny = 1.0, 0.0
        else:
            nx, ny = dx / dist, dy / dist

        # Move distance is explicitly determined as:
        # max(user_interact_distance, target_half_extent_xy + interact_clearance).
        # This avoids stopping inside large objects while keeping interaction range.
        target_half_extent_xy = self._target_half_extent_xy(target_actor)
        stand = max(
            10.0,
            float(self.cfg.interact_distance),
            float(target_half_extent_xy + self.cfg.interact_clearance),
        )
        goal = Vector3(tx - nx * stand, ty - ny * stand, float(agent_tf.location.z))
        move_backend = "simple_move_towards"
        nav_result: dict[str, Any] | None = None
        current_location = None
        hit = None

        if self.cfg.use_navmesh:
            path = await ts.UnaryAPI.query_navigation_path(
                conn,
                start=Vector3(
                    float(agent_tf.location.x),
                    float(agent_tf.location.y),
                    float(agent_tf.location.z),
                ),
                end=goal,
                allow_partial=True,
                require_navigable_end_location=False,
                timeout=2.0,
            )
            if path and path.get("points"):
                nav_result = await ts.UnaryAPI.navigate_to_location(
                    conn,
                    actor_id=self._agent_id,
                    target_location=goal,
                    accept_radius=float(self.cfg.nav_accept_radius),
                    allow_partial=True,
                    speed_uu_per_sec=float(self.cfg.move_speed),
                    timeout=120.0,
                )
                move_backend = "navigate_to_location"
                if nav_result and nav_result.get("success"):
                    current_location = nav_result.get("final_location")
            elif self.cfg.strict_navmesh:
                return {
                    "success": False,
                    "message": "strict_navmesh enabled: no valid navmesh path found, skip straight-line fallback",
                    "move_backend": "navigate_to_location",
                    "current_location": None,
                    "target_location": {"x": tx, "y": ty, "z": tz},
                    "goal_location": {"x": float(goal.x), "y": float(goal.y), "z": float(goal.z)},
                    "computed_standoff_distance": float(stand),
                    "target_half_extent_xy": float(target_half_extent_xy),
                    "nav_result": None,
                    "hit": None,
                }

            if (
                self.cfg.strict_navmesh
                and nav_result is not None
                and not bool(nav_result.get("success"))
            ):
                return {
                    "success": False,
                    "message": "strict_navmesh enabled: navmesh navigation failed, skip straight-line fallback",
                    "move_backend": "navigate_to_location",
                    "current_location": None,
                    "target_location": {"x": tx, "y": ty, "z": tz},
                    "goal_location": {"x": float(goal.x), "y": float(goal.y), "z": float(goal.z)},
                    "computed_standoff_distance": float(stand),
                    "target_half_extent_xy": float(target_half_extent_xy),
                    "nav_result": nav_result,
                    "hit": None,
                }

        if current_location is None:
            current_location, hit = await ts.UnaryAPI.simple_move_towards(
                conn,
                actor_id=self._agent_id,
                target_location=goal,
                orientation_mode=RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT,
                speed_uu_per_sec=self.cfg.move_speed,
                tolerance_uu=self.cfg.move_tolerance,
                timeout=120.0,
            )
            move_backend = "simple_move_towards"
        # Face target for follow-up interaction.
        if current_location is not None:
            await self._face_target(conn, target_actor)
        return {
            "success": current_location is not None,
            "message": "moved_near_target" if current_location is not None else "move failed",
            "move_backend": move_backend,
            "current_location": current_location,
            "target_location": {"x": tx, "y": ty, "z": tz},
            "goal_location": {"x": float(goal.x), "y": float(goal.y), "z": float(goal.z)},
            "computed_standoff_distance": float(stand),
            "target_half_extent_xy": float(target_half_extent_xy),
            "nav_result": nav_result,
            "hit": str(hit) if hit else None,
        }

    async def _do_rotation(self, conn: Any) -> dict[str, Any]:
        tf = await ts.UnaryAPI.get_actor_transform(conn, self._agent_id)
        if tf is None:
            return {"success": False, "message": "get_actor_transform failed"}

        # Rotation parameters are derived from current visual FoV convention:
        # e.g. FoV 180 => 2 turns for full 360; FoV 120 => 3 turns.
        effective_yaw = (
            float(self.cfg.rotation_yaw_deg)
            if self.cfg.rotation_yaw_deg > 0
            else float(self.cfg.fov_yaw_deg)
        )
        effective_times = (
            int(self.cfg.rotation_times)
            if self.cfg.rotation_times > 0
            else max(1, int(round(360.0 / max(1.0, float(self.cfg.fov_yaw_deg)))))
        )
        euler = quaternion_to_euler(tf.rotation, is_degree=True)
        euler.z += effective_yaw * effective_times
        tf.rotation = euler_to_quaternion(euler, is_degree=True)
        ok = await ts.UnaryAPI.set_actor_transform(conn, self._agent_id, tf)
        return {
            "success": bool(ok),
            "message": "rotated" if ok else "rotation failed",
            "yaw_per_step": effective_yaw,
            "times": effective_times,
            "yaw_after": float(euler.z),
        }

    async def _face_target(self, conn: Any, target_actor: dict[str, Any]) -> bool:
        tf = await ts.UnaryAPI.get_actor_transform(conn, self._agent_id)
        if tf is None:
            return False
        loc = target_actor["location"]
        dx = float(loc["x"]) - float(tf.location.x)
        dy = float(loc["y"]) - float(tf.location.y)
        yaw_deg = math.degrees(math.atan2(dy, dx))
        euler = quaternion_to_euler(tf.rotation, is_degree=True)
        euler.z = yaw_deg
        tf.rotation = euler_to_quaternion(euler, is_degree=True)
        return bool(await ts.UnaryAPI.set_actor_transform(conn, self._agent_id, tf))

    async def _ensure_near_and_facing(self, conn: Any, target_actor: dict[str, Any]) -> dict[str, Any]:
        move_result = await self._do_move2(conn, target_actor)
        return {
            "success": bool(move_result.get("success")),
            "message": move_result.get("message", ""),
            "move_result": move_result,
        }

    async def _do_pick_up(self, conn: Any, target_actor: dict[str, Any]) -> dict[str, Any]:
        prep = await self._ensure_near_and_facing(conn, target_actor)
        if not prep["success"]:
            return {"success": False, "message": "failed to approach target before pick_up", "prep": prep}
        result = await ts.UnaryAPI.pick_up_object(
            conn,
            actor_id=self._agent_id,
            target_object_id=target_actor["id"],
            target_object_location=Vector3(
                float(target_actor["location"]["x"]),
                float(target_actor["location"]["y"]),
                float(target_actor["location"]["z"]),
            ),
            hand=RLDemoHandType.HAND_RIGHT,
            timeout=8.0,
        )
        if result.get("success"):
            self._held_object_id = target_actor["id"]
        result["prep"] = prep
        return result

    async def _do_put_down(self, conn: Any) -> dict[str, Any]:
        tf = await ts.UnaryAPI.get_actor_transform(conn, self._agent_id)
        if tf is None:
            return {"success": False, "message": "get_actor_transform failed"}

        yaw = quaternion_to_euler(tf.rotation, is_degree=True).z
        yaw_rad = math.radians(float(yaw))
        drop_loc = Vector3(
            float(tf.location.x + self.cfg.drop_distance * math.cos(yaw_rad)),
            float(tf.location.y + self.cfg.drop_distance * math.sin(yaw_rad)),
            float(tf.location.z),
        )
        result = await ts.UnaryAPI.drop_object(
            conn,
            actor_id=self._agent_id,
            target_drop_location=drop_loc,
            hand=RLDemoHandType.HAND_RIGHT,
            enable_physics=True,
            timeout=8.0,
        )
        if result.get("success"):
            self._held_object_id = None
        return result

    async def _execute_action(
        self,
        conn: Any,
        actors: list[dict[str, Any]],
        action_name: str,
        target_actor: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if action_name == "move2":
            if not target_actor:
                return {"success": False, "message": "target actor not found"}
            return await self._do_move2(conn, target_actor)

        if action_name == "rotation":
            return await self._do_rotation(conn)

        if action_name == "pick_up":
            if not target_actor:
                return {"success": False, "message": "target actor not found"}
            return await self._do_pick_up(conn, target_actor)

        if action_name == "put_down":
            return await self._do_put_down(conn)

        if action_name in {"open", "close", "turn_on", "turn_off"}:
            prep = None
            if target_actor is not None:
                prep = await self._ensure_near_and_facing(conn, target_actor)
            return {
                "success": False,
                "message": (
                    f"{action_name} 当前 TongSim 开源 Python SDK 未提供直接 unary API，"
                    "已跳过（可后续在 UE 侧新增 gRPC 接口后接入）。"
                ),
                "prep": prep,
            }

        return {"success": False, "message": f"unsupported action: {action_name}"}

    async def run(self) -> dict[str, Any]:
        raw = json.loads(self.cfg.subtasks_json.read_text(encoding="utf-8"))
        subtasks: list[dict[str, Any]] = list(raw.get("subtasks", []))

        report: dict[str, Any] = {
            "subtasks_file": str(self.cfg.subtasks_json),
            "started_at": datetime.utcnow().isoformat() + "Z",
            "agent_id": None,
            "supported_action_api": {
                "move2": "UnaryAPI.simple_move_towards",
                "rotation": "UnaryAPI.get_actor_transform + UnaryAPI.set_actor_transform",
                "pick_up": "UnaryAPI.pick_up_object",
                "put_down": "UnaryAPI.drop_object",
                "open": None,
                "close": None,
                "turn_on": None,
                "turn_off": None,
            },
            "subtask_results": [],
        }

        with ts.TongSim(grpc_endpoint=self.cfg.grpc_endpoint) as ue:
            conn = ue.context.conn
            actors = await ts.UnaryAPI.query_info(conn)
            self._agent_id = self._pick_agent_id(actors, self.cfg.agent_id)
            report["agent_id"] = self._agent_id

            agent_tf = await ts.UnaryAPI.get_actor_transform(conn, self._agent_id)
            if agent_tf is None:
                raise RuntimeError("无法读取 agent transform，执行中止。")

            self._camera_id = await ts.CaptureAPI.create_camera(
                conn,
                transform=Transform(location=Vector3(agent_tf.location), rotation=agent_tf.rotation),
                params=DEFAULT_CAPTURE_PARAMS,
                capture_name="TongBenchTaskRunnerCam",
            )
            try:
                for idx, task in enumerate(subtasks, start=1):
                    actors = await ts.UnaryAPI.query_info(conn)

                    params = task.get("available_parameters", {}) or {}
                    target_label = None
                    if isinstance(params, dict):
                        for key in ("object", "object_a", "object2", "object_b"):
                            values = params.get(key)
                            if isinstance(values, list) and values:
                                target_label = str(values[0])
                                break

                    target_actor = (
                        self._resolve_target_actor(actors, target_label)
                        if target_label
                        else None
                    )

                    action_logs: list[dict[str, Any]] = []
                    for step_i, action_name in enumerate(task.get("action_space", []), start=1):
                        before = await self._capture(conn, f"task{idx:03d}_step{step_i:02d}_{action_name}_before")
                        result = await self._execute_action(conn, actors, str(action_name), target_actor)
                        after = await self._capture(conn, f"task{idx:03d}_step{step_i:02d}_{action_name}_after")

                        action_logs.append(
                            {
                                "action": action_name,
                                "target_label": target_label,
                                "target_actor_id": target_actor["id"] if target_actor else None,
                                "derived_args": {
                                    "move2_interact_distance": self.cfg.interact_distance,
                                    "move2_interact_clearance": self.cfg.interact_clearance,
                                    "move2_use_navmesh": self.cfg.use_navmesh,
                                    "move2_strict_navmesh": self.cfg.strict_navmesh,
                                    "move2_nav_accept_radius": self.cfg.nav_accept_radius,
                                    "rotation_fov_yaw_deg": self.cfg.fov_yaw_deg,
                                    "rotation_yaw_deg": self.cfg.rotation_yaw_deg,
                                    "rotation_times": self.cfg.rotation_times,
                                },
                                "before_image": before,
                                "after_image": after,
                                "result": result,
                            }
                        )

                    report["subtask_results"].append(
                        {
                            "subtask_id": task.get("subtask_id"),
                            "task_type": task.get("task_type"),
                            "instruction_template": task.get("instruction_template"),
                            "actions": action_logs,
                        }
                    )
            finally:
                if self._camera_id is not None:
                    await ts.CaptureAPI.destroy_camera(conn, self._camera_id)

        report["finished_at"] = datetime.utcnow().isoformat() + "Z"
        return report


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(
        description=(
            "在 TongSim 中执行 outputs/subtasks_generated_50.json，并为每个动作保存执行前后场景图。"
        )
    )
    parser.add_argument("--grpc-endpoint", default="127.0.0.1:5726")
    parser.add_argument("--subtasks-json", default="outputs/subtasks_generated_50.json")
    parser.add_argument("--output-dir", default="outputs/tongsim_task_runs")
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--move-speed", type=float, default=300.0)
    parser.add_argument("--move-tolerance", type=float, default=80.0)
    parser.add_argument("--rotation-yaw-deg", type=float, default=120.0)
    parser.add_argument(
        "--rotation-times",
        type=int,
        default=0,
        help="<=0 时按 fov 自动计算：round(360/fov)。例如 fov=180->2, fov=120->3。",
    )
    parser.add_argument(
        "--fov-yaw-deg",
        type=float,
        default=120.0,
        help="用于 rotation 自动推导次数；如 180 则两次可扫 360。",
    )
    parser.add_argument("--interact-distance", type=float, default=120.0)
    parser.add_argument(
        "--interact-clearance",
        type=float,
        default=40.0,
        help="在目标 AABB 半径基础上额外留出的交互安全距离（uu）。",
    )
    parser.add_argument(
        "--use-navmesh",
        action="store_true",
        help="优先使用 NavMesh 导航（query_navigation_path + navigate_to_location）绕障碍移动。",
    )
    parser.add_argument(
        "--strict-navmesh",
        action="store_true",
        help="开启后禁止降级为直线移动；导航失败则直接判定该动作失败。",
    )
    parser.add_argument("--nav-accept-radius", type=float, default=100.0)
    parser.add_argument("--drop-distance", type=float, default=80.0)
    args = parser.parse_args()
    if args.strict_navmesh and not args.use_navmesh:
        args.use_navmesh = True

    run_dir = Path(args.output_dir) / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    return RunnerConfig(
        grpc_endpoint=args.grpc_endpoint,
        subtasks_json=Path(args.subtasks_json),
        output_dir=run_dir,
        agent_id=args.agent_id,
        move_speed=args.move_speed,
        move_tolerance=args.move_tolerance,
        rotation_yaw_deg=args.rotation_yaw_deg,
        rotation_times=args.rotation_times,
        fov_yaw_deg=args.fov_yaw_deg,
        interact_distance=args.interact_distance,
        interact_clearance=args.interact_clearance,
        use_navmesh=bool(args.use_navmesh),
        strict_navmesh=bool(args.strict_navmesh),
        nav_accept_radius=args.nav_accept_radius,
        drop_distance=args.drop_distance,
    )


def main() -> None:
    cfg = parse_args()
    runner = SubtaskTongSimRunner(cfg)
    report = asyncio.run(runner.run())

    report_path = cfg.output_dir / "execution_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] report => {report_path}")
    print(f"[DONE] frames => {cfg.output_dir / 'frames'}")


if __name__ == "__main__":
    main()
