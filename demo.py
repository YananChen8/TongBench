"""
Quick demo for the top-level `tongsim_offline` package.

Run:
  python -m tongsim_offline.demo
"""

from __future__ import annotations

from tongbench import (
    AgentState,
    MoveToObject,
    ObjectState,
    OfflineEnv,
    PickUp,
    PutDown,
    TurnOn,
    Vec3,
)


def show(env: OfflineEnv, tag: str) -> None:
    obs = env.observe().to_dict()
    print(f"\n== {tag} ==")
    print("pos:", obs["agent_position"], "yaw:", obs["agent_external_state"]["yaw_deg"])
    print("inventory:", obs["inventory_ids"])
    print("visible:", obs["visible_object_ids"])
    print("nearby_operable:", obs["nearby_operable_ids"])


def main() -> None:
    env = OfflineEnv()
    env.reset(
        agent=AgentState(position=Vec3(0, 0, 0), yaw_deg=0),
        objects=[
            ObjectState("vase", name="Vase", position=Vec3(200, 0, 0), pickable=True),
            ObjectState("table", name="Table", position=Vec3(400, 0, 0), pickable=False),
            ObjectState("remote", name="Remote", position=Vec3(420, 40, 0), pickable=True),
            ObjectState("lamp", name="Lamp", position=Vec3(380, -50, 0), pickable=False, powerable=True),
        ],
    )

    show(env, "start")
    env.step(MoveToObject(target="vase"))
    env.step(PickUp(target="vase"))
    show(env, "picked vase")

    env.step(MoveToObject(target="table"))
    env.step(PutDown(object_id="vase"))
    show(env, "dropped vase")

    env.step(MoveToObject(target="lamp"))
    env.step(TurnOn(target="lamp"))
    show(env, "turned on lamp")


if __name__ == "__main__":
    main()

