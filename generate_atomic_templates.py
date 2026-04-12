#!/usr/bin/env python3
"""
Generate grounded atomic task templates from:
- touchdata.json: scene object instances with labels / geometry / contacts
- touchdata_objects.json: affordances per object id
- actions_prompts.yaml: available actions and constraints

Design choice:
- Generate TEMPLATE-LEVEL operators, not one template per object instance.
- Group by (operator_family, canonical_label).
- Keep candidate object_ids so later sampling can instantiate concrete tasks.
- Use affordances as the authoritative filter for open/close, pick_up/put_down, turn_on/turn_off.
- Do not hallucinate hidden state.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError as e:
    raise SystemExit("Please install pyyaml: pip install pyyaml") from e


# ----------------------------
# utilities
# ----------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def bbox_center(bbox: List[float]) -> Optional[Tuple[float, float, float]]:
    if not bbox or len(bbox) != 6:
        return None
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)


def bbox_size(bbox: List[float]) -> Optional[Tuple[float, float, float]]:
    if not bbox or len(bbox) != 6:
        return None
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return (max(0.0, xmax - xmin), max(0.0, ymax - ymin), max(0.0, zmax - zmin))


def nearest_anchor_text(objs: List["SceneObject"]) -> str:
    """Simple readable description for multiple candidate instances."""
    if not objs:
        return "unknown location"
    xs = [o.location[0] for o in objs if o.location]
    ys = [o.location[1] for o in objs if o.location]
    if not xs or not ys:
        return "various locations"
    x, y = sum(xs) / len(xs), sum(ys) / len(ys)
    return f"approx_center_xy=({x:.1f},{y:.1f})"


def make_unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ----------------------------
# schema
# ----------------------------

@dataclass
class SceneObject:
    object_id: str
    label: str
    location: Optional[List[float]]
    rotation: Optional[List[float]]
    bounding_box: Optional[List[float]]
    contacts: List[str]
    openable: bool
    pickable: bool
    powerable: bool


@dataclass
class AtomicTemplate:
    template_id: str
    operator_family: str
    primitive: bool
    scene: str
    instruction_templates: List[str]
    task_intent: str
    arguments: Dict[str, str]
    available_parameters: Dict[str, List[str]]
    candidate_object_ids: List[str]
    input_resources: List[str]
    output_resources: List[str]
    preconditions: List[str]
    effects_add: List[str]
    effects_del: List[str]
    action_space: List[str]
    evaluation_checkpoints: Dict[str, float]
    metadata: Dict[str, Any]


# ----------------------------
# normalization and grouping
# ----------------------------

LABEL_NORMALIZATION = {
    "coffeetable": "coffee_table",
    "tablelamp": "table_lamp",
    "wastebasket": "trash_can",
    "bathshelf": "bath_shelf",
    "floorboard": "floor",
    "computer": "laptop",
}

# Optional nicer task intents by operator family.
INTENT_TEXT = {
    "NavigateTo": "Reach a target object instance in the current scene.",
    "Open": "Open an interactable object in the scene.",
    "Close": "Close an interactable object in the scene.",
    "PickUp": "Pick up a movable object in the scene.",
    "PutDown": "Release a currently held object.",
    "TurnOn": "Turn on a powerable object in the scene.",
    "TurnOff": "Turn off a powerable object in the scene.",
    "RotateView": "Rotate the agent view to scan the environment.",
}

# Instruction templates by operator family.
INSTRUCTION_BANK = {
    "NavigateTo": [
        "Move to the {object}.",
        "Go to the {object}.",
        "Navigate to the {object}.",
    ],
    "Open": [
        "Find the {object} and open it.",
        "Go to the {object} and open it.",
        "Open the {object}.",
    ],
    "Close": [
        "Find the {object} and close it.",
        "Go to the {object} and close it.",
        "Close the {object}.",
    ],
    "PickUp": [
        "Find the {object} and pick it up.",
        "Go to the {object} and pick it up.",
        "Pick up the {object}.",
    ],
    "PutDown": [
        "Put down the held object.",
        "Drop the object you are carrying.",
    ],
    "TurnOn": [
        "Find the {object} and turn it on.",
        "Go to the {object} and switch it on.",
        "Turn on the {object}.",
    ],
    "TurnOff": [
        "Find the {object} and turn it off.",
        "Go to the {object} and switch it off.",
        "Turn off the {object}.",
    ],
    "RotateView": [
        "Rotate your view to scan the environment.",
        "Turn and scan the surrounding scene.",
    ],
}


def canonical_label(label: str) -> str:
    key = slugify(label)
    return LABEL_NORMALIZATION.get(key, key)


def build_scene_objects(
    scene_json: Dict[str, Any],
    affordance_json: Dict[str, Any],
) -> List[SceneObject]:
    affordances = affordance_json.get("objects", affordance_json)
    objs: List[SceneObject] = []

    for object_id, raw in scene_json.items():
        label = raw.get("label")
        if not label:
            continue

        aff = affordances.get(object_id, {})
        objs.append(
            SceneObject(
                object_id=object_id,
                label=canonical_label(str(label)),
                location=raw.get("Location"),
                rotation=raw.get("Rotation"),
                bounding_box=raw.get("BoundingBox"),
                contacts=raw.get("Contact", []) or [],
                openable=bool(aff.get("openable", False)),
                pickable=bool(aff.get("pickable", False)),
                powerable=bool(aff.get("powerable", False)),
            )
        )
    return objs


# ----------------------------
# template generation
# ----------------------------

def template_common_metadata(objs: List[SceneObject]) -> Dict[str, Any]:
    sizes = [bbox_size(o.bounding_box) for o in objs]
    sizes = [s for s in sizes if s is not None]
    dims = None
    if sizes:
        dims = {
            "avg_bbox_size": {
                "x": round(sum(s[0] for s in sizes) / len(sizes), 3),
                "y": round(sum(s[1] for s in sizes) / len(sizes), 3),
                "z": round(sum(s[2] for s in sizes) / len(sizes), 3),
            }
        }

    return {
        "instance_count": len(objs),
        "object_labels": make_unique(o.label for o in objs),
        "location_hint": nearest_anchor_text(objs),
        **(dims or {}),
    }


def build_navigate_template(label: str, objs: List[SceneObject], scene_name: str) -> AtomicTemplate:
    rid = f"{label}_exists"
    return AtomicTemplate(
        template_id=f"navigate_to__{label}",
        operator_family="NavigateTo",
        primitive=False,
        scene=scene_name,
        instruction_templates=INSTRUCTION_BANK["NavigateTo"],
        task_intent=INTENT_TEXT["NavigateTo"],
        arguments={"object": "SceneObjectLabel"},
        available_parameters={"object": [label]},
        candidate_object_ids=[o.object_id for o in objs],
        input_resources=[rid],
        output_resources=[f"agent_at_{label}"],
        preconditions=[f"exists({label})"],
        effects_add=[f"at(agent,{label})"],
        effects_del=[],
        action_space=["move2"],
        evaluation_checkpoints={"at_object": 1.0},
        metadata=template_common_metadata(objs),
    )


def build_openclose_template(label: str, objs: List[SceneObject], scene_name: str, open_mode: bool) -> AtomicTemplate:
    family = "Open" if open_mode else "Close"
    rid = f"{label}_exists"
    return AtomicTemplate(
        template_id=f"{family.lower()}__{label}",
        operator_family=family,
        primitive=False,
        scene=scene_name,
        instruction_templates=INSTRUCTION_BANK[family],
        task_intent=INTENT_TEXT[family],
        arguments={"object": "OpenableObjectLabel"},
        available_parameters={"object": [label]},
        candidate_object_ids=[o.object_id for o in objs],
        input_resources=[rid],
        output_resources=[f"{label}_{'opened' if open_mode else 'closed'}"],
        preconditions=[f"exists({label})", f"openable({label})", f"at(agent,{label})"],
        effects_add=[f"{'open' if open_mode else 'closed'}({label})"],
        effects_del=[f"{'closed' if open_mode else 'open'}({label})"],
        action_space=["move2", family.lower()],
        evaluation_checkpoints={"visible": 0.2, "at_object": 0.3, "done": 0.5},
        metadata=template_common_metadata(objs),
    )


def build_pick_template(label: str, objs: List[SceneObject], scene_name: str) -> AtomicTemplate:
    rid = f"{label}_exists"
    return AtomicTemplate(
        template_id=f"pick_up__{label}",
        operator_family="PickUp",
        primitive=False,
        scene=scene_name,
        instruction_templates=INSTRUCTION_BANK["PickUp"],
        task_intent=INTENT_TEXT["PickUp"],
        arguments={"object": "PickableObjectLabel"},
        available_parameters={"object": [label]},
        candidate_object_ids=[o.object_id for o in objs],
        input_resources=[rid],
        output_resources=[f"{label}_in_hand"],
        preconditions=[
            f"exists({label})",
            f"pickable({label})",
            "hand_empty(agent)",
            f"at(agent,{label})",
        ],
        effects_add=[f"holding(agent,{label})"],
        effects_del=["hand_empty(agent)"],
        action_space=["move2", "pick_up"],
        evaluation_checkpoints={"visible": 0.2, "at_object": 0.3, "picked": 0.5},
        metadata=template_common_metadata(objs),
    )


def build_putdown_template(scene_name: str) -> AtomicTemplate:
    return AtomicTemplate(
        template_id="put_down__held_object",
        operator_family="PutDown",
        primitive=False,
        scene=scene_name,
        instruction_templates=INSTRUCTION_BANK["PutDown"],
        task_intent=INTENT_TEXT["PutDown"],
        arguments={"object": "HeldObjectIdOptional"},
        available_parameters={},
        candidate_object_ids=[],
        input_resources=["item_held"],
        output_resources=["item_dropped"],
        preconditions=["holding(agent, some_object)"],
        effects_add=["dropped(some_object)"],
        effects_del=["holding(agent, some_object)"],
        action_space=["put_down"],
        evaluation_checkpoints={"held": 0.5, "dropped": 0.5},
        metadata={"instance_count": 0, "note": "Generic inventory-level template."},
    )


def build_power_template(label: str, objs: List[SceneObject], scene_name: str, on_mode: bool) -> AtomicTemplate:
    family = "TurnOn" if on_mode else "TurnOff"
    rid = f"{label}_exists"
    return AtomicTemplate(
        template_id=f"{family.lower()}__{label}",
        operator_family=family,
        primitive=False,
        scene=scene_name,
        instruction_templates=INSTRUCTION_BANK[family],
        task_intent=INTENT_TEXT[family],
        arguments={"object": "PowerableObjectLabel"},
        available_parameters={"object": [label]},
        candidate_object_ids=[o.object_id for o in objs],
        input_resources=[rid],
        output_resources=[f"{label}_{'on' if on_mode else 'off'}"],
        preconditions=[f"exists({label})", f"powerable({label})", f"at(agent,{label})"],
        effects_add=[f"{'on' if on_mode else 'off'}({label})"],
        effects_del=[f"{'off' if on_mode else 'on'}({label})"],
        action_space=["move2", "turn_on" if on_mode else "turn_off"],
        evaluation_checkpoints={"visible": 0.2, "at_object": 0.3, "done": 0.5},
        metadata=template_common_metadata(objs),
    )


def build_rotate_template(scene_name: str) -> AtomicTemplate:
    return AtomicTemplate(
        template_id="rotate_view",
        operator_family="RotateView",
        primitive=True,
        scene=scene_name,
        instruction_templates=INSTRUCTION_BANK["RotateView"],
        task_intent=INTENT_TEXT["RotateView"],
        arguments={"yaw_deg": "float", "times": "int"},
        available_parameters={"yaw_deg": ["45.0", "90.0", "120.0"], "times": ["1", "3"]},
        candidate_object_ids=[],
        input_resources=[],
        output_resources=["view_rotated"],
        preconditions=[],
        effects_add=["rotated_view(agent)"],
        effects_del=[],
        action_space=["rotation"],
        evaluation_checkpoints={"rotated": 1.0},
        metadata={"note": "Primitive exploration operator, not tied to a target object."},
    )


def generate_templates(
    scene_json: Dict[str, Any],
    affordance_json: Dict[str, Any],
    scene_name: str = "daily_life_scene",
) -> Dict[str, Any]:
    objects = build_scene_objects(scene_json, affordance_json)

    by_label: Dict[str, List[SceneObject]] = defaultdict(list)
    openable_by_label: Dict[str, List[SceneObject]] = defaultdict(list)
    pickable_by_label: Dict[str, List[SceneObject]] = defaultdict(list)
    powerable_by_label: Dict[str, List[SceneObject]] = defaultdict(list)

    for obj in objects:
        by_label[obj.label].append(obj)
        if obj.openable:
            openable_by_label[obj.label].append(obj)
        if obj.pickable:
            pickable_by_label[obj.label].append(obj)
        if obj.powerable:
            powerable_by_label[obj.label].append(obj)

    templates: List[AtomicTemplate] = []

    # Semantic navigation templates for every labeled object category.
    for label in sorted(by_label):
        templates.append(build_navigate_template(label, by_label[label], scene_name))

    for label in sorted(openable_by_label):
        templates.append(build_openclose_template(label, openable_by_label[label], scene_name, open_mode=True))
        templates.append(build_openclose_template(label, openable_by_label[label], scene_name, open_mode=False))

    for label in sorted(pickable_by_label):
        templates.append(build_pick_template(label, pickable_by_label[label], scene_name))

    for label in sorted(powerable_by_label):
        templates.append(build_power_template(label, powerable_by_label[label], scene_name, on_mode=True))
        templates.append(build_power_template(label, powerable_by_label[label], scene_name, on_mode=False))

    # Generic templates.
    templates.append(build_putdown_template(scene_name))
    templates.append(build_rotate_template(scene_name))

    summary = {
        "scene": scene_name,
        "num_scene_objects_with_label": len(objects),
        "num_unique_labels": len(by_label),
        "num_openable_labels": len(openable_by_label),
        "num_pickable_labels": len(pickable_by_label),
        "num_powerable_labels": len(powerable_by_label),
        "num_templates": len(templates),
        "label_counts": dict(sorted(Counter(o.label for o in objects).items())),
    }

    return {
        "summary": summary,
        "templates": [asdict(t) for t in templates],
    }


# ----------------------------
# prompt pack
# ----------------------------

def build_prompt_pack() -> Dict[str, str]:
    system_prompt = """You are generating ATOMIC TASK TEMPLATES for an embodied indoor simulation benchmark.

Goal:
Produce TEMPLATE-LEVEL atomic tasks, not one task per object instance.

Inputs you will receive:
1. Scene object metadata JSON
2. Affordance JSON with booleans: openable, pickable, powerable
3. Available action definitions and constraints

Core rules:
- Use object labels as the main grounding signal.
- Use affordances as the authoritative action gate.
- Do NOT assume hidden state such as already open / already closed / inside / contains / support unless explicitly provided.
- If multiple objects share one label, merge them into one template and keep candidate object_ids.
- Prefer operator families over instance duplication:
  NavigateTo(label), Open(label), Close(label), PickUp(label), TurnOn(label), TurnOff(label)
- Keep PutDown as a generic inventory-level template unless a target surface/container relation is explicitly available.
- Keep RotateView as a primitive exploration template, not a semantic object-goal template.
- Output only templates that are executable using the currently defined actions.

Output schema:
{
  "templates": [
    {
      "template_id": "pick_up__book",
      "operator_family": "PickUp",
      "primitive": false,
      "instruction_templates": ["Pick up the {object}."],
      "task_intent": "Pick up a movable object in the scene.",
      "arguments": {"object": "PickableObjectLabel"},
      "available_parameters": {"object": ["book"]},
      "candidate_object_ids": ["BP_Book_01_C_1", "BP_Book_02_C_1"],
      "input_resources": ["book_exists"],
      "output_resources": ["book_in_hand"],
      "preconditions": ["exists(book)", "pickable(book)", "hand_empty(agent)", "at(agent,book)"],
      "effects_add": ["holding(agent,book)"],
      "effects_del": ["hand_empty(agent)"],
      "action_space": ["move2", "pick_up"],
      "evaluation_checkpoints": {"visible": 0.2, "at_object": 0.3, "picked": 0.5}
    }
  ]
}"""

    validator_prompt = """You are validating atomic task templates.

Reject a template if:
- it uses an action forbidden by affordance flags
- it depends on hidden state not present in the data
- it duplicates another template at the same operator_family + canonical object label level
- it encodes an object instance-specific task when a label-level template is sufficient
- it uses put_down with an invented target location
- it treats contact as guaranteed containment/support
- it uses unlabeled or semantically unclear objects as the primary target

For every rejected template, return:
{
  "template_id": "...",
  "reject_reason": "...",
  "fix": "..."
}"""

    intent_prompt = """You are generating HIGH-LEVEL TASK INTENTS for graph composition from atomic templates.

Rules:
- Start from one coherent high-level intent.
- Only compose subtasks whose resource flow and semantics align with the same goal.
- Do not compose tasks purely because output_resources and input_resources match.
- Prefer intents like:
  "retrieve an item"
  "open access and inspect"
  "turn on a device"
- Avoid meaningless mixtures like:
  rotate view -> pick up book -> close curtain -> turn on lamp
  unless a single clear intent justifies all steps.

Return:
{
  "intent": "...",
  "required_templates": ["..."],
  "resource_flow": [["a","b"], ["b","c"]],
  "why_coherent": "..."
}"""

    return {
        "template_generation_system_prompt.txt": system_prompt,
        "template_validator_prompt.txt": validator_prompt,
        "intent_anchor_prompt.txt": intent_prompt,
    }


# ----------------------------
# cli
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", default="./env_data/touchdata.json", type=Path, required=False)
    parser.add_argument("--affordance_json", default="./env_data/touchdata_objects.json", type=Path, required=False)
    parser.add_argument("--actions_yaml", default="./env_data/actions_prompts.yaml", type=Path, required=False)
    parser.add_argument("--output_json", default="./outputs/atomic_templates.json", type=Path, required=False)
    parser.add_argument("--prompt_dir", default="./env_data/prompts", type=Path, required=False)
    parser.add_argument("--scene_name", default="daily_life_scene", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_json = load_json(args.scene_json)
    affordance_json = load_json(args.affordance_json)

    # actions_yaml is optional for now because action families are fixed by current code.
    # It is kept in the CLI for future expansion and compatibility.
    if args.actions_yaml and args.actions_yaml.exists():
        _ = load_yaml(args.actions_yaml)

    output = generate_templates(
        scene_json=scene_json,
        affordance_json=affordance_json,
        scene_name=args.scene_name,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if args.prompt_dir:
        args.prompt_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in build_prompt_pack().items():
            (args.prompt_dir / filename).write_text(content, encoding="utf-8")

    print(f"Wrote templates to: {args.output_json}")
    if args.prompt_dir:
        print(f"Wrote prompts to: {args.prompt_dir}")


if __name__ == "__main__":
    main()
