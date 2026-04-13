#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

EXCLUDED_LABELS = {
    "adapter",
    "ceiling",
    "floor",
    "frame",
    "hook",
    "onoff",
    "rod",
    "socket",
    "wall",
}

LABEL_NORMALIZATION = {
    "coffeetable": "coffee_table",
    "tablelamp": "table_lamp",
    "wastebasket": "trash_can",
    "bathshelf": "bath_shelf",
    "floorboard": "floor",
    "computer": "laptop",
    "decorativedoor": "door",
    "cupboard_kitchen_room5": "cupboard",
    "cupboard_tv": "cupboard",
    "bowlcupboard": "cupboard",
    "shoescabinet": "cabinet",
    "electricfan": "fan",
    "decorationlamp": "lamp",
    "ceilinglamp": "lamp",
}

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
    return text

def normalize_label(label: str) -> str:
    key = slugify(label)
    return LABEL_NORMALIZATION.get(key, key)

def keep_label(label: str) -> bool:
    return bool(label) and label not in EXCLUDED_LABELS

def collect_scene_labels(scene_json: Dict[str, Any]) -> Set[str]:
    labels: Set[str] = set()
    for _, rec in scene_json.items():
        raw = rec.get("label")
        if not raw:
            continue
        label = normalize_label(str(raw))
        if keep_label(label):
            labels.add(label)
    return labels

def collect_affordance_labels(
    scene_json: Dict[str, Any],
    affordance_json: Dict[str, Any],
) -> Dict[str, Set[str]]:
    affordance_map = affordance_json.get("objects", affordance_json)
    out = {"openable": set(), "pickable": set(), "powerable": set()}

    for object_id, rec in scene_json.items():
        raw = rec.get("label")
        if not raw:
            continue
        label = normalize_label(str(raw))
        if not keep_label(label):
            continue

        aff = affordance_map.get(object_id, {})
        if aff.get("openable") is True:
            out["openable"].add(label)
        if aff.get("pickable") is True:
            out["pickable"].add(label)
        if aff.get("powerable") is True:
            out["powerable"].add(label)

    return out

def make_template(
    template_id: str,
    scene: str,
    action: str,
    task_description: str,
    available_objects: List[str],
    pre_state: List[str],
    post_state: List[str],
) -> Dict[str, Any]:
    return {
        "template_id": template_id,
        "scene": scene,
        "action_space": [action],
        "task_description": task_description,
        "available_objects": available_objects,
        "pre_state": pre_state,
        "post_state": post_state,
    }

def build_find_template(scene: str, labels: List[str]) -> Dict[str, Any]:
    return make_template(
        template_id="find_{object}",
        scene=scene,
        action="rotation",
        task_description="Find a target object by rotating the agent view until it enters view.",
        available_objects=labels,
        pre_state=["exists({object})"],
        post_state=["in_view({object})"],
    )

def build_navigate_template(scene: str, labels: List[str]) -> Dict[str, Any]:
    return make_template(
        template_id="navigate_to_{object}",
        scene=scene,
        action="move2",
        task_description="Move to an object that is already in view until it becomes reachable.",
        available_objects=labels,
        pre_state=["in_view({object})"],
        post_state=["with_reach({object})"],
    )

def build_pick_up_template(scene: str, labels: List[str]) -> Dict[str, Any]:
    return make_template(
        template_id="pick_up_{object}",
        scene=scene,
        action="pick_up",
        task_description="Pick up a pickable object that is within reach.",
        available_objects=labels,
        pre_state=["with_reach({object})", "empty"],
        post_state=["holding({object})"],
    )

def build_put_down_template(scene: str, labels: List[str]) -> Dict[str, Any]:
    return make_template(
        template_id="put_down_{object}",
        scene=scene,
        action="put_down",
        task_description="Put down a currently held object.",
        available_objects=labels,
        pre_state=["holding({object})"],
        post_state=["empty"],
    )

def build_open_template(scene: str, labels: List[str]) -> Dict[str, Any]:
    return make_template(
        template_id="open_{object}",
        scene=scene,
        action="open",
        task_description="Open an openable object that is within reach.",
        available_objects=labels,
        pre_state=["with_reach({object})", "closed({object})"],
        post_state=["open({object})"],
    )

def build_close_template(scene: str, labels: List[str]) -> Dict[str, Any]:
    return make_template(
        template_id="close_{object}",
        scene=scene,
        action="close",
        task_description="Close an openable object that is within reach.",
        available_objects=labels,
        pre_state=["with_reach({object})", "open({object})"],
        post_state=["closed({object})"],
    )

def build_turn_on_template(scene: str, labels: List[str]) -> Dict[str, Any]:
    return make_template(
        template_id="turn_on_{object}",
        scene=scene,
        action="turn_on",
        task_description="Turn on a powerable object that is within reach.",
        available_objects=labels,
        pre_state=["with_reach({object})", "off({object})"],
        post_state=["on({object})"],
    )

def build_turn_off_template(scene: str, labels: List[str]) -> Dict[str, Any]:
    return make_template(
        template_id="turn_off_{object}",
        scene=scene,
        action="turn_off",
        task_description="Turn off a powerable object that is within reach.",
        available_objects=labels,
        pre_state=["with_reach({object})", "on({object})"],
        post_state=["off({object})"],
    )

def generate_templates(
    scene_json: Dict[str, Any],
    affordance_json: Dict[str, Any],
    actions_yaml: Dict[str, Any],
    scene_name: str,
) -> Dict[str, Any]:
    labels = sorted(collect_scene_labels(scene_json))
    affordance_labels = collect_affordance_labels(scene_json, affordance_json)
    action_keys = list(actions_yaml.keys())
    templates: List[Dict[str, Any]] = []

    if "rotation" in action_keys and labels:
        templates.append(build_find_template(scene_name, labels))
    if "move2" in action_keys and labels:
        templates.append(build_navigate_template(scene_name, labels))

    pickable_labels = sorted(affordance_labels["pickable"])
    if "pick_up" in action_keys and pickable_labels:
        templates.append(build_pick_up_template(scene_name, pickable_labels))
    if "put_down" in action_keys and pickable_labels:
        templates.append(build_put_down_template(scene_name, pickable_labels))

    openable_labels = sorted(affordance_labels["openable"])
    if "open" in action_keys and openable_labels:
        templates.append(build_open_template(scene_name, openable_labels))
    if "close" in action_keys and openable_labels:
        templates.append(build_close_template(scene_name, openable_labels))

    powerable_labels = sorted(affordance_labels["powerable"])
    if "turn_on" in action_keys and powerable_labels:
        templates.append(build_turn_on_template(scene_name, powerable_labels))
    if "turn_off" in action_keys and powerable_labels:
        templates.append(build_turn_off_template(scene_name, powerable_labels))

    return {
        "scene": scene_name,
        "templates": templates,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", type=Path, required=True)
    parser.add_argument("--affordance_json", type=Path, required=True)
    parser.add_argument("--actions_yaml", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--scene_name", type=str, default="daily_life_scene")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    scene_json = load_json(args.scene_json)
    affordance_json = load_json(args.affordance_json)
    actions_yaml = load_yaml(args.actions_yaml)

    output = generate_templates(
        scene_json=scene_json,
        affordance_json=affordance_json,
        actions_yaml=actions_yaml,
        scene_name=args.scene_name,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote templates to: {args.output_json}")

if __name__ == "__main__":
    main()
