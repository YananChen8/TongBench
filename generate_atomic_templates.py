#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

ACTIVE_ACTIONS = [
    "pick_up",
    "place",
    "open",
    "close",
    "pour",
    "switch_on",
    "switch_off",
    "hang_object",
    "hand_over",
    "cut",
    "plug_in",
    "unplug",
    "wash_object",
    "drop",
    "look_at",
    "point_at",
    "walk_to",
    "turn",
    "stand_up",
    "wash_hand",
]

EXCLUDED_LABELS = {
    "ceiling",
    "floor",
    "frame",
    "hook",
    "onoff",
    "rod",
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

POURABLE_LABELS = {"bottle", "drinkcontainer", "cup", "bowl", "kettle"}
HANGABLE_LABELS = {"towel", "cap", "bag", "clothes", "coat", "hat"}
CUTTABLE_LABELS = {"bread", "banana", "tomato", "eggplant", "fruit"}
PLUGGABLE_LABELS = {"plug", "adapter", "laptop", "tv", "radio", "lamp", "fan", "aircondition"}
WASHABLE_LABELS = {"plate", "bowl", "cup", "knife", "fork", "bottle", "tray", "toy", "towel"}

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text

def normalize_label(label: str) -> str:
    return LABEL_NORMALIZATION.get(slugify(label), slugify(label))

def keep_label(label: str) -> bool:
    return bool(label) and label not in EXCLUDED_LABELS

def collect_scene_labels(scene_json: Dict[str, Any]) -> Set[str]:
    labels: Set[str] = set()
    for rec in scene_json.values():
        raw = rec.get("label")
        if not raw:
            continue
        label = normalize_label(str(raw))
        if keep_label(label):
            labels.add(label)
    return labels

def collect_affordance_labels(scene_json: Dict[str, Any], affordance_json: Dict[str, Any]) -> Dict[str, Set[str]]:
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

def subset(labels: Set[str], allowed: Set[str]) -> List[str]:
    return sorted(x for x in labels if x in allowed)

def tpl(template_id: str, scene: str, action: str, desc: str, objs: List[str], pre: List[str], post: List[str]) -> Dict[str, Any]:
    return {
        "template_id": template_id,
        "scene": scene,
        "action_space": [action],
        "task_description": desc,
        "available_objects": objs,
        "pre_state": pre,
        "post_state": post,
    }

def generate_templates(scene_json: Dict[str, Any], affordance_json: Dict[str, Any], scene_name: str) -> Dict[str, Any]:
    labels = sorted(collect_scene_labels(scene_json))
    label_set = set(labels)
    aff = collect_affordance_labels(scene_json, affordance_json)
    templates: List[Dict[str, Any]] = []

    if "look_at" in ACTIVE_ACTIONS and labels:
        templates.append(tpl(
            "look_at_{object}", scene_name, "look_at",
            "Look at an object until it enters view.",
            labels,
            ["exists({object})"],
            ["in_view({object})"],
        ))

    if "point_at" in ACTIVE_ACTIONS and labels:
        templates.append(tpl(
            "point_at_{object}", scene_name, "point_at",
            "Point at an object that is already in view.",
            labels,
            ["in_view({object})"],
            ["pointed_at({object})"],
        ))

    if "walk_to" in ACTIVE_ACTIONS and labels:
        templates.append(tpl(
            "walk_to_{object}", scene_name, "walk_to",
            "Walk to an object that is already in view until it becomes reachable.",
            labels,
            ["in_view({object})"],
            ["with_reach({object})"],
        ))

    pickable = sorted(aff["pickable"])
    if "pick_up" in ACTIVE_ACTIONS and pickable:
        templates.append(tpl(
            "pick_up_{object}", scene_name, "pick_up",
            "Pick up a pickable object that is within reach.",
            pickable,
            ["with_reach({object})", "empty"],
            ["holding({object})"],
        ))

    if "place" in ACTIVE_ACTIONS and pickable:
        templates.append(tpl(
            "place_{object}", scene_name, "place",
            "Place a currently held object.",
            pickable,
            ["holding({object})"],
            ["empty"],
        ))

    if "drop" in ACTIVE_ACTIONS and pickable:
        templates.append(tpl(
            "drop_{object}", scene_name, "drop",
            "Drop a currently held object.",
            pickable,
            ["holding({object})"],
            ["empty"],
        ))

    openable = sorted(aff["openable"])
    if "open" in ACTIVE_ACTIONS and openable:
        templates.append(tpl(
            "open_{object}", scene_name, "open",
            "Open an openable object that is within reach.",
            openable,
            ["with_reach({object})", "closed({object})"],
            ["open({object})"],
        ))

    if "close" in ACTIVE_ACTIONS and openable:
        templates.append(tpl(
            "close_{object}", scene_name, "close",
            "Close an openable object that is within reach.",
            openable,
            ["with_reach({object})", "open({object})"],
            ["closed({object})"],
        ))

    powerable = sorted(aff["powerable"])
    if "switch_on" in ACTIVE_ACTIONS and powerable:
        templates.append(tpl(
            "switch_on_{object}", scene_name, "switch_on",
            "Switch on a powerable object that is within reach.",
            powerable,
            ["with_reach({object})", "off({object})"],
            ["on({object})"],
        ))

    if "switch_off" in ACTIVE_ACTIONS and powerable:
        templates.append(tpl(
            "switch_off_{object}", scene_name, "switch_off",
            "Switch off a powerable object that is within reach.",
            powerable,
            ["with_reach({object})", "on({object})"],
            ["off({object})"],
        ))

    pourable = subset(label_set, POURABLE_LABELS)
    if "pour" in ACTIVE_ACTIONS and pourable:
        templates.append(tpl(
            "pour_{object}", scene_name, "pour",
            "Pour from a held container-like object.",
            pourable,
            ["holding({object})", "filled({object})"],
            ["holding({object})", "empty({object})"],
        ))

    hangable = subset(label_set, HANGABLE_LABELS)
    if "hang_object" in ACTIVE_ACTIONS and hangable:
        templates.append(tpl(
            "hang_{object}", scene_name, "hang_object",
            "Hang a currently held object.",
            hangable,
            ["holding({object})"],
            ["empty"],
        ))

    if "hand_over" in ACTIVE_ACTIONS and pickable:
        templates.append(tpl(
            "hand_over_{object}", scene_name, "hand_over",
            "Hand over a currently held object.",
            pickable,
            ["holding({object})"],
            ["empty"],
        ))

    cuttable = subset(label_set, CUTTABLE_LABELS)
    if "cut" in ACTIVE_ACTIONS and cuttable:
        templates.append(tpl(
            "cut_{object}", scene_name, "cut",
            "Cut a reachable cuttable object while holding a knife.",
            cuttable,
            ["with_reach({object})", "holding(knife)"],
            ["cut({object})", "holding(knife)"],
        ))

    pluggable = subset(label_set, PLUGGABLE_LABELS)
    if "plug_in" in ACTIVE_ACTIONS and pluggable:
        templates.append(tpl(
            "plug_in_{object}", scene_name, "plug_in",
            "Plug in a reachable pluggable object.",
            pluggable,
            ["with_reach({object})", "unplugged({object})"],
            ["plugged_in({object})"],
        ))

    if "unplug" in ACTIVE_ACTIONS and pluggable:
        templates.append(tpl(
            "unplug_{object}", scene_name, "unplug",
            "Unplug a reachable pluggable object.",
            pluggable,
            ["with_reach({object})", "plugged_in({object})"],
            ["unplugged({object})"],
        ))

    washable = subset(label_set, WASHABLE_LABELS)
    if "wash_object" in ACTIVE_ACTIONS and washable:
        templates.append(tpl(
            "wash_{object}", scene_name, "wash_object",
            "Wash a reachable washable object.",
            washable,
            ["with_reach({object})", "dirty({object})"],
            ["clean({object})"],
        ))

    if "stand_up" in ACTIVE_ACTIONS:
        templates.append(tpl(
            "stand_up", scene_name, "stand_up",
            "Stand up from a seated posture.",
            [],
            ["sitting"],
            ["standing"],
        ))

    if "wash_hand" in ACTIVE_ACTIONS:
        templates.append(tpl(
            "wash_hand", scene_name, "wash_hand",
            "Wash hands.",
            [],
            ["dirty_hand"],
            ["clean_hand"],
        ))

    return {"scene": scene_name, "active_actions": ACTIVE_ACTIONS, "templates": templates}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", type=Path, required=True)
    parser.add_argument("--affordance_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--scene_name", type=str, default="daily_life_scene")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    scene_json = load_json(args.scene_json)
    affordance_json = load_json(args.affordance_json)
    out = generate_templates(scene_json, affordance_json, args.scene_name)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote templates to: {args.output_json}")

if __name__ == "__main__":
    main()
