#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-flash"
DEFAULT_API_KEY = "sk-5fa9123a51b44422b8c09a64c7548f66"

DEFAULT_SYSTEM_PROMPT_PATH = "/data02/cyn3/Selfagent/tongbench/prompts/task_generation_system_prompt.txt"
DEFAULT_USER_TEMPLATE_PATH = "/data02/cyn3/Selfagent/tongbench/prompts/task_generation_user_template.txt"

LABEL_ALIAS = {
    "bathshelf": "bath_shelf",
    "bowlcupboard": "cupboard",
    "coffeetable": "coffee_table",
    "computer": "laptop",
    "decorativedoor": "door",
    "diningtable": "dining_table",
    "drinkcontainer": "drink_container",
    "paperball": "paper_ball",
    "remotecontrol": "remote_control",
    "storagebox": "storage_box",
    "tablelamp": "table_lamp",
    "trashcan": "trash_can",
    "wastebasket": "trash_can",
    "ceilinglamp": "ceiling_lamp",
    "nightstand": "bedside",
    "shoescabinet": "storage_box",
    "shoebox": "storage_box",
    "suitcase": "storage_box",
}

SKIP_LABELS = {
    "ceiling",
    "frame",
    "floorboard",
    "wall",
    "carpet",
}

SEMANTIC_FAMILIES = {
    "guest_hospitality": {
        "required_any": [["cup", "coffee_table"], ["fruit", "plate"], ["cup", "tray"]],
        "description": "hosting a guest, serving refreshments, and making the shared area presentable",
    },
    "toy_tidy_up": {
        "required_any": [["toy", "storage_box"], ["toy", "wardrobe"], ["toy", "desk"]],
        "description": "reducing clutter by gathering scattered toys into a more organized area",
    },
    "workspace_preparation": {
        "required_any": [["laptop", "desk"], ["chair", "desk"], ["table_lamp", "desk"]],
        "description": "preparing a workspace for studying, reading, or computer use",
    },
    "air_and_light_adjustment": {
        "required_any": [["window", "curtains"], ["window", "aircondition"], ["table_lamp", "window"]],
        "description": "improving room comfort through light and air adjustment",
    },
    "kitchen_after_use_reset": {
        "required_any": [["dishwasher", "cupboard"], ["cup", "dishwasher"], ["plate", "dishwasher"]],
        "description": "restoring the kitchen after use and preparing it for later use",
    },
    "simple_tea_serving": {
        "required_any": [["kettle", "cup"], ["cup", "coffee_table"], ["tray", "cup"]],
        "description": "preparing a simple tea-serving arrangement for someone in the room",
    },
}

POWER_ACTIONS = {"switch_on", "switch_off", "turn_on", "turn_off", "power_on", "power_off", "plug_in", "unplug"}
PICK_ACTIONS = {"pick_up", "pickup", "pick", "grab", "hold", "take", "drop", "put_down"}
OPEN_ACTIONS = {"open", "close"}
REACH_ACTIONS = {"look_at", "walk_to"}

PRED_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$')


@dataclass
class SceneObject:
    object_id: str
    raw_label: str
    label: str
    location: List[float]
    contacts: List[str]
    openable: bool
    pickable: bool
    powerable: bool
    actions: List[str]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def canonical_label(label: str) -> str:
    norm = normalize(label)
    return LABEL_ALIAS.get(norm, norm)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def canonicalize_actions(payload: Dict[str, Any]) -> List[str]:
    raw: List[str] = []
    raw.extend(payload.get("supported_actions", []) or [])
    raw.extend(payload.get("candidate_actions", []) or [])
    actions = sorted({str(a).strip().lower() for a in raw if str(a).strip()})
    return actions


def parse_scene(scene_root: Dict[str, Any]) -> List[SceneObject]:
    results: List[SceneObject] = []
    for payload in scene_root.get("objects", []):
        raw_label = payload.get("rdf") or payload.get("object_name")
        object_id = payload.get("object_id")
        if not raw_label or not object_id:
            continue

        label = canonical_label(raw_label)
        if label in SKIP_LABELS:
            continue

        location_dict = payload.get("pose", {}).get("location", {})
        location = [
            float(location_dict.get("x", 0.0)),
            float(location_dict.get("y", 0.0)),
            float(location_dict.get("z", 0.0)),
        ]

        actions = set(canonicalize_actions(payload))
        results.append(
            SceneObject(
                object_id=object_id,
                raw_label=raw_label,
                label=label,
                location=location,
                contacts=[],
                openable=bool(OPEN_ACTIONS & actions),
                pickable=bool(PICK_ACTIONS & actions),
                powerable=bool(POWER_ACTIONS & actions),
                actions=sorted(actions),
            )
        )
    return results


def build_atomic_summary(atomic: Dict[str, Any]) -> Dict[str, Any]:
    templates = atomic.get("templates", [])
    summary: Dict[str, Any] = {
        "active_actions": atomic.get("active_actions", []),
        "templates": {},
    }
    for t in templates:
        tid = t["template_id"]
        summary["templates"][tid] = {
            "available_objects": t.get("available_objects", []),
            "pre_state": t.get("pre_state", []),
            "post_state": t.get("post_state", []),
        }
    return summary


def group_by_label(objects: Sequence[SceneObject]) -> Dict[str, List[SceneObject]]:
    out: Dict[str, List[SceneObject]] = defaultdict(list)
    for obj in objects:
        out[obj.label].append(obj)
    for label in out:
        out[label].sort(key=lambda x: (x.location or [0, 0, 0], x.object_id))
    return out


def shortlist_objects(by_label: Dict[str, List[SceneObject]], max_per_label: int = 4) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for label, items in sorted(by_label.items()):
        out[label] = []
        for obj in items[:max_per_label]:
            out[label].append(
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "location": obj.location,
                    "openable": obj.openable,
                    "pickable": obj.pickable,
                    "powerable": obj.powerable,
                    "supported_actions": obj.actions,
                    "contacts": obj.contacts[:6],
                }
            )
    return out


def infer_scene_capabilities(by_label: Dict[str, List[SceneObject]]) -> Dict[str, Any]:
    counts = {label: len(items) for label, items in sorted(by_label.items())}
    capabilities = {
        "pickable_labels": sorted([label for label, items in by_label.items() if any(x.pickable for x in items)]),
        "openable_labels": sorted([label for label, items in by_label.items() if any(x.openable for x in items)]),
        "powerable_labels": sorted([label for label, items in by_label.items() if any(x.powerable for x in items)]),
        "counts": counts,
    }
    themes: List[Dict[str, str]] = []
    present = set(by_label.keys())
    for name, spec in SEMANTIC_FAMILIES.items():
        ok = False
        for req_group in spec["required_any"]:
            if all(token in present for token in req_group):
                ok = True
                break
        if ok:
            themes.append({"theme": name, "description": spec["description"]})
    capabilities["suggested_themes"] = themes
    return capabilities


def render_prompts_from_files(
    objects: Sequence[SceneObject],
    atomic_summary: Dict[str, Any],
    env_text: str,
    num_tasks: int,
    system_prompt_path: str,
    user_template_path: str,
) -> Tuple[str, str]:
    by_label = group_by_label(objects)
    scene_caps = infer_scene_capabilities(by_label)
    object_catalog = shortlist_objects(by_label)

    system_prompt = load_text(system_prompt_path)
    user_template = load_text(user_template_path)
    environment_excerpt = env_text[:4000]

    user_prompt = user_template.format(
        num_tasks=num_tasks,
        scene_capabilities_json=json.dumps(scene_caps, ensure_ascii=False, indent=2),
        object_catalog_json=json.dumps(object_catalog, ensure_ascii=False, indent=2),
        atomic_templates_json=json.dumps(atomic_summary, ensure_ascii=False, indent=2),
        environment_guidance_json=json.dumps(environment_excerpt, ensure_ascii=False),
    )
    return system_prompt, user_prompt


def parse_json_from_text(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'(\[.*\]|\{.*\})', text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def call_chat_api(system_prompt: str, user_prompt: str, model: str, api_key: str, base_url: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed. Run: pip install openai")
    if not api_key or api_key == "PASTE_YOUR_FREE_API_KEY_HERE":
        raise RuntimeError(
            "No usable API key found. Edit DEFAULT_API_KEY in the script or set OPENAI_API_KEY / OPENROUTER_API_KEY."
        )
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    data = parse_json_from_text(content)
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    if isinstance(data, list):
        return data
    raise ValueError("Model JSON must be either a task array or an object with key 'tasks'.")


def resolve_api_key(args_api_key: Optional[str]) -> str:
    return (
        args_api_key
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
        or DEFAULT_API_KEY
    )


def parse_predicate_text(text: str) -> Tuple[str, List[str]]:
    text = (text or "").strip()
    m = PRED_RE.match(text)
    if not m:
        return text, []
    name = m.group(1).strip()
    arg_text = m.group(2).strip()
    if not arg_text:
        return name, []
    args = [a.strip() for a in arg_text.split(",")]
    return name, args


def build_object_index(objects: Sequence[SceneObject]) -> Dict[str, SceneObject]:
    return {obj.object_id: obj for obj in objects}


def validate_task(task: Dict[str, Any], object_index: Dict[str, SceneObject]) -> List[str]:
    issues: List[str] = []
    final_state = task.get("final_state", {}).get("predicates", [])
    initial_state = set(task.get("initial_state", {}).get("predicates", []))

    for pred_text in final_state:
        name, args = parse_predicate_text(pred_text)

        if name in {"on", "in"} and args:
            obj_id = args[0]
            obj = object_index.get(obj_id)
            if obj is None:
                issues.append(f"{pred_text}: object {obj_id} not found in scene.")
            elif not obj.pickable:
                issues.append(f"{pred_text}: {obj_id} is not pickable from supported actions {obj.actions}.")

        if name in {"open", "closed"} and args:
            obj_id = args[0]
            obj = object_index.get(obj_id)
            if obj is None:
                issues.append(f"{pred_text}: object {obj_id} not found in scene.")
            elif not obj.openable:
                issues.append(f"{pred_text}: {obj_id} is not openable from supported actions {obj.actions}.")

        if name in {"powered_on", "powered_off"} and args:
            obj_id = args[0]
            obj = object_index.get(obj_id)
            if obj is None:
                issues.append(f"{pred_text}: object {obj_id} not found in scene.")
            elif not obj.powerable:
                issues.append(f"{pred_text}: {obj_id} is not powerable from supported actions {obj.actions}.")

        if name == "powered_on" and args:
            obj_id = args[0]
            if f"powered_off({obj_id})" not in initial_state and f"plugged_in({obj_id})" not in initial_state:
                # soft warning, not a hard rejection
                issues.append(f"{pred_text}: initial state does not include powered_off({obj_id}) or plugged_in({obj_id}); planner may need bridge predicates.")

    return issues


def validate_tasks(tasks: List[Dict[str, Any]], objects: Sequence[SceneObject]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    object_index = build_object_index(objects)
    valid: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for task in tasks:
        issues = validate_task(task, object_index)
        if issues:
            task_copy = dict(task)
            task_copy["_validation_errors"] = issues
            rejected.append(task_copy)
        else:
            valid.append(task)
    return valid, rejected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate high-level tasks from existing_interactable_objects.json and call an LLM API."
    )
    parser.add_argument("--scene", default="./env_data/existing_interactable_objects.json")
    parser.add_argument("--atomic", default="./outputs/atomic_templates.json")
    parser.add_argument("--env", default="./prompts/env.yaml")
    parser.add_argument("--output", default="./outputs/tasks.json")
    parser.add_argument("--rejected-output", default="./outputs/tasks_rejected.json")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT_PATH)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE_PATH)
    parser.add_argument("--keep-invalid", action="store_true", help="Write invalid tasks too instead of rejecting them.")
    args = parser.parse_args()

    try:
        scene = load_json(args.scene)
        atomic = load_json(args.atomic)
        env_text = load_text(args.env)

        objects = parse_scene(scene)
        if not objects:
            raise ValueError("No valid scene objects found.")

        atomic_summary = build_atomic_summary(atomic)
        system_prompt, user_prompt = render_prompts_from_files(
            objects=objects,
            atomic_summary=atomic_summary,
            env_text=env_text,
            num_tasks=args.num_tasks,
            system_prompt_path=args.system_prompt,
            user_template_path=args.user_template,
        )

        api_key = resolve_api_key(args.api_key)
        tasks = call_chat_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
        )

        if not isinstance(tasks, list):
            raise ValueError("Generated tasks must be a list.")

        valid_tasks, rejected_tasks = validate_tasks(tasks, objects)

        tasks_to_write = tasks if args.keep_invalid else valid_tasks

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(tasks_to_write, f, ensure_ascii=False, indent=2)

        with open(args.rejected_output, "w", encoding="utf-8") as f:
            json.dump(rejected_tasks, f, ensure_ascii=False, indent=2)

        print(
            json.dumps(
                {
                    "ok": True,
                    "output": args.output,
                    "rejected_output": args.rejected_output,
                    "generated_count": len(tasks),
                    "valid_count": len(valid_tasks),
                    "rejected_count": len(rejected_tasks),
                    "system_prompt_path": args.system_prompt,
                    "user_template_path": args.user_template,
                    "api_called": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    except Exception as e:
        error = {"error": "Invalid task structure", "details": [str(e)]}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(error, f, ensure_ascii=False, indent=2)
        print(json.dumps(error, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
