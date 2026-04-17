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
}

SKIP_LABELS = {
    "ceiling",
    "frame",
    "floorboard",
    "wall",
    "carpet",
    "decoration",
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


def parse_scene(scene: Dict[str, Any], affordance_root: Dict[str, Any]) -> List[SceneObject]:
    affordances = affordance_root.get("objects", affordance_root)
    results: List[SceneObject] = []
    for object_id, payload in scene.items():
        raw_label = payload.get("label")
        if not raw_label:
            continue
        label = canonical_label(raw_label)
        if label in SKIP_LABELS:
            continue
        caps = affordances.get(object_id, {})
        results.append(
            SceneObject(
                object_id=object_id,
                raw_label=raw_label,
                label=label,
                location=payload.get("Location", []),
                contacts=payload.get("Contact", []),
                openable=bool(caps.get("openable", False)),
                pickable=bool(caps.get("pickable", False)),
                powerable=bool(caps.get("powerable", False)),
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
            "available_objects": sorted(set(t.get("available_objects", []))),
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


PREDICATE_PATTERN = re.compile(
    r'^(empty|[A-Za-z_][A-Za-z0-9_]*\((?:"[^"]+"\s*(?:,\s*"[^"]+"\s*)*)\))$'
)


def extract_object_ids_from_predicates(predicates: Sequence[str]) -> List[str]:
    ids: List[str] = []
    for pred in predicates:
        ids.extend(re.findall(r'"([^"]+)"', pred))
    return ids


def validate_generated_tasks(tasks: Any, valid_object_ids: set[str]) -> List[Dict[str, Any]]:
    if not isinstance(tasks, list):
        raise ValueError("Model output is not a JSON array.")

    cleaned: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            continue
        required = ["task_id", "description", "intention", "initial_state", "final_state"]
        if any(k not in task for k in required):
            continue

        task_id = str(task["task_id"]).strip()
        description = str(task["description"]).strip()
        intention = str(task["intention"]).strip()
        initial_state = task["initial_state"]
        final_state = task["final_state"]

        if not task_id or not description or not intention:
            continue
        if task_id in seen_ids:
            task_id = f"T{idx:03d}"
        seen_ids.add(task_id)

        if not isinstance(initial_state, dict) or not isinstance(final_state, dict):
            continue

        for state in [initial_state, final_state]:
            if "predicates" not in state or "bindings" not in state:
                raise ValueError(f"Task {task_id} has invalid state structure.")
            if not isinstance(state["predicates"], list) or not isinstance(state["bindings"], dict):
                raise ValueError(f"Task {task_id} has invalid state structure.")
            normalized_preds: List[str] = []
            for pred in state["predicates"]:
                pred = str(pred).strip()
                if not PREDICATE_PATTERN.match(pred):
                    raise ValueError(f"Task {task_id} contains an invalid predicate: {pred}")
                normalized_preds.append(pred)
            state["predicates"] = normalized_preds
            for _, object_id in list(state["bindings"].items()):
                if not isinstance(object_id, str):
                    raise ValueError(f"Task {task_id} has non-string bindings.")
                if object_id not in valid_object_ids:
                    raise ValueError(f"Task {task_id} references unknown object id: {object_id}")

            ids_in_predicates = extract_object_ids_from_predicates(state["predicates"])
            for object_id in ids_in_predicates:
                if object_id not in valid_object_ids:
                    raise ValueError(f"Task {task_id} references unknown object id in predicates: {object_id}")

        cleaned.append(
            {
                "task_id": task_id,
                "description": description,
                "intention": intention,
                "initial_state": initial_state,
                "final_state": final_state,
            }
        )

    if not cleaned:
        raise ValueError("No valid tasks remained after validation.")
    return cleaned


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
    # print(api_key)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-level tasks by reading prompt files and calling an LLM API.")
    parser.add_argument("--scene", default="./env_data/touchdata.json")
    parser.add_argument("--affordances", default="./env_data/touchdata_objects.json")
    parser.add_argument("--atomic", default="./outputs/atomic_templates.json")
    parser.add_argument("--env", default="./prompts/env.yaml")
    parser.add_argument("--output", default="./outputs/tasks.json")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT_PATH)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE_PATH)
    args = parser.parse_args()

    try:
        scene = load_json(args.scene)
        affordances = load_json(args.affordances)
        atomic = load_json(args.atomic)
        env_text = load_text(args.env)

        objects = parse_scene(scene, affordances)
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
        valid_object_ids = {obj.object_id for obj in objects}
        result = tasks  # validate_generated_tasks(tasks, valid_object_ids)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(json.dumps({
            "ok": True,
            "output": args.output,
            "system_prompt_path": args.system_prompt,
            "user_template_path": args.user_template,
            "api_called": True,
        }, ensure_ascii=False, indent=2))

    except Exception as e:
        error = {"error": "Invalid task structure", "details": [str(e)]}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(error, f, ensure_ascii=False, indent=2)
        print(json.dumps(error, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
