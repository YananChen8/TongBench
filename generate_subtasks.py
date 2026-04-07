from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def normalize_label_from_object_id(object_id: str) -> str | None:
    s = object_id.split(".")[0]
    if s.startswith("BP_"):
        s = s[3:]
    elif s.startswith("ChildActor_GEN_VARIABLE_BP_"):
        s = s[len("ChildActor_GEN_VARIABLE_BP_") :]

    s = re.sub(r"_C(_\d+)?$", "", s)
    s = re.sub(r"_[0-9]+$", "", s)
    s = re.sub(r"[^A-Za-z]+", "_", s).strip("_").lower()
    if not s:
        return None
    return s


def load_env(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # schema A: {"obj_id": {"label": "...", ...}, ...}
    if isinstance(data, dict) and "objects" not in data:
        return {k: v for k, v in data.items() if isinstance(v, dict)}

    # schema B: {"objects": {"obj_id": {"openable":..., "pickable":..., ...}, ...}}
    if isinstance(data, dict) and isinstance(data.get("objects"), dict):
        normalized: dict[str, dict[str, Any]] = {}
        for obj_id, info in data["objects"].items():
            if not isinstance(info, dict):
                continue
            item = dict(info)
            item.setdefault("label", normalize_label_from_object_id(obj_id))
            normalized[obj_id] = item
        return normalized

    raise ValueError(f"Unsupported env data format: {path}")


def load_and_merge_env(paths: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        current = load_env(path)
        for obj_id, info in current.items():
            if obj_id not in merged:
                merged[obj_id] = {}
            incoming = dict(info)
            if "label" in merged[obj_id] and "label" in incoming:
                incoming.pop("label")
            merged[obj_id].update(incoming)
    return merged


def summarize_env(env_data: dict[str, Any], max_examples_per_label: int = 5) -> dict[str, Any]:
    label_to_ids: dict[str, list[str]] = defaultdict(list)
    for obj_id, info in env_data.items():
        label = info.get("label")
        if isinstance(label, str) and label.strip():
            label_to_ids[label.strip()].append(obj_id)

    counts = Counter({label: len(ids) for label, ids in label_to_ids.items()})
    return {
        "total_objects": len(env_data),
        "labeled_categories": len(label_to_ids),
        "label_counts": dict(counts.most_common()),
        "label_examples": {
            label: ids[:max_examples_per_label] for label, ids in sorted(label_to_ids.items(), key=lambda x: x[0])
        },
    }


def build_messages(
    *,
    env_intro: str,
    action_prompts: str,
    subtask_example: str,
    eval_example: str,
    env_data: dict[str, Any],
    env_summary: dict[str, Any],
    n_subtasks: int,
    scene_name: str,
) -> list[dict[str, str]]:
    system_prompt = f"""
You are an "indoor-scene subtask generator".
You must strictly rely on the provided environment data to generate executable, evaluable, and atomic subtasks.
Do not fabricate object categories that do not exist in the environment.

Environment description:
{env_intro}

Available actions:
{action_prompts}

Subtask examples:
{subtask_example}

Evaluation examples:
{eval_example}
""".strip()

    user_prompt = f"""
Based on the provided environment, generate {n_subtasks} different subtask templates.
Use "{scene_name}" as the scene name for all subtasks.

Output must be a JSON object in the format below:
{{
  "subtasks": [
    {{
      "subtask_id": "...",
      "task_type": "...",
      "instruction_template": "...",
      "task_intent": "...",
      "scene": "{scene_name}",
      "available_parameters": {{"object": ["..."]}},
      "input_resources": ["..."],
      "output_resources": ["..."],
      "action_space": ["move2", "pick_up"],
      "evaluation_checkpoints": {{"visible": 0.2, "at_object": 0.3, "done": 0.5}}
    }}
  ]
}}

Hard constraints:
1) Only use labels that truly exist in the environment as parameter candidates; do not invent categories.
2) Generate {n_subtasks} different task types (not just one template with object substitution).
3) The result set must cover all actions: move2, rotation, pick_up, put_down, open, close, turn_on, turn_off.
4) Each subtask should be atomic or at most a two-step composition, and action_space can only use:
   ["move2", "rotation", "pick_up", "put_down", "open", "close", "turn_on", "turn_off"]
5) Each subtask must provide explainable evaluation_checkpoints with weights summing to 1.0.
6) Output JSON only, with no extra explanation.

Environment summary (for quick reference):
{json.dumps(env_summary, ensure_ascii=False, indent=2)}

Complete environment data (JSON):
{json.dumps(env_data, ensure_ascii=False)}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_chat_completions(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    base_url: str = "https://api.openai.com/v1",
    timeout: int = 180,
) -> str:
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API request failed: HTTP {e.code}, detail={detail}") from e

    return data["choices"][0]["message"]["content"]


def generate_fallback_subtasks(env_summary: dict[str, Any], n_subtasks: int, scene_name: str) -> dict[str, Any]:
    labels = list(env_summary.get("label_counts", {}).keys()) or ["object"]

    task_pool: list[dict[str, Any]] = [
        {"task_type": "locate_object", "instruction_template": "Find a {object}.", "task_intent": "Locate target", "action_space": ["rotation"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_visible"], "evaluation_checkpoints": {"visible": 1.0}},
        {"task_type": "move_to_object", "instruction_template": "Move next to the {object}.", "task_intent": "Reach target", "action_space": ["move2"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}"], "evaluation_checkpoints": {"at_object": 1.0}},
        {"task_type": "observe_and_align", "instruction_template": "Rotate to observe, then align with the {object}.", "task_intent": "Align after scanning", "action_space": ["rotation"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_visible", "agent_aligned_to_{object}"], "evaluation_checkpoints": {"visible": 0.5, "aligned": 0.5}},
        {"task_type": "inspect_then_move", "instruction_template": "First observe the {object}, then move next to it.", "task_intent": "Perceive then navigate", "action_space": ["rotation", "move2"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_visible", "agent_at_{object}"], "evaluation_checkpoints": {"visible": 0.4, "at_object": 0.6}},
        {"task_type": "pick_object", "instruction_template": "Find the {object} and pick it up.", "task_intent": "Pick up target", "action_space": ["move2", "pick_up"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_in_hand"], "evaluation_checkpoints": {"at_object": 0.4, "picked": 0.6}},
        {"task_type": "pick_and_drop", "instruction_template": "Pick up the {object} and immediately put it down.", "task_intent": "Pick and place", "action_space": ["move2", "pick_up", "put_down"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_in_hand", "{object}_released"], "evaluation_checkpoints": {"picked": 0.5, "dropped": 0.5}},
        {"task_type": "approach_two_objects", "instruction_template": "Go to {object_a}, then go to {object_b}.", "task_intent": "Two-target navigation", "action_space": ["move2"], "input_resources": ["{object_a}_exists", "{object_b}_exists"], "output_resources": ["agent_at_{object_a}", "agent_at_{object_b}"], "evaluation_checkpoints": {"at_a": 0.5, "at_b": 0.5}},
        {"task_type": "open_object", "instruction_template": "Move to the {object} and open it.", "task_intent": "Open an openable object", "action_space": ["move2", "open"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_opened"], "evaluation_checkpoints": {"at_object": 0.4, "opened": 0.6}},
        {"task_type": "close_object", "instruction_template": "Move to the {object} and close it.", "task_intent": "Close an openable object", "action_space": ["move2", "close"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_closed"], "evaluation_checkpoints": {"at_object": 0.4, "closed": 0.6}},
        {"task_type": "open_then_close", "instruction_template": "Open the {object}, then close it.", "task_intent": "Open-close cycle", "action_space": ["move2", "open", "close"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_opened", "{object}_closed"], "evaluation_checkpoints": {"opened": 0.5, "closed": 0.5}},
        {"task_type": "turn_on_object", "instruction_template": "Move to the {object} and turn it on.", "task_intent": "Power on", "action_space": ["move2", "turn_on"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_on"], "evaluation_checkpoints": {"at_object": 0.4, "turned_on": 0.6}},
        {"task_type": "turn_off_object", "instruction_template": "Move to the {object} and turn it off.", "task_intent": "Power off", "action_space": ["move2", "turn_off"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_off"], "evaluation_checkpoints": {"at_object": 0.4, "turned_off": 0.6}},
        {"task_type": "turn_on_then_off", "instruction_template": "Turn on the {object}, then turn it off.", "task_intent": "Power toggle cycle", "action_space": ["move2", "turn_on", "turn_off"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_on", "{object}_off"], "evaluation_checkpoints": {"turned_on": 0.5, "turned_off": 0.5}},
        {"task_type": "rotate_scan_full", "instruction_template": "Rotate in place and confirm the {object} is visible.", "task_intent": "Full-direction scan", "action_space": ["rotation"], "input_resources": ["{object}_exists"], "output_resources": ["scan_completed", "{object}_visible"], "evaluation_checkpoints": {"scan": 0.4, "visible": 0.6}},
        {"task_type": "double_rotate_scan", "instruction_template": "Rotate twice and confirm the {object} is visible again.", "task_intent": "Repeated scan and reacquisition", "action_space": ["rotation"], "input_resources": ["{object}_exists"], "output_resources": ["scan_completed_twice", "{object}_visible_again"], "evaluation_checkpoints": {"scan": 0.5, "reacquired": 0.5}},
        {"task_type": "three_step_pick", "instruction_template": "Observe the {object}, move closer, then pick it up.", "task_intent": "Three-step perception-interaction", "action_space": ["rotation", "move2", "pick_up"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_visible", "agent_at_{object}", "{object}_in_hand"], "evaluation_checkpoints": {"visible": 0.2, "at_object": 0.3, "picked": 0.5}},
        {"task_type": "pick_navigate_drop", "instruction_template": "Pick up the {object}, navigate to a new position, then put it down.", "task_intent": "Carry and place", "action_space": ["move2", "pick_up", "put_down"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_in_hand", "agent_repositioned", "{object}_released"], "evaluation_checkpoints": {"picked": 0.3, "repositioned": 0.3, "dropped": 0.4}},
        {"task_type": "open_then_pick", "instruction_template": "Open the {object} area and then pick up a {object2}.", "task_intent": "Open then pick", "action_space": ["move2", "open", "pick_up"], "input_resources": ["{object}_exists", "{object2}_exists"], "output_resources": ["{object}_opened", "{object2}_in_hand"], "evaluation_checkpoints": {"opened": 0.4, "picked": 0.6}},
        {"task_type": "power_then_pick", "instruction_template": "Turn on the {object}, then pick up a {object2}.", "task_intent": "Power then interact", "action_space": ["move2", "turn_on", "pick_up"], "input_resources": ["{object}_exists", "{object2}_exists"], "output_resources": ["{object}_on", "{object2}_in_hand"], "evaluation_checkpoints": {"turned_on": 0.4, "picked": 0.6}},
        {"task_type": "close_then_turn_off", "instruction_template": "Close the {object} and turn off the {object2}.", "task_intent": "Dual-object shutdown", "action_space": ["move2", "close", "turn_off"], "input_resources": ["{object}_exists", "{object2}_exists"], "output_resources": ["{object}_closed", "{object2}_off"], "evaluation_checkpoints": {"closed": 0.5, "turned_off": 0.5}},
    ]

    subtasks: list[dict[str, Any]] = []
    for i in range(n_subtasks):
        tpl = task_pool[i % len(task_pool)]
        obj = labels[i % len(labels)]
        obj_b = labels[(i + 3) % len(labels)]

        params: dict[str, list[str]]
        if "{object_a}" in tpl["instruction_template"]:
            params = {"object_a": [obj], "object_b": [obj_b]}
        elif "{object2}" in tpl["instruction_template"]:
            params = {"object": [obj], "object2": [obj_b]}
        else:
            params = {"object": [obj]}

        subtasks.append(
            {
                "subtask_id": f"subtask_{tpl['task_type']}_{i+1:02d}",
                "task_type": tpl["task_type"],
                "instruction_template": tpl["instruction_template"],
                "task_intent": tpl["task_intent"],
                "scene": scene_name,
                "available_parameters": params,
                "input_resources": tpl["input_resources"],
                "output_resources": tpl["output_resources"],
                "action_space": tpl["action_space"],
                "evaluation_checkpoints": tpl["evaluation_checkpoints"],
            }
        )

    return {"subtasks": subtasks}


def validate_diversity(result: dict[str, Any], expected_num: int, min_unique_types: int) -> None:
    subtasks = result.get("subtasks", [])
    if len(subtasks) != expected_num:
        raise ValueError(f"Invalid subtask count: expected={expected_num}, got={len(subtasks)}")

    task_types = [s.get("task_type") for s in subtasks]
    unique_task_types = {t for t in task_types if isinstance(t, str) and t.strip()}
    if len(unique_task_types) < min_unique_types:
        raise ValueError(
            f"Insufficient task-type diversity: expected at least {min_unique_types}, got {len(unique_task_types)}"
        )


def validate_action_coverage(result: dict[str, Any]) -> None:
    required = {"move2", "rotation", "pick_up", "put_down", "open", "close", "turn_on", "turn_off"}
    seen = {a for s in result.get("subtasks", []) for a in s.get("action_space", [])}
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"Incomplete action coverage, missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtasks from environment data and prompt templates.")
    parser.add_argument(
        "--env-data",
        nargs="+",
        default=["env_data/touchdata.json"],
        help="Environment JSON paths. Multiple files supported; later files supplement/override earlier ones.",
    )
    parser.add_argument("--prompts-dir", default="prompts", help="Prompt directory (must include env.yaml/actions_prompts.yaml, etc.).")
    parser.add_argument("--scene", default="daily_life_scene", help="Scene field value in generated subtasks.")
    parser.add_argument("--num-subtasks", type=int, default=10, help="Number of subtasks to generate.")
    parser.add_argument("--output", default="outputs/subtasks_generated_10.json", help="Output JSON path.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Model name.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="API base url")
    parser.add_argument("--dry-run", action="store_true", help="Do not call API; generate with local fallback rules.")
    parser.add_argument("--min-unique-types", type=int, default=None, help="Minimum number of unique task_types; default equals num-subtasks.")
    parser.add_argument("--require-all-actions", action="store_true", help="Require complete action coverage in results.")
    args = parser.parse_args()

    env_data = load_and_merge_env([Path(p) for p in args.env_data])
    env_summary = summarize_env(env_data)

    if args.dry_run:
        result = generate_fallback_subtasks(env_summary, args.num_subtasks, args.scene)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please set OPENAI_API_KEY first.")

        prompts_dir = Path(args.prompts_dir)
        messages = build_messages(
            env_intro=load_text(prompts_dir / "env.yaml"),
            action_prompts=load_text(prompts_dir / "actions_prompts.yaml"),
            subtask_example=load_text(prompts_dir / "subtask_example.yaml"),
            eval_example=load_text(prompts_dir / "eval_gen_example.yaml"),
            env_data=env_data,
            env_summary=env_summary,
            n_subtasks=args.num_subtasks,
            scene_name=args.scene,
        )
        raw = call_chat_completions(
            api_key=api_key,
            model=args.model,
            messages=messages,
            base_url=args.base_url,
        )
        result = json.loads(raw)

    min_unique_types = args.min_unique_types if args.min_unique_types is not None else args.num_subtasks
    validate_diversity(result, args.num_subtasks, min_unique_types)
    if args.require_all_actions:
        validate_action_coverage(result)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(result.get('subtasks', []))} subtasks -> {out}")


if __name__ == "__main__":
    main()
