from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_env(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
你是一个“室内场景子任务生成器”。
你必须严格依据给定环境数据生成可执行、可评估、原子化的子任务，不要虚构不存在的物体类别。

下面是环境说明：
{env_intro}

下面是可用动作说明：
{action_prompts}

下面是子任务示例：
{subtask_example}

下面是评估示例：
{eval_example}
""".strip()

    user_prompt = f"""
请根据提供的环境，生成 {n_subtasks} 个不同的子任务模板，场景名统一为 "{scene_name}"。

输出必须是一个 JSON 对象，格式如下：
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

硬性约束：
1) 只使用环境中真实存在的 label 作为参数候选，不要生成不存在的类别。
2) 必须生成 {n_subtasks} 种不同 task_type（不能只是同一模板替换object）。
3) 所有动作都要在结果集中覆盖：move, move2, rotation, pick_up, put_down, open, close, turn_on, turn_off。
4) 每个子任务尽量原子化或两步组合，且 action_space 只能从以下动作名中选：
   ["move", "move2", "rotation", "pick_up", "put_down", "open", "close", "turn_on", "turn_off"]
5) 每个子任务都要给出可解释的 evaluation_checkpoints，权重和为 1.0。
6) 只输出 JSON，不要输出额外解释。

环境统计摘要（便于快速参考）：
{json.dumps(env_summary, ensure_ascii=False, indent=2)}

完整环境数据（JSON）：
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
        {"task_type": "locate_object", "instruction_template": "找到一个{object}。", "task_intent": "定位目标", "action_space": ["rotation"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_visible"], "evaluation_checkpoints": {"visible": 1.0}},
        {"task_type": "move_to_object", "instruction_template": "移动到{object}旁边。", "task_intent": "到达目标", "action_space": ["move2"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}"], "evaluation_checkpoints": {"at_object": 1.0}},
        {"task_type": "strafe_then_observe", "instruction_template": "先侧向移动，再找到{object}。", "task_intent": "位姿变化后观察", "action_space": ["move", "rotation"], "input_resources": ["{object}_exists"], "output_resources": ["agent_repositioned", "{object}_visible"], "evaluation_checkpoints": {"repositioned": 0.4, "visible": 0.6}},
        {"task_type": "inspect_then_move", "instruction_template": "先观察到{object}，再移动到它旁边。", "task_intent": "感知后导航", "action_space": ["rotation", "move2"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_visible", "agent_at_{object}"], "evaluation_checkpoints": {"visible": 0.4, "at_object": 0.6}},
        {"task_type": "pick_object", "instruction_template": "找到{object}并将其拿起。", "task_intent": "拾取目标", "action_space": ["move2", "pick_up"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_in_hand"], "evaluation_checkpoints": {"at_object": 0.4, "picked": 0.6}},
        {"task_type": "pick_and_drop", "instruction_template": "拿起{object}后立刻放下。", "task_intent": "抓取放置", "action_space": ["move2", "pick_up", "put_down"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_in_hand", "{object}_released"], "evaluation_checkpoints": {"picked": 0.5, "dropped": 0.5}},
        {"task_type": "approach_two_objects", "instruction_template": "先到{object_a}，再到{object_b}。", "task_intent": "双目标导航", "action_space": ["move2"], "input_resources": ["{object_a}_exists", "{object_b}_exists"], "output_resources": ["agent_at_{object_a}", "agent_at_{object_b}"], "evaluation_checkpoints": {"at_a": 0.5, "at_b": 0.5}},
        {"task_type": "open_object", "instruction_template": "移动到{object}并打开它。", "task_intent": "开启可开合物", "action_space": ["move2", "open"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_opened"], "evaluation_checkpoints": {"at_object": 0.4, "opened": 0.6}},
        {"task_type": "close_object", "instruction_template": "移动到{object}并关闭它。", "task_intent": "关闭可开合物", "action_space": ["move2", "close"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_closed"], "evaluation_checkpoints": {"at_object": 0.4, "closed": 0.6}},
        {"task_type": "open_then_close", "instruction_template": "先打开{object}再将其关闭。", "task_intent": "开合完整流程", "action_space": ["move2", "open", "close"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_opened", "{object}_closed"], "evaluation_checkpoints": {"opened": 0.5, "closed": 0.5}},
        {"task_type": "turn_on_object", "instruction_template": "移动到{object}并将其打开电源。", "task_intent": "开电源", "action_space": ["move2", "turn_on"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_on"], "evaluation_checkpoints": {"at_object": 0.4, "turned_on": 0.6}},
        {"task_type": "turn_off_object", "instruction_template": "移动到{object}并将其关闭电源。", "task_intent": "关电源", "action_space": ["move2", "turn_off"], "input_resources": ["{object}_exists"], "output_resources": ["agent_at_{object}", "{object}_off"], "evaluation_checkpoints": {"at_object": 0.4, "turned_off": 0.6}},
        {"task_type": "turn_on_then_off", "instruction_template": "先打开{object}电源，再关闭它。", "task_intent": "电源开关闭环", "action_space": ["move2", "turn_on", "turn_off"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_on", "{object}_off"], "evaluation_checkpoints": {"turned_on": 0.5, "turned_off": 0.5}},
        {"task_type": "rotate_scan_full", "instruction_template": "原地旋转并确认看到{object}。", "task_intent": "全向扫描", "action_space": ["rotation"], "input_resources": ["{object}_exists"], "output_resources": ["scan_completed", "{object}_visible"], "evaluation_checkpoints": {"scan": 0.4, "visible": 0.6}},
        {"task_type": "move_back_and_forth", "instruction_template": "向前移动后返回并再次看到{object}。", "task_intent": "往返移动与再识别", "action_space": ["move", "rotation"], "input_resources": ["{object}_exists"], "output_resources": ["agent_repositioned", "{object}_visible_again"], "evaluation_checkpoints": {"moved": 0.5, "reacquired": 0.5}},
        {"task_type": "three_step_pick", "instruction_template": "先观察{object}，再靠近并拿起它。", "task_intent": "三步感知交互", "action_space": ["rotation", "move2", "pick_up"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_visible", "agent_at_{object}", "{object}_in_hand"], "evaluation_checkpoints": {"visible": 0.2, "at_object": 0.3, "picked": 0.5}},
        {"task_type": "pick_move_drop", "instruction_template": "拿起{object}后移动一步再放下。", "task_intent": "携带后放置", "action_space": ["move2", "pick_up", "move", "put_down"], "input_resources": ["{object}_exists"], "output_resources": ["{object}_in_hand", "agent_repositioned", "{object}_released"], "evaluation_checkpoints": {"picked": 0.3, "moved": 0.3, "dropped": 0.4}},
        {"task_type": "open_then_pick", "instruction_template": "打开{object}附近区域后拿起一个{object2}。", "task_intent": "开合后抓取", "action_space": ["move2", "open", "pick_up"], "input_resources": ["{object}_exists", "{object2}_exists"], "output_resources": ["{object}_opened", "{object2}_in_hand"], "evaluation_checkpoints": {"opened": 0.4, "picked": 0.6}},
        {"task_type": "power_then_pick", "instruction_template": "打开{object}电源后，拿起{object2}。", "task_intent": "通电后交互", "action_space": ["move2", "turn_on", "pick_up"], "input_resources": ["{object}_exists", "{object2}_exists"], "output_resources": ["{object}_on", "{object2}_in_hand"], "evaluation_checkpoints": {"turned_on": 0.4, "picked": 0.6}},
        {"task_type": "close_then_turn_off", "instruction_template": "关闭{object}并将{object2}电源关闭。", "task_intent": "双目标关闭", "action_space": ["move2", "close", "turn_off"], "input_resources": ["{object}_exists", "{object2}_exists"], "output_resources": ["{object}_closed", "{object2}_off"], "evaluation_checkpoints": {"closed": 0.5, "turned_off": 0.5}},
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
        raise ValueError(f"subtasks数量不正确: expected={expected_num}, got={len(subtasks)}")

    task_types = [s.get("task_type") for s in subtasks]
    unique_task_types = {t for t in task_types if isinstance(t, str) and t.strip()}
    if len(unique_task_types) < min_unique_types:
        raise ValueError(
            f"任务类型不够多样: 期望至少{min_unique_types}个不同task_type, 实际={len(unique_task_types)}"
        )


def validate_action_coverage(result: dict[str, Any]) -> None:
    required = {"move", "move2", "rotation", "pick_up", "put_down", "open", "close", "turn_on", "turn_off"}
    seen = {a for s in result.get("subtasks", []) for a in s.get("action_space", [])}
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"动作覆盖不完整，缺少: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="基于环境与prompt模板调用大模型生成子任务。")
    parser.add_argument("--env-data", default="env_data/touchdata.json", help="环境JSON文件路径")
    parser.add_argument("--prompts-dir", default="prompts", help="prompt目录，需包含env.yaml/actions_prompts.yaml等")
    parser.add_argument("--scene", default="daily_life_scene", help="子任务中的scene字段")
    parser.add_argument("--num-subtasks", type=int, default=10, help="生成子任务数量")
    parser.add_argument("--output", default="outputs/subtasks_generated_10.json", help="输出JSON路径")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="模型名")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="API base url")
    parser.add_argument("--dry-run", action="store_true", help="不调用API，使用本地回退规则生成示例")
    parser.add_argument("--min-unique-types", type=int, default=None, help="至少需要多少种不同task_type，默认等于任务数量")
    parser.add_argument("--require-all-actions", action="store_true", help="要求结果覆盖全部动作")
    args = parser.parse_args()

    env_data = load_env(Path(args.env_data))
    env_summary = summarize_env(env_data)

    if args.dry_run:
        result = generate_fallback_subtasks(env_summary, args.num_subtasks, args.scene)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")

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
