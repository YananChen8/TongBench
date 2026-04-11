#!/usr/bin/env python3
"""
Generate composite tasks by combining subtasks with strict IO alignment.

Notes:
- Output is de-lexicalized: object names are replaced with placeholder keys
  (object, object_a, object_b, object2, ...).
- task_id is sequential: 1,2,3,...
- io_alignment is omitted.
"""

from __future__ import annotations

import argparse
from asyncio import tasks
import itertools
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

PLACEHOLDER_RE = re.compile(r"\{(.*?)\}")


@dataclass(frozen=True)
class SubtaskInstance:
    subtask_id: str
    task_type: str
    instruction_template: str
    task_intent: str
    input_templates: Tuple[str, ...]
    output_templates: Tuple[str, ...]


def _expand_subtask(subtask: Dict) -> List[SubtaskInstance]:
    available = subtask.get("available_parameters", {}) or {}
    keys = list(available.keys())
    values_list = [available[k] for k in keys]

    if not keys:
        combos = [dict()]
    else:
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*values_list)]

    instances: List[SubtaskInstance] = []
    for _ in combos:
        instances.append(
            SubtaskInstance(
                subtask_id=subtask.get("subtask_id", ""),
                task_type=subtask.get("task_type", ""),
                instruction_template=subtask.get("instruction_template", ""),
                task_intent=subtask.get("task_intent", ""),
                input_templates=tuple(subtask.get("input_resources", [])),
                output_templates=tuple(subtask.get("output_resources", [])),
            )
        )
    return instances


def _collect_initial_resources(subtasks: Sequence[Dict]) -> Set[str]:
    initial: Set[str] = set()
    for s in subtasks:
        for res in s.get("input_resources", []) or []:
            if res.endswith("_exists") or not PLACEHOLDER_RE.search(res):
                initial.add(res)
    return initial


def _build_long_intent(intents: Sequence[str]) -> str:
    return " -> ".join([i for i in intents if i])


def _infer_output_templates(output_templates: Sequence[str]) -> Set[str]:
    inferred: Set[str] = set()
    for out in output_templates:
        for key in PLACEHOLDER_RE.findall(out):
            if key.startswith("object"):
                inferred.add(f"{{{key}}}_exists")
    return inferred


def _generate_one_task(
    instances: Sequence[SubtaskInstance],
    initial_resources: Set[str],
    rng: random.Random,
    min_len: int,
    max_len: int,
    max_attempts: int = 2000,
) -> Dict:
    length = rng.randint(min_len, max_len)

    for _ in range(max_attempts):
        resource_pool = set(initial_resources)
        sequence: List[SubtaskInstance] = []

        start_candidates = [
            i for i in instances if set(i.input_templates).issubset(resource_pool)
        ]
        if not start_candidates:
            continue
        first = rng.choice(start_candidates)
        sequence.append(first)
        resource_pool.update(first.output_templates)
        resource_pool.update(_infer_output_templates(first.output_templates))

        while len(sequence) < length:
            prev = sequence[-1]
            prev_effective = set(prev.output_templates) | _infer_output_templates(
                prev.output_templates
            )
            candidates = []
            for inst in instances:
                if not set(inst.input_templates).issubset(resource_pool):
                    continue
                if not (set(inst.input_templates) & prev_effective):
                    continue
                candidates.append(inst)

            if not candidates:
                break

            nxt = rng.choice(candidates)
            sequence.append(nxt)
            resource_pool.update(nxt.output_templates)
            resource_pool.update(_infer_output_templates(nxt.output_templates))

        if len(sequence) >= 2:
            break

    if len(sequence) < 2:
        raise RuntimeError("Failed to generate a task with >=2 subtasks. Increase data overlap.")

    return {
        "long_intent": _build_long_intent([s.task_intent for s in sequence]),
        "subtasks": [
            {
                "subtask_id": s.subtask_id,
                "task_type": s.task_type,
                "params": {k: k for k in PLACEHOLDER_RE.findall(s.instruction_template)},
                "instruction": s.instruction_template,
                "task_intent": s.task_intent,
                "input_resources": list(s.input_templates),
                "output_resources": list(s.output_templates),
            }
            for s in sequence
        ],
    }


def generate_tasks(
    subtasks: Sequence[Dict],
    num_tasks: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    instances: List[SubtaskInstance] = []
    for s in subtasks:
        instances.extend(_expand_subtask(s))

    initial_resources = _collect_initial_resources(subtasks)

    tasks: List[Dict] = []
    for _ in range(num_tasks):
        tasks.append(
            _generate_one_task(
                instances=instances,
                initial_resources=initial_resources,
                rng=rng,
                min_len=min_len,
                max_len=max_len,
            )
        )

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate composite tasks from subtasks.")
    parser.add_argument("--subtasks", required=True, help="Comma-separated subtasks JSON files")
    parser.add_argument("--out", required=True, help="Output path for tasks JSON")
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subtasks_all: List[Dict] = []
    for path in [p for p in args.subtasks.split(",") if p]:
        with open(path, "r", encoding="utf-8") as f:
            subtasks_all.extend(json.load(f).get("subtasks", []))

    tasks = generate_tasks(
        subtasks=subtasks_all,
        num_tasks=args.num_tasks,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed,
    )

    for i, t in enumerate(tasks, start=1):
        t["task_id"] = i

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tasks": tasks}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


# python outputs/generate_tasks.py \
#   --subtasks outputs/subtasks_generated_10.json,outputs/subtasks_generated_20.json,outputs/subtasks_generated_30.json \
#   --out outputs/tasks_generated.json \
#   --num-tasks 10 \
#   --min-len 4 \
#   --max-len 6
