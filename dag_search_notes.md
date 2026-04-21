# DAG Task Search Notes

Files:
- `planner_rules.yaml`: complete rule file for state normalization, affordance checks, mutex rules, action semantics, and search limits.
- `dag_task_search.py`: exhaustive task-grounded DAG builder.
- `dag_outputs_final/summary.json`: run summary on the uploaded `tasks.json`.

Design:
- Transition edges only point to newly created states.
- If an action reaches a state that already exists, the code records an `equivalence_edge` instead of creating a second node.
- This keeps the transition graph acyclic while still preserving duplicate-state information.
- Search is task-scoped: only objects that appear in the task initial or goal predicates are grounded.
- The search also uses a goal-dependency closure so that irrelevant action templates are not expanded.

Important behavior:
- Scene metadata is used to auto-add missing `exists(object)` predicates for task-scope objects present in the scene.
- Object-ID typos are normalized through alias rules.
- Affordances are treated as authoritative for `pick_up`, `open/close`, and `switch_on/switch_off`.
- `place_in` requires the target container to be open when the target is openable.
- `place`, `drop`, and `hand_over` are suppressed by default because they remove location information and make the graph under-specified.

Run:
```bash
python dag_task_search.py --task-id task_03 --output-dir dag_outputs
python dag_task_search.py --output-dir dag_outputs
```

Output per task:
- `nodes`: canonical states
- `edges`: transition edges in the DAG
- `equivalence_edges`: links to already-seen equivalent states
- `goal_paths`: all root-to-goal paths in the DAG
- `topological_order`: DAG order over transition nodes
