import argparse
import difflib
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml

PRED_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$')


@dataclass(frozen=True, order=True)
class Pred:
    name: str
    args: Tuple[str, ...] = ()

    def to_string(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(self.args)})"


@dataclass(frozen=True)
class GroundAction:
    name: str
    template_id: str
    pre: Tuple[Pred, ...]
    add: Tuple[Pred, ...]
    delete: Tuple[Pred, ...]
    meta: Tuple[Tuple[str, str], ...]
    source: str
    action_type: str


class Planner:
    def __init__(self, base_dir: Path, out_dir: Path, rules_path: Path, subtasks_path: Path):
        self.base_dir = base_dir
        self.out_dir = out_dir
        self.rules = yaml.safe_load(rules_path.read_text())
        self.tasks = json.loads((out_dir / 'tasks.json').read_text())
        self.templates = json.loads((out_dir / 'atomic_templates.json').read_text())['templates']
        self.subtasks = json.loads(subtasks_path.read_text())
        self.scene = json.loads((base_dir / 'touchdata.json').read_text())
        self.affordances = json.loads((base_dir / 'touchdata_objects.json').read_text())['objects']

        self.obj_alias = self.rules['normalization'].get('object_id_aliases', {})
        self.label_alias = self.rules['normalization'].get('label_aliases', {})
        self.transient_predicates = set(self.rules['state'].get('transient_predicates', []))
        self.container_labels = {self.norm_label(x) for x in self.rules.get('container_labels', [])}
        self.surface_labels = {self.norm_label(x) for x in self.rules.get('surface_labels', [])}
        self.hanger_labels = {self.norm_label(x) for x in self.rules.get('hanger_labels', [])}
        self.pluggable_labels = {self.norm_label(x) for x in self.rules.get('pluggable_labels', [])}
        self.suppressed_templates = set(self.rules.get('suppressed_templates', []))
        self.action_rules = self.rules.get('action_rules', {})
        self.allowed_primitive_templates = {
            'place_{object1}_on_{object2}',
            'place_{object1}_in_{object2}',
        }
        self.object_labels = {
            obj_id: self.norm_label((meta or {}).get('label', ''))
            for obj_id, meta in self.scene.items()
        }
        self.by_task = {t['task_id']: t for t in self.tasks}

    def normalize_token(self, text: str) -> str:
        x = text.strip()
        if self.rules['normalization'].get('normalize_case', True):
            x = x.lower()
        if self.rules['normalization'].get('remove_underscores', True):
            x = x.replace('_', '')
        if self.rules['normalization'].get('remove_spaces', True):
            x = x.replace(' ', '')
        if self.rules['normalization'].get('remove_hyphens', True):
            x = x.replace('-', '')
        return x

    def norm_label(self, label: str) -> str:
        raw = self.normalize_token(label)
        mapped = self.label_alias.get(raw, raw)
        return self.normalize_token(mapped)

    def resolve_object_id(self, obj_id: str) -> str:
        if obj_id in self.scene or obj_id in self.affordances:
            return obj_id
        if obj_id in self.obj_alias:
            aliased = self.obj_alias[obj_id]
            if aliased in self.scene or aliased in self.affordances:
                return aliased
        norm = self.normalize_token(obj_id)
        candidates = sorted(set(self.scene) | set(self.affordances))
        for cand in candidates:
            if self.normalize_token(cand) == norm:
                return cand
        # Fuzzy fallback for near-miss ids like BP_Cupboard_04_C_3 vs BP_BowlCupboard_04_C_3
        cand_norms = {cand: self.normalize_token(cand) for cand in candidates}
        close = difflib.get_close_matches(norm, list(cand_norms.values()), n=1, cutoff=0.80)
        if close:
            best_norm = close[0]
            for cand, c_norm in cand_norms.items():
                if c_norm == best_norm:
                    return cand
        return obj_id

    def parse_pred(self, text: str) -> Pred:
        text = text.strip()
        m = PRED_RE.match(text)
        if not m:
            return Pred(text, ())
        name = m.group(1).strip()
        arg_text = m.group(2).strip()
        if not arg_text:
            return Pred(name, ())
        args = tuple(self.resolve_object_id(a.strip()) for a in arg_text.split(','))
        return Pred(name, args)

    def instantiate_atom(self, pattern: str, binding: Dict[str, str]) -> Pred:
        s = pattern
        for k, v in binding.items():
            s = s.replace('{' + k + '}', v)
        pred = self.parse_pred(s)
        if not binding:
            return pred
        new_args = tuple(binding.get(arg, arg) for arg in pred.args)
        return Pred(pred.name, new_args)

    def instantiate_special_delete(self, pattern: str, binding: Dict[str, str]) -> Pred:
        s = pattern
        for k, v in binding.items():
            s = s.replace('{' + k + '}', v)
        m = PRED_RE.match(s)
        if not m:
            return Pred(s, ())
        name = m.group(1).strip()
        arg_text = m.group(2).strip()
        if not arg_text:
            return Pred(name, ())
        args = tuple(a.strip() for a in arg_text.split(','))
        return Pred(name, args)

    def state_from_strings(self, preds: Iterable[str]) -> frozenset:
        return frozenset(self.parse_pred(p) for p in preds)

    def state_to_strings(self, state: Iterable[Pred]) -> List[str]:
        return sorted(p.to_string() for p in state)

    def canonical_state_key(self, state: Iterable[Pred]) -> Tuple[str, ...]:
        filtered = []
        for p in state:
            if p.name in self.transient_predicates and self.rules['search'].get('ignore_transient_in_state_key', True):
                continue
            filtered.append(p.to_string())
        return tuple(sorted(filtered))

    def obj_label(self, obj_id: str) -> str:
        label = self.object_labels.get(obj_id, '')
        if label:
            return label
        raw = self.normalize_token(obj_id)
        heuristics = [
            ('bowlcupboard', 'cupboard'),
            ('cupboard', 'cupboard'),
            ('shoebox', 'storagebox'),
            ('shoesbox', 'storagebox'),
            ('sink', 'sink'),
            ('tablelamp', 'tablelamp'),
            ('lamp', 'lamp'),
            ('toy', 'toy'),
            ('bowl', 'bowl'),
            ('cup', 'cup'),
            ('plate', 'plate'),
            ('shoe', 'shoe'),
            ('desk', 'desk'),
        ]
        for needle, mapped in heuristics:
            if needle in raw:
                return self.norm_label(mapped)
        return ''

    def is_pickable(self, obj_id: str) -> bool:
        return bool(self.affordances.get(obj_id, {}).get('pickable', False))

    def is_openable(self, obj_id: str) -> bool:
        return bool(self.affordances.get(obj_id, {}).get('openable', False))

    def is_powerable(self, obj_id: str) -> bool:
        return bool(self.affordances.get(obj_id, {}).get('powerable', False))

    def object_matches_category(self, obj_id: str, category: str) -> bool:
        c = self.norm_label(category)
        label = self.obj_label(obj_id)
        if label == c:
            return True
        special = {
            'trashcan': {'trashcan', 'wastebasket'},
            'wastebasket': {'trashcan', 'wastebasket'},
            'storagebox': {'storagebox', 'shoescabinet', 'shoescabinet', 'shoebox'},
            'shoescabinet': {'storagebox', 'shoescabinet', 'shoebox'},
            'shoebox': {'storagebox', 'shoescabinet', 'shoebox'},
            'cabinet': {'cabinet', 'cupboard', 'bowlcupboard', 'shoescabinet', 'shoebox', 'storagebox'},
            'cupboard': {'cabinet', 'cupboard', 'bowlcupboard', 'shoescabinet', 'shoebox', 'storagebox'},
            'tablelamp': {'tablelamp', 'lamp', 'decorationlamp'},
            'lamp': {'tablelamp', 'lamp', 'decorationlamp'},
            'fan': {'fan', 'electricfan'},
            'electricfan': {'fan', 'electricfan'},
            'computer': {'computer', 'laptop'},
            'laptop': {'computer', 'laptop'},
            'coffeetable': {'coffeetable', 'coffeetable'},
        }
        return label in special.get(c, set())

    def task_scope(self, task: dict) -> Set[str]:
        objs: Set[str] = set()
        for side in ('initial_state', 'final_state'):
            for raw in task[side]['predicates']:
                pred = self.parse_pred(raw)
                for a in pred.args:
                    if a:
                        objs.add(a)
                if pred.name == 'exists' and pred.args:
                    objs.add(pred.args[0])
        return objs

    def preprocess_task_predicates(self, task: dict) -> Tuple[dict, List[str], List[str]]:
        new_task = json.loads(json.dumps(task))
        dropped: List[str] = []
        unsat: List[str] = []

        def keep_pred(raw: str) -> bool:
            pred = self.parse_pred(raw)
            if pred.name in {'open', 'closed'} and pred.args:
                obj = pred.args[0]
                if not self.is_openable(obj):
                    if self.rules['semantics'].get('drop_open_close_for_non_openable', True):
                        dropped.append(f"Dropped {raw} because {obj} is not openable.")
                        return False
            if pred.name in {'powered_on', 'powered_off'} and pred.args:
                obj = pred.args[0]
                if not self.is_powerable(obj):
                    if self.rules['semantics'].get('drop_power_preds_for_non_powerable', True):
                        dropped.append(f"Dropped {raw} because {obj} is not powerable.")
                        return False
            if pred.name in {'on', 'in'} and pred.args:
                obj = pred.args[0]
                if not self.is_pickable(obj):
                    unsat.append(f"Unsatisfied movable-goal predicate {raw}: {obj} is not pickable.")
            return True

        for side in ('initial_state', 'final_state'):
            kept = []
            for raw in new_task[side]['predicates']:
                if keep_pred(raw):
                    kept.append(raw)
            new_task[side]['predicates'] = kept
        return new_task, dropped, unsat

    def augment_initial_exists(self, initial: frozenset, task_scope: Set[str]) -> frozenset:
        s = set(initial)
        for obj in task_scope:
            if obj in self.scene or obj in self.affordances:
                s.add(Pred('exists', (obj,)))
        return frozenset(s)

    def split_delete_implied_extra(self, action: GroundAction) -> Tuple[List[Pred], List[Pred], List[Pred]]:
        if action.source == 'subtask':
            return list(action.delete), [], []
        cfg = self.action_rules.get(action.template_id, {})
        delete = [self.instantiate_special_delete(x, dict(action.meta)) for x in cfg.get('deletes', [])]
        implied = [self.instantiate_atom(x, dict(action.meta)) for x in cfg.get('implies', [])]
        extra_adds = [self.instantiate_atom(x, dict(action.meta)) for x in cfg.get('adds', [])]
        return delete, implied, extra_adds

    def build_subtask_delete_list(self, pre: Tuple[Pred, ...], add: Tuple[Pred, ...]) -> Tuple[Pred, ...]:
        add_set = set(add)
        return tuple(p for p in pre if p not in add_set and p.name != 'exists')

    def subtask_candidate_allowed(self, subtask: dict, role: str, obj_id: str) -> bool:
        stype = subtask.get('subtask_type', '')
        if 'acquire' in stype and not self.is_pickable(obj_id):
            return False
        if ('clean' in stype or 'wash' in stype) and role == 'r1' and not self.is_pickable(obj_id):
            return False
        if 'open_container' in stype and not self.is_openable(obj_id):
            return False
        if 'close_container' in stype and not self.is_openable(obj_id):
            return False
        if ('power_on' in stype or 'power_off' in stype or 'plug_in' in stype or 'unplug' in stype) and not self.is_powerable(obj_id):
            return False
        return True

    def subtask_roles_for_scope(self, subtask: dict, task_scope: Set[str]) -> Dict[str, List[str]]:
        role_map: Dict[str, List[str]] = {}
        for role, categories in subtask.get('roles', {}).items():
            matched = [
                o for o in task_scope
                if any(self.object_matches_category(o, c) for c in categories)
                and self.subtask_candidate_allowed(subtask, role, o)
            ]
            if not matched:
                return {}
            role_map[role] = sorted(set(matched))
        return role_map

    def expand_subtask(self, subtask: dict, task_scope: Set[str]) -> List[GroundAction]:
        role_map = self.subtask_roles_for_scope(subtask, task_scope)
        if not role_map:
            return []
        role_names = sorted(role_map)
        out: List[GroundAction] = []

        def backtrack(i: int, binding: Dict[str, str]):
            if i == len(role_names):
                if len(set(binding.values())) != len(binding):
                    return
                pre = tuple(self.instantiate_atom(x, binding) for x in subtask['pre_state'])
                add = tuple(self.instantiate_atom(x, binding) for x in subtask['post_state'])
                delete = self.build_subtask_delete_list(pre, add)
                suffix = '_'.join(binding[r] for r in role_names)
                out.append(GroundAction(
                    name=f"{subtask['subtask_type']}__{suffix}",
                    template_id=subtask['subtask_id'],
                    pre=pre,
                    add=add,
                    delete=delete,
                    meta=tuple(sorted(binding.items())),
                    source='subtask',
                    action_type=subtask['subtask_type'].split('_{')[0],
                ))
                return
            role = role_names[i]
            for obj in role_map[role]:
                binding[role] = obj
                backtrack(i + 1, binding)
                binding.pop(role, None)

        backtrack(0, {})
        return out

    def expand_place_template(self, template: dict, task_scope: Set[str]) -> List[GroundAction]:
        tid = template['template_id']
        if tid in self.suppressed_templates or tid not in self.allowed_primitive_templates:
            return []
        available = template.get('available_objects', {})
        if not isinstance(available, dict):
            return []
        k1, k2 = list(available.keys())
        allowed1 = list(available[k1])
        allowed2 = list(available[k2])

        def match_token_to_objects(token: str, require_pickable: bool = False) -> List[str]:
            resolved = self.resolve_object_id(token)
            objs: List[str] = []
            if resolved in task_scope:
                if (not require_pickable) or self.is_pickable(resolved):
                    objs.append(resolved)
            else:
                for o in task_scope:
                    if require_pickable and not self.is_pickable(o):
                        continue
                    if self.object_matches_category(o, token):
                        objs.append(o)
            return objs

        cands1: List[str] = []
        for tok in allowed1:
            cands1.extend(match_token_to_objects(tok, require_pickable=True))
        cands2: List[str] = []
        for tok in allowed2:
            cands2.extend(match_token_to_objects(tok, require_pickable=False))

        out: List[GroundAction] = []
        for obj1 in sorted(set(cands1)):
            for obj2 in sorted(set(cands2)):
                if obj1 == obj2:
                    continue
                binding = {k1: obj1, k2: obj2}
                pre = tuple(self.instantiate_atom(x, binding) for x in template['pre_state'])
                add = tuple(self.instantiate_atom(x, binding) for x in template['post_state'])
                delete, implied, extra_adds = self.split_delete_implied_extra(
                    GroundAction(tid, tid, (), (), (), tuple(sorted(binding.items())), 'primitive', 'place')
                )
                out.append(GroundAction(
                    name=tid.replace('{' + k1 + '}', obj1).replace('{' + k2 + '}', obj2),
                    template_id=tid,
                    pre=pre,
                    add=add,
                    delete=tuple(delete + implied + extra_adds),
                    meta=tuple(sorted(binding.items())),
                    source='primitive',
                    action_type='place',
                ))
        return out

    def build_actions_for_task(self, task: dict, task_scope: Set[str]) -> List[GroundAction]:
        actions: List[GroundAction] = []
        for st in self.subtasks:
            actions.extend(self.expand_subtask(st, task_scope))
        for template in self.templates:
            actions.extend(self.expand_place_template(template, task_scope))
        if self.rules['search'].get('include_only_goal_relevant_templates', True):
            actions = self.filter_goal_relevant_actions(task, actions)
        return actions

    def filter_goal_relevant_actions(self, task: dict, actions: List[GroundAction]) -> List[GroundAction]:
        goal = self.state_from_strings(task['final_state']['predicates'])
        required = set(goal)
        changed = True
        while changed:
            changed = False
            for a in actions:
                adds = set(a.add)
                delete, implied, extra_adds = self.split_delete_implied_extra(a)
                _ = delete
                adds.update(implied)
                adds.update(extra_adds)
                if required & adds:
                    for p in a.pre:
                        if p not in required:
                            required.add(p)
                            changed = True
        keep = []
        for a in actions:
            adds = set(a.add)
            _, implied, extra_adds = self.split_delete_implied_extra(a)
            adds.update(implied)
            adds.update(extra_adds)
            if required & adds:
                keep.append(a)
        return keep

    def bound_objects(self, action: GroundAction) -> Set[str]:
        return {v for _, v in action.meta}

    def wildcard_delete(self, state: Set[Pred], pat: Pred) -> Set[Pred]:
        out = set()
        for p in state:
            if p.name != pat.name or len(p.args) != len(pat.args):
                out.add(p)
                continue
            ok = True
            for got, want in zip(p.args, pat.args):
                if want != '*' and got != want:
                    ok = False
                    break
            if not ok:
                out.add(p)
        return out

    def remove_mutex_conflicts(self, state: Set[Pred], new_pred: Pred) -> Set[Pred]:
        for group in self.rules['state'].get('mutex_groups', []):
            if new_pred.name in group:
                for other in group:
                    if other != new_pred.name:
                        state.discard(Pred(other, new_pred.args if new_pred.args else ()))
        if new_pred.name == 'holding':
            for p in list(state):
                if p.name == 'holding' and p != new_pred:
                    state.discard(p)
            state.discard(Pred('empty', ()))
        if new_pred.name == 'empty':
            for p in list(state):
                if p.name == 'holding':
                    state.discard(p)
        if new_pred.name in {'holding', 'on', 'in'} and new_pred.args:
            obj = new_pred.args[0]
            for p in list(state):
                if p.name in {'holding', 'on', 'in'} and p.args and p.args[0] == obj and p != new_pred:
                    state.discard(p)
        return state

    def valid_action(self, state: Set[Pred], action: GroundAction, last_action_type: Optional[str]) -> bool:
        if any(p not in state for p in action.pre):
            return False
        if self.rules['search'].get('disallow_consecutive_same_action_type', False) and last_action_type == action.action_type:
            return False
        if self.rules['semantics'].get('require_exists_for_all_actions', True):
            for obj in self.bound_objects(action):
                if Pred('exists', (obj,)) not in state:
                    return False
        if action.source == 'subtask':
            # Subtask grounding already applies affordance/category filters.
            return True
        b = dict(action.meta)
        if action.template_id == 'place_{object1}_on_{object2}':
            return True
        if action.template_id == 'place_{object1}_in_{object2}':
            obj2 = b['object2']
            if self.rules['semantics'].get('require_open_container_for_place_in', True) and self.is_openable(obj2):
                return Pred('open', (obj2,)) in state
            return True
        return True

    def apply_action(self, state: frozenset, action: GroundAction) -> frozenset:
        s = set(state)
        delete, implied, extra_adds = self.split_delete_implied_extra(action)
        for d in delete:
            s = self.wildcard_delete(s, d)
        for p in implied:
            s = self.remove_mutex_conflicts(s, p)
            s.add(p)
        for p in action.add:
            s = self.remove_mutex_conflicts(s, p)
            s.add(p)
        for p in extra_adds:
            s = self.remove_mutex_conflicts(s, p)
            s.add(p)
        if any(p.name == 'holding' for p in s):
            s.discard(Pred('empty', ()))
        elif self.rules['state'].get('singleton_empty', True):
            s.add(Pred('empty', ()))
        return frozenset(s)

    def goal_satisfied(self, state: frozenset, goal: frozenset) -> bool:
        return goal.issubset(state)

    def enumerate_goal_paths(self, nodes: List[dict], dag_edges: List[dict], goal_nodes: List[str]) -> List[List[dict]]:
        adj = defaultdict(list)
        for e in dag_edges:
            adj[e['from']].append(e)
        goal_set = set(goal_nodes)
        out: List[List[dict]] = []

        def dfs(node_id: str, path: List[dict]):
            if node_id in goal_set:
                out.append(list(path))
            for e in adj.get(node_id, []):
                path.append({'from': e['from'], 'to': e['to'], 'action': e['action'], 'source': e['source']})
                dfs(e['to'], path)
                path.pop()

        dfs('n0', [])
        return out

    def search_task(self, task_id: str) -> dict:
        raw_task = self.by_task[task_id]
        task_scope = self.task_scope(raw_task)
        task, dropped_predicates, static_unsat = self.preprocess_task_predicates(raw_task)
        initial = self.augment_initial_exists(self.state_from_strings(task['initial_state']['predicates']), task_scope)
        goal = self.state_from_strings(task['final_state']['predicates'])
        actions = self.build_actions_for_task(task, task_scope)
        max_depth = int(self.rules['search']['max_depth'])
        max_nodes = int(self.rules['search']['max_nodes'])

        nodes = [{'id': 'n0', 'depth': 0, 'state': self.state_to_strings(initial), 'is_goal': self.goal_satisfied(initial, goal)}]
        dag_edges: List[dict] = []
        equivalence_edges: List[dict] = []
        key_to_node = {self.canonical_state_key(initial): 'n0'}
        q = deque([('n0', initial, 0, None)])

        while q and len(nodes) < max_nodes:
            node_id, state, depth, last_action_type = q.popleft()
            if self.goal_satisfied(state, goal) and self.rules['search'].get('stop_expanding_at_goal', True):
                continue
            if depth >= max_depth:
                continue
            cur_key = self.canonical_state_key(state)
            for action in actions:
                if not self.valid_action(set(state), action, last_action_type):
                    continue
                nxt = self.apply_action(state, action)
                nxt_key = self.canonical_state_key(nxt)
                if nxt_key == cur_key:
                    continue
                if nxt_key in key_to_node:
                    equivalence_edges.append({
                        'from': node_id,
                        'to': key_to_node[nxt_key],
                        'action': action.name,
                        'source': action.source,
                        'type': 'equivalence',
                    })
                    continue
                new_id = f"n{len(nodes)}"
                key_to_node[nxt_key] = new_id
                nodes.append({
                    'id': new_id,
                    'depth': depth + 1,
                    'state': self.state_to_strings(nxt),
                    'is_goal': self.goal_satisfied(nxt, goal),
                })
                dag_edges.append({
                    'from': node_id,
                    'to': new_id,
                    'action': action.name,
                    'source': action.source,
                    'type': 'transition',
                })
                q.append((new_id, nxt, depth + 1, action.action_type))

        goal_nodes = [n['id'] for n in nodes if n['is_goal']]
        goal_paths = self.enumerate_goal_paths(nodes, dag_edges, goal_nodes) if self.rules['search'].get('export_all_goal_paths', True) else []

        return {
            'task_id': task_id,
            'description': raw_task.get('description', ''),
            'output_language': self.rules.get('output_language', 'English'),
            'is_dag': True,
            'task_scope': sorted(task_scope),
            'preprocess': {
                'dropped_predicates': dropped_predicates,
                'static_unsatisfied_constraints': static_unsat,
            },
            'initial_state': self.state_to_strings(initial),
            'goal_state': self.state_to_strings(goal),
            'nodes': nodes,
            'dag_edges': dag_edges,
            'equivalence_edges': equivalence_edges,
            'goal_nodes': goal_nodes,
            'goal_paths': goal_paths,
            'has_solution': bool(goal_nodes) and not static_unsat,
            'node_count': len(nodes),
            'transition_edge_count': len(dag_edges),
            'equivalence_edge_count': len(equivalence_edges),
            'grounded_action_count': len(actions),
            'grounded_subtask_count': sum(1 for a in actions if a.source == 'subtask'),
            'grounded_place_primitive_count': sum(1 for a in actions if a.source == 'primitive'),
        }

    def search_all(self, output_dir: Path) -> List[dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = []
        for task in self.tasks:
            result = self.search_task(task['task_id'])
            (output_dir / f"{task['task_id']}_dag.json").write_text(json.dumps(result, indent=2))
            summary.append({
                'task_id': task['task_id'],
                'has_solution': result['has_solution'],
                'node_count': result['node_count'],
                'transition_edge_count': result['transition_edge_count'],
                'equivalence_edge_count': result['equivalence_edge_count'],
                'grounded_action_count': result['grounded_action_count'],
                'goal_path_count': len(result['goal_paths']),
                'static_unsatisfied_constraints': result['preprocess']['static_unsatisfied_constraints'],
                'dropped_predicates': result['preprocess']['dropped_predicates'],
            })
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='./env_data')
    ap.add_argument('--out-dir', default='./outputs')
    ap.add_argument('--rules', default='./prompts/planner_rules.yaml')
    ap.add_argument('--subtasks', default='./outputs/subtask_templates_compressed.json')
    ap.add_argument('--task-id', default=None)
    ap.add_argument('--output-dir', default='./dag_outputs_v2')
    args = ap.parse_args()

    planner = Planner(Path(args.base_dir), Path(args.out_dir), Path(args.rules), Path(args.subtasks))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.task_id:
        result = planner.search_task(args.task_id)
        out = outdir / f'{args.task_id}_dag.json'
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({
            'task_id': args.task_id,
            'has_solution': result['has_solution'],
            'node_count': result['node_count'],
            'transition_edge_count': result['transition_edge_count'],
            'equivalence_edge_count': result['equivalence_edge_count'],
            'grounded_action_count': result['grounded_action_count'],
            'goal_path_count': len(result['goal_paths']),
            'output': str(out),
        }, indent=2))
    else:
        summary = planner.search_all(outdir)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
