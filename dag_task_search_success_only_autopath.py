import argparse
import difflib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml
# import debugpy
# try:
#     debugpy.configure({"subProcess": False})
# except Exception:
#     pass

# 下面再放原来的 import

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

    def resolve_task_id(self, task_id: str) -> str:
        if task_id in self.by_task:
            return task_id
        m = re.match(r'^(task_)(\d+)$', task_id)
        if m:
            prefix, num = m.groups()
            cands = [
                f'{prefix}{int(num)}',
                f'{prefix}{int(num):02d}',
                f'{prefix}{int(num):03d}',
                f'{prefix}{int(num):04d}',
            ]
            for cand in cands:
                if cand in self.by_task:
                    return cand
        raise KeyError(f"Task id '{task_id}' not found. Available tasks: {sorted(self.by_task)[:20]}")

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

    def full_state_key(self, state: Iterable[Pred]) -> Tuple[str, ...]:
        return tuple(sorted(p.to_string() for p in state))

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
            ('sofa', 'sofa'),
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
            'storagebox': {'storagebox', 'shoescabinet', 'shoebox', 'suitcase'},
            'shoescabinet': {'storagebox', 'shoescabinet', 'shoebox', 'suitcase'},
            'shoebox': {'storagebox', 'shoescabinet', 'shoebox', 'suitcase'},
            'suitcase': {'storagebox', 'shoescabinet', 'shoebox', 'suitcase'},
            'cabinet': {'cabinet', 'cupboard', 'bowlcupboard', 'shoescabinet', 'shoebox', 'storagebox'},
            'cupboard': {'cabinet', 'cupboard', 'bowlcupboard', 'shoescabinet', 'shoebox', 'storagebox'},
            'tablelamp': {'tablelamp', 'lamp', 'decorationlamp'},
            'lamp': {'tablelamp', 'lamp', 'decorationlamp'},
            'fan': {'fan', 'electricfan'},
            'electricfan': {'fan', 'electricfan'},
            'computer': {'computer', 'laptop'},
            'laptop': {'computer', 'laptop'},
            'coffeetable': {'coffeetable', 'coffee_table'},
            'sofa': {'sofa'},
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
                _, implied, extra_adds = self.split_delete_implied_extra(a)
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

        if new_pred.name in {'in_view', 'with_reach', 'pointed_at'}:
            for p in list(state):
                if p.name == new_pred.name and p != new_pred:
                    state.discard(p)

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
            return True
        if action.template_id == 'place_{object1}_on_{object2}':
            return True
        if action.template_id == 'place_{object1}_in_{object2}':
            obj2 = dict(action.meta)['object2']
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

    def enumerate_success_traces(self, initial: frozenset, goal: frozenset, actions: List[GroundAction], max_depth: int,
                                 max_traces: int = 10000, max_path_nodes: int = 10) -> List[List[dict]]:
        traces: List[List[dict]] = []

        def dfs(state: frozenset, depth: int, last_action_type: Optional[str],
                state_keys_on_path: Set[Tuple[str, ...]], trace: List[dict]) -> None:
            if len(traces) >= max_traces:
                return
            if self.goal_satisfied(state, goal):
                traces.append(list(trace))
                return
            if (depth + 1) >= max_path_nodes:
                return
            if depth >= max_depth:
                return

            cur_key = self.full_state_key(state)
            for action in actions:
                if not self.valid_action(set(state), action, last_action_type):
                    continue
                nxt = self.apply_action(state, action)
                nxt_key = self.full_state_key(nxt)
                if nxt_key == cur_key:
                    continue
                if nxt_key in state_keys_on_path:
                    continue
                trace.append({
                    'action': action.name,
                    'source': action.source,
                    'action_type': action.action_type,
                    'state': self.state_to_strings(nxt),
                })
                state_keys_on_path.add(nxt_key)
                dfs(nxt, depth + 1, action.action_type, state_keys_on_path, trace)
                state_keys_on_path.remove(nxt_key)
                trace.pop()

        dfs(initial, 0, None, {self.full_state_key(initial)}, [])
        return traces

    def build_success_prefix_graph(self, initial: frozenset, success_traces: List[List[dict]]) -> Tuple[List[dict], List[dict], List[str]]:
        nodes: List[dict] = [{
            'id': 'n0',
            'depth': 0,
            'state': self.state_to_strings(initial),
            'is_goal': False,
            'prefix_count': len(success_traces),
        }]
        edges: List[dict] = []
        prefix_to_node: Dict[Tuple[str, ...], str] = {(): 'n0'}
        goal_nodes: Set[str] = set()
        node_map = {'n0': nodes[0]}

        for trace in success_traces:
            prefix: List[str] = []
            prev_node = 'n0'
            for i, step in enumerate(trace, start=1):
                prefix.append(step['action'])
                key = tuple(prefix)
                if key not in prefix_to_node:
                    node_id = f'n{len(nodes)}'
                    prefix_to_node[key] = node_id
                    node = {
                        'id': node_id,
                        'depth': i,
                        'state': step['state'],
                        'is_goal': False,
                        'prefix_count': 0,
                    }
                    nodes.append(node)
                    node_map[node_id] = node
                    edges.append({
                        'from': prev_node,
                        'to': node_id,
                        'action': step['action'],
                        'source': step['source'],
                        'type': 'transition',
                    })
                prev_node = prefix_to_node[key]
                node_map[prev_node]['prefix_count'] = node_map[prev_node].get('prefix_count', 0) + 1
            if trace:
                goal_nodes.add(prev_node)

        for gid in goal_nodes:
            node_map[gid]['is_goal'] = True
        return nodes, edges, sorted(goal_nodes, key=lambda x: int(x[1:]))

    def search_task(self, task_id: str, override_max_path_nodes: Optional[int] = None) -> dict:
        task_id = self.resolve_task_id(task_id)
        raw_task = self.by_task[task_id]
        task_scope = self.task_scope(raw_task)
        task, dropped_predicates, static_unsat = self.preprocess_task_predicates(raw_task)
        initial = self.augment_initial_exists(self.state_from_strings(task['initial_state']['predicates']), task_scope)
        goal = self.state_from_strings(task['final_state']['predicates'])
        actions = self.build_actions_for_task(task, task_scope)
        max_depth = int(self.rules['search']['max_depth'])
        max_traces = int(self.rules['search'].get('max_success_traces', 10000))
        max_path_nodes = int(override_max_path_nodes if override_max_path_nodes is not None else self.rules['search'].get('max_path_nodes', 10))

        success_traces = [] if static_unsat else self.enumerate_success_traces(
            initial, goal, actions, max_depth, max_traces=max_traces, max_path_nodes=max_path_nodes
        )
        nodes, dag_edges, goal_nodes = self.build_success_prefix_graph(initial, success_traces)

        goal_paths: List[List[dict]] = []
        edge_lookup = defaultdict(dict)
        for e in dag_edges:
            edge_lookup[e['from']][e['action']] = e['to']
        for trace in success_traces:
            prev = 'n0'
            path = []
            for step in trace:
                to = edge_lookup[prev][step['action']]
                path.append({'from': prev, 'to': to, 'action': step['action'], 'source': step['source']})
                prev = to
            goal_paths.append(path)

        return {
            'task_id': task_id,
            'description': raw_task.get('description', ''),
            'output_language': self.rules.get('output_language', 'English'),
            'is_dag': True,
            'graph_mode': 'success_prefix_graph_no_state_merge',
            'task_scope': sorted(task_scope),
            'preprocess': {
                'dropped_predicates': dropped_predicates,
                'static_unsatisfied_constraints': static_unsat,
            },
            'initial_state': self.state_to_strings(initial),
            'goal_state': self.state_to_strings(goal),
            'nodes': nodes,
            'dag_edges': dag_edges,
            'equivalence_edges': [],
            'goal_nodes': goal_nodes,
            'goal_paths': goal_paths,
            'success_traces': [[step['action'] for step in t] for t in success_traces],
            'has_solution': bool(success_traces) and not static_unsat,
            'node_count': len(nodes),
            'transition_edge_count': len(dag_edges),
            'equivalence_edge_count': 0,
            'grounded_action_count': len(actions),
            'grounded_subtask_count': sum(1 for a in actions if a.source == 'subtask'),
            'grounded_place_primitive_count': sum(1 for a in actions if a.source == 'primitive'),
            'success_trace_count': len(success_traces),
            'max_success_traces': max_traces,
            'max_path_nodes': max_path_nodes,
        }

    def auto_search_task_by_path_nodes(self, task_id: str, min_path_nodes: int, max_path_nodes: int, step: int = 1) -> dict:
        if min_path_nodes < 1:
            raise ValueError('min_path_nodes must be >= 1')
        if max_path_nodes < min_path_nodes:
            raise ValueError('max_path_nodes must be >= min_path_nodes')
        if step < 1:
            raise ValueError('step must be >= 1')

        tried = []
        last_result = None
        for cur in range(min_path_nodes, max_path_nodes + 1, step):
            result = self.search_task(task_id, override_max_path_nodes=cur)
            tried.append({
                'max_path_nodes': cur,
                'has_solution': result['has_solution'],
                'success_trace_count': result.get('success_trace_count', 0),
                'node_count': result['node_count'],
            })
            last_result = result
            if result['has_solution']:
                result['auto_search'] = {
                    'enabled': True,
                    'min_path_nodes': min_path_nodes,
                    'max_path_nodes': max_path_nodes,
                    'step': step,
                    'selected_max_path_nodes': cur,
                    'trials': tried,
                }
                return result

        assert last_result is not None
        last_result['auto_search'] = {
            'enabled': True,
            'min_path_nodes': min_path_nodes,
            'max_path_nodes': max_path_nodes,
            'step': step,
            'selected_max_path_nodes': None,
            'trials': tried,
        }
        return last_result

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
                'success_trace_count': result.get('success_trace_count', 0),
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
    ap.add_argument('--output-dir', default='./dag_outputs_success_only_autopath')
    ap.add_argument('--max-success-traces', type=int, default=100)
    ap.add_argument('--max-path-nodes', type=int, default=15, help='Upper bound for max-path-nodes. Used directly unless --auto-max-path-nodes is set.')
    ap.add_argument('--min-path-nodes', type=int, default=2, help='Lower bound when auto searching max-path-nodes.')
    ap.add_argument('--path-nodes-step', type=int, default=1, help='Step size when auto searching max-path-nodes.')
    ap.add_argument('--auto-max-path-nodes', action='store_true', default=True, help='Try max-path-nodes from small to large and stop at the first successful result.')
    args = ap.parse_args()

    planner = Planner(Path(args.base_dir), Path(args.out_dir), Path(args.rules), Path(args.subtasks))
    planner.rules.setdefault('search', {})['max_success_traces'] = args.max_success_traces
    planner.rules.setdefault('search', {})['max_path_nodes'] = args.max_path_nodes
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.task_id:
        if args.auto_max_path_nodes:
            result = planner.auto_search_task_by_path_nodes(
                args.task_id,
                min_path_nodes=args.min_path_nodes,
                max_path_nodes=args.max_path_nodes,
                step=args.path_nodes_step,
            )
        else:
            result = planner.search_task(args.task_id)
        out = outdir / f"{result['task_id']}_dag.json"
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({
            'task_id': result['task_id'],
            'has_solution': result['has_solution'],
            'node_count': result['node_count'],
            'transition_edge_count': result['transition_edge_count'],
            'success_trace_count': result.get('success_trace_count', 0),
            'goal_path_count': len(result['goal_paths']),
            'max_path_nodes': result.get('max_path_nodes', args.max_path_nodes),
            'selected_max_path_nodes': result.get('auto_search', {}).get('selected_max_path_nodes'),
            'output': str(out),
        }, indent=2))
    else:
        summary = planner.search_all(outdir)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
