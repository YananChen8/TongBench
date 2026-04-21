import argparse
import json
import re
from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set

try:
    import yaml
except Exception as e:
    raise RuntimeError('PyYAML is required') from e

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


class Planner:
    def __init__(self, base_dir: Path, out_dir: Path, rules_path: Path):
        self.out_dir = out_dir
        self.base_dir = base_dir
        self.rules = yaml.safe_load(rules_path.read_text())
        self.tasks = json.loads((out_dir / 'tasks.json').read_text())
        self.templates = json.loads((out_dir / 'atomic_templates.json').read_text())
        self.scene = json.loads((base_dir / 'touchdata.json').read_text())
        self.affordances = json.loads((base_dir / 'touchdata_objects.json').read_text())['objects']
        self.obj_alias = self.rules['normalization'].get('object_id_aliases', {})
        self.label_alias = self.rules['normalization'].get('label_aliases', {})
        self.suppressed_templates = set(self.rules.get('suppressed_templates', []))
        self.transient_predicates = set(self.rules['state'].get('transient_predicates', []))
        self.container_labels = {self.norm_label(x) for x in self.rules.get('container_labels', [])}
        self.surface_labels = {self.norm_label(x) for x in self.rules.get('surface_labels', [])}
        self.hanger_labels = {self.norm_label(x) for x in self.rules.get('hanger_labels', [])}
        self.pluggable_labels = {self.norm_label(x) for x in self.rules.get('pluggable_labels', [])}
        self.object_labels = {
            obj_id: self.norm_label((meta or {}).get('label', ''))
            for obj_id, meta in self.scene.items()
        }
        self.action_rules = self.rules.get('action_rules', {})

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
        return self.normalize_token(self.label_alias.get(raw, raw))

    def resolve_object_id(self, obj_id: str) -> str:
        if obj_id in self.scene or obj_id in self.affordances:
            return obj_id
        if obj_id in self.obj_alias:
            return self.obj_alias[obj_id]
        return obj_id

    def parse_pred(self, s: str) -> Pred:
        s = s.strip()
        m = PRED_RE.match(s)
        if not m:
            return Pred(name=s, args=())
        name = m.group(1).strip()
        arg_text = m.group(2).strip()
        if not arg_text:
            return Pred(name=name, args=())
        args = tuple(self.resolve_object_id(a.strip()) for a in arg_text.split(','))
        return Pred(name=name, args=args)

    def state_from_strings(self, preds: Iterable[str]) -> frozenset:
        return frozenset(self.parse_pred(p) for p in preds)

    def state_to_strings(self, state: Iterable[Pred]) -> List[str]:
        return sorted(p.to_string() for p in state)

    def task_by_id(self, task_id: str) -> dict:
        for t in self.tasks:
            if t['task_id'] == task_id:
                return t
        raise KeyError(task_id)

    def task_scope(self, task: dict) -> Set[str]:
        objs = set()
        for part in ['initial_state', 'final_state']:
            for p in task[part]['predicates']:
                pred = self.parse_pred(p)
                for a in pred.args:
                    if a in self.scene or a in self.affordances:
                        objs.add(a)
        return objs

    def is_pickable(self, obj_id: str) -> bool:
        return bool(self.affordances.get(obj_id, {}).get('pickable', False))

    def is_openable(self, obj_id: str) -> bool:
        return bool(self.affordances.get(obj_id, {}).get('openable', False))

    def is_powerable(self, obj_id: str) -> bool:
        return bool(self.affordances.get(obj_id, {}).get('powerable', False))

    def obj_label(self, obj_id: str) -> str:
        return self.object_labels.get(obj_id, '')

    def object_matches_category(self, obj_id: str, category: str) -> bool:
        c = self.norm_label(category)
        label = self.obj_label(obj_id)
        if label == c:
            return True
        if c == 'trashcan' and label in {'wastebasket', 'trashcan'}:
            return True
        if c == 'storagebox' and label in {'storagebox', 'shoescabinet'}:
            return True
        if c == 'tablelamp' and label in {'tablelamp', 'lamp', 'decorationlamp'}:
            return True
        if c == 'fan' and label in {'electricfan', 'fan'}:
            return True
        if c == 'computer' and label == 'computer':
            return True
        if c == 'laptop' and label in {'computer', 'laptop'}:
            return True
        if c == 'coffeetable' and label == 'coffeetable':
            return True
        return False

    def instantiate_atom(self, pattern: str, binding: Dict[str, str]) -> Pred:
        s = pattern
        for k, v in binding.items():
            s = s.replace('{' + k + '}', v)
        return self.parse_pred(s)

    def find_task_hang_targets(self, task_scope: Set[str], task: dict) -> List[str]:
        targets = []
        for p in task['final_state']['predicates']:
            pred = self.parse_pred(p)
            if pred.name == 'on' and len(pred.args) == 2:
                obj1, obj2 = pred.args
                if obj2 in task_scope and self.obj_label(obj2) in self.hanger_labels:
                    targets.append(obj2)
        if targets:
            return sorted(set(targets))
        fallback = [o for o in task_scope if self.obj_label(o) in self.hanger_labels]
        return sorted(set(fallback))

    def expand_template(self, template: dict, task: dict, task_scope: Set[str]) -> List[GroundAction]:
        template_id = template['template_id']
        if template_id in self.suppressed_templates:
            return []
        grounded = []
        available = template.get('available_objects', [])
        if template_id == 'hang_{object}':
            obj_candidates = [o for o in task_scope if any(self.object_matches_category(o, c) for c in available)]
            targets = self.find_task_hang_targets(task_scope, task)
            for obj in obj_candidates:
                for target in targets:
                    binding = {'object': obj, 'target': target}
                    pre = [self.instantiate_atom(x, binding) for x in template['pre_state']]
                    add = [Pred('on', (obj, target)), Pred('empty', ())]
                    delete = [Pred('holding', (obj,))]
                    grounded.append(GroundAction(
                        name=f'hang_{obj}_on_{target}',
                        template_id=template_id,
                        pre=tuple(pre),
                        add=tuple(add),
                        delete=tuple(delete),
                        meta=tuple(sorted(binding.items())),
                    ))
            return grounded

        if isinstance(available, list):
            if '{object}' in template_id:
                for obj in sorted(task_scope):
                    if template_id == 'pick_up_{object}':
                        if not self.is_pickable(obj):
                            continue
                    elif template_id in {'open_{object}', 'close_{object}'}:
                        if not self.is_openable(obj):
                            continue
                    elif template_id in {'switch_on_{object}', 'switch_off_{object}'}:
                        if not self.is_powerable(obj):
                            continue
                    elif template_id in {'look_at_{object}', 'walk_to_{object}', 'point_at_{object}'}:
                        pass
                    elif template_id in {'plug_in_{object}', 'unplug_{object}'}:
                        if self.obj_label(obj) not in self.pluggable_labels and obj not in self.affordances:
                            continue
                    elif not any(self.object_matches_category(obj, c) for c in available):
                        continue
                    binding = {'object': obj}
                    pre = [self.instantiate_atom(x, binding) for x in template['pre_state']]
                    add = [self.instantiate_atom(x, binding) for x in template['post_state']]
                    delete = self.make_delete_list(template_id, binding)
                    grounded.append(GroundAction(
                        name=template_id.replace('{object}', obj),
                        template_id=template_id,
                        pre=tuple(pre),
                        add=tuple(add),
                        delete=tuple(delete),
                        meta=tuple(sorted(binding.items())),
                    ))
            else:
                pre = tuple(self.parse_pred(x) for x in template['pre_state'])
                add = tuple(self.parse_pred(x) for x in template['post_state'])
                delete = tuple(self.make_delete_list(template_id, {}))
                grounded.append(GroundAction(
                    name=template_id,
                    template_id=template_id,
                    pre=pre,
                    add=add,
                    delete=delete,
                    meta=tuple(),
                ))
            return grounded

        if isinstance(available, dict):
            keys = list(available.keys())
            assert len(keys) == 2
            k1, k2 = keys
            allowed1 = list(available[k1])
            allowed2 = list(available[k2])
            if template_id == 'place_{object1}_on_{object2}':
                allowed2 = sorted(set(list(allowed2) + list(self.rules.get('surface_labels', []))))
            if template_id == 'place_{object1}_in_{object2}':
                allowed2 = sorted(set(list(allowed2) + list(self.rules.get('container_labels', []))))
            if template_id in {'place_{object1}_on_{object2}', 'place_{object1}_in_{object2}'}:
                cands1 = [o for o in task_scope if self.is_pickable(o)]
            else:
                cands1 = [o for o in task_scope if any(self.object_matches_category(o, c) for c in allowed1)]
            cands2 = [o for o in task_scope if any(self.object_matches_category(o, c) for c in allowed2)]
            for obj1 in sorted(cands1):
                for obj2 in sorted(cands2):
                    if obj1 == obj2:
                        continue
                    binding = {k1: obj1, k2: obj2}
                    pre = [self.instantiate_atom(x, binding) for x in template['pre_state']]
                    add = [self.instantiate_atom(x, binding) for x in template['post_state']]
                    delete = self.make_delete_list(template_id, binding)
                    grounded.append(GroundAction(
                        name=template_id.replace('{' + k1 + '}', obj1).replace('{' + k2 + '}', obj2),
                        template_id=template_id,
                        pre=tuple(pre),
                        add=tuple(add),
                        delete=tuple(delete),
                        meta=tuple(sorted(binding.items())),
                    ))
            return grounded
        return []

    def make_delete_list(self, template_id: str, binding: Dict[str, str]) -> List[Pred]:
        cfg = self.action_rules.get(template_id, {})
        deletes = [self.instantiate_special_delete(x, binding) for x in cfg.get('deletes', [])]
        implied = [self.instantiate_atom(x, binding) for x in cfg.get('implies', [])]
        extra_adds = [self.instantiate_atom(x, binding) for x in cfg.get('adds', [])]
        # attach extra adds later through apply
        return deletes + implied + extra_adds  # implied/adds are filtered in split_extra_effects

    def instantiate_special_delete(self, pattern: str, binding: Dict[str, str]) -> Pred:
        s = pattern
        for k, v in binding.items():
            s = s.replace('{' + k + '}', v)
        m = PRED_RE.match(s)
        if not m:
            return Pred(name=s, args=())
        name = m.group(1).strip()
        arg_text = m.group(2).strip()
        args = tuple(a.strip() for a in arg_text.split(',')) if arg_text else tuple()
        args = tuple(self.resolve_object_id(a) if a != '*' else '*' for a in args)
        return Pred(name=name, args=args)

    def split_delete_implied_extra(self, action: GroundAction) -> Tuple[List[Pred], List[Pred], List[Pred]]:
        cfg = self.action_rules.get(action.template_id, {})
        delete = [self.instantiate_special_delete(x, dict(action.meta)) for x in cfg.get('deletes', [])]
        implied = [self.instantiate_atom(x, dict(action.meta)) for x in cfg.get('implies', [])]
        extra_adds = [self.instantiate_atom(x, dict(action.meta)) for x in cfg.get('adds', [])]
        return delete, implied, extra_adds

    def valid_action(self, state: Set[Pred], action: GroundAction) -> bool:
        if any(p not in state for p in action.pre):
            return False
        if self.rules['semantics'].get('require_exists_for_all_actions', True):
            for obj in self.bound_objects(action):
                if Pred('exists', (obj,)) not in state:
                    return False
        tid = action.template_id
        b = dict(action.meta)
        if tid == 'pick_up_{object}':
            obj = b['object']
            if not self.is_pickable(obj):
                return False
            if any(p.name == 'holding' and p.args != (obj,) for p in state):
                return False
            for p in state:
                if p.name == 'in' and p.args[0] == obj:
                    cont = p.args[1]
                    if self.is_openable(cont) and Pred('open', (cont,)) not in state:
                        return False
        elif tid == 'open_{object}' or tid == 'close_{object}':
            obj = b['object']
            if not self.is_openable(obj):
                return False
        elif tid in {'switch_on_{object}', 'switch_off_{object}'}:
            obj = b['object']
            if not self.is_powerable(obj):
                return False
        elif tid in {'plug_in_{object}', 'unplug_{object}'}:
            obj = b['object']
            if self.obj_label(obj) not in self.pluggable_labels and obj not in self.affordances:
                return False
        elif tid == 'place_{object1}_on_{object2}':
            obj2 = b['object2']
            if self.obj_label(obj2) not in self.surface_labels and obj2 not in self.bound_objects(action):
                return False
        elif tid == 'place_{object1}_in_{object2}':
            obj2 = b['object2']
            if self.obj_label(obj2) not in self.container_labels:
                return False
            if self.rules['semantics'].get('require_open_container_for_place_in', True) and self.is_openable(obj2):
                if Pred('open', (obj2,)) not in state:
                    return False
        elif tid == 'wash_{object}':
            pass
        elif tid == 'hang_{object}':
            obj = b['object']
            if Pred('holding', (obj,)) not in state:
                return False
        return True

    def bound_objects(self, action: GroundAction) -> Set[str]:
        return {v for _, v in action.meta}

    def wildcard_delete(self, state: Set[Pred], pat: Pred) -> Set[Pred]:
        out = set()
        for p in state:
            if p.name != pat.name:
                out.add(p)
                continue
            if len(p.args) != len(pat.args):
                out.add(p)
                continue
            ok = True
            for a, b in zip(p.args, pat.args):
                if b != '*' and a != b:
                    ok = False
                    break
            if not ok:
                out.add(p)
        return out

    def apply_action(self, state: frozenset, action: GroundAction) -> frozenset:
        s = set(state)
        delete, implied, extra_adds = self.split_delete_implied_extra(action)
        for d in delete:
            s = self.wildcard_delete(s, d)
        for p in implied:
            s.add(p)
        for p in action.add:
            s = self.remove_mutex_conflicts(s, p)
            s.add(p)
        for p in extra_adds:
            s = self.remove_mutex_conflicts(s, p)
            s.add(p)
        # holding/empty exclusivity
        if any(p.name == 'holding' for p in s):
            s.discard(Pred('empty', ()))
        elif self.rules['state'].get('singleton_empty', True):
            s.add(Pred('empty', ()))
        return frozenset(s)

    def remove_mutex_conflicts(self, state: Set[Pred], new_pred: Pred) -> Set[Pred]:
        groups = self.rules['state'].get('mutex_groups', [])
        for g in groups:
            if new_pred.name in g:
                for other in g:
                    if other == new_pred.name:
                        continue
                    if new_pred.args:
                        state.discard(Pred(other, new_pred.args))
                    else:
                        state.discard(Pred(other, ()))
        if new_pred.name == 'holding':
            for p in list(state):
                if p.name == 'holding' and p != new_pred:
                    state.discard(p)
            state.discard(Pred('empty', ()))
        if new_pred.name == 'empty':
            for p in list(state):
                if p.name == 'holding':
                    state.discard(p)
        if new_pred.name in {'holding', 'on', 'in'}:
            obj = new_pred.args[0]
            for p in list(state):
                if p.name in {'holding', 'on', 'in'} and p.args and p.args[0] == obj and p != new_pred:
                    state.discard(p)
        if new_pred.name in {'on', 'in'}:
            obj = new_pred.args[0]
            for p in list(state):
                if p.name in {'on', 'in'} and p.args and p.args[0] == obj and p != new_pred:
                    state.discard(p)
        return state

    def canonical_state_key(self, state: frozenset) -> Tuple[str, ...]:
        return tuple(sorted(p.to_string() for p in state))

    def goal_satisfied(self, state: frozenset, goal: frozenset) -> bool:
        return goal.issubset(state)

    def build_actions_for_task(self, task: dict, task_scope: Set[str]) -> List[GroundAction]:
        actions = []
        for t in self.templates['templates']:
            actions.extend(self.expand_template(t, task, task_scope))
        if self.rules['search'].get('include_only_goal_relevant_templates', False):
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
        filtered = []
        for a in actions:
            adds = set(a.add)
            _, implied, extra_adds = self.split_delete_implied_extra(a)
            adds.update(implied)
            adds.update(extra_adds)
            if required & adds:
                filtered.append(a)
        return filtered


    def augment_initial_exists(self, initial: frozenset, task_scope: Set[str]) -> frozenset:
        s = set(initial)
        for obj in task_scope:
            if obj in self.scene:
                s.add(Pred('exists', (obj,)))
        return frozenset(s)

    def search_task(self, task_id: str) -> dict:
        task = self.task_by_id(task_id)
        task_scope = self.task_scope(task)
        initial = self.state_from_strings(task['initial_state']['predicates'])
        goal = self.state_from_strings(task['final_state']['predicates'])
        initial = self.augment_initial_exists(initial, task_scope)
        actions = self.build_actions_for_task(task, task_scope)
        max_depth = int(self.rules['search']['max_depth'])
        max_nodes = int(self.rules['search']['max_nodes'])

        nodes = []
        edges = []
        equivalence_edges = []
        key_to_node = {}
        parent_paths = defaultdict(list)
        goal_nodes = []

        init_key = self.canonical_state_key(initial)
        key_to_node[init_key] = 'n0'
        nodes.append({
            'id': 'n0',
            'depth': 0,
            'state': self.state_to_strings(initial),
            'is_goal': self.goal_satisfied(initial, goal),
        })
        q = deque([('n0', initial, 0)])

        while q and len(nodes) < max_nodes:
            node_id, state, depth = q.popleft()
            is_goal = self.goal_satisfied(state, goal)
            if is_goal:
                goal_nodes.append(node_id)
                if self.rules['search'].get('stop_expanding_at_goal', True):
                    continue
            if depth >= max_depth:
                continue
            for action in actions:
                if not self.valid_action(set(state), action):
                    continue
                nxt = self.apply_action(state, action)
                nxt_key = self.canonical_state_key(nxt)
                if nxt_key == self.canonical_state_key(state):
                    continue
                if nxt_key in key_to_node:
                    equivalence_edges.append({
                        'from': node_id,
                        'to': key_to_node[nxt_key],
                        'action': action.name,
                        'type': 'equivalence',
                    })
                    continue
                new_id = f"n{len(nodes)}"
                key_to_node[nxt_key] = new_id
                parent_paths[new_id].append(node_id)
                goal_flag = self.goal_satisfied(nxt, goal)
                nodes.append({
                    'id': new_id,
                    'depth': depth + 1,
                    'state': self.state_to_strings(nxt),
                    'is_goal': goal_flag,
                })
                edges.append({
                    'from': node_id,
                    'to': new_id,
                    'action': action.name,
                    'type': 'transition',
                })
                q.append((new_id, nxt, depth + 1))

        if nodes and nodes[0]['id'] in goal_nodes:
            goal_nodes = list(dict.fromkeys(goal_nodes))
        else:
            goal_nodes = [n['id'] for n in nodes if n['is_goal']]

        topo = [n['id'] for n in sorted(nodes, key=lambda x: (x['depth'], int(x['id'][1:])))]
        paths = []
        if self.rules['search'].get('export_all_goal_paths', True):
            adjacency = defaultdict(list)
            for e in edges:
                adjacency[e['from']].append((e['to'], e['action']))
            goal_set = set(goal_nodes)
            def dfs(cur: str, cur_path: List[dict]):
                if cur in goal_set:
                    paths.append(cur_path.copy())
                    return
                for nxt, action_name in adjacency.get(cur, []):
                    cur_path.append({'from': cur, 'to': nxt, 'action': action_name})
                    dfs(nxt, cur_path)
                    cur_path.pop()
            dfs('n0', [])

        return {
            'task_id': task_id,
            'description': task['description'],
            'intention': task['intention'],
            'initial_state': self.state_to_strings(initial),
            'goal_state': self.state_to_strings(goal),
            'task_scope_objects': sorted(task_scope),
            'grounded_action_count': len(actions),
            'node_count': len(nodes),
            'transition_edge_count': len(edges),
            'equivalence_edge_count': len(equivalence_edges),
            'goal_nodes': goal_nodes,
            'has_solution': bool(goal_nodes),
            'nodes': nodes,
            'edges': edges,
            'equivalence_edges': equivalence_edges,
            'topological_order': topo,
            'goal_paths': paths,
        }

    def search_all(self, output_dir: Path) -> Dict[str, dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        summary = []
        for task in self.tasks:
            result = self.search_task(task['task_id'])
            results[task['task_id']] = result
            (output_dir / f"{task['task_id']}_dag.json").write_text(json.dumps(result, indent=2))
            summary.append({
                'task_id': task['task_id'],
                'has_solution': result['has_solution'],
                'goal_nodes': result['goal_nodes'],
                'node_count': result['node_count'],
                'transition_edge_count': result['transition_edge_count'],
                'equivalence_edge_count': result['equivalence_edge_count'],
                'grounded_action_count': result['grounded_action_count'],
                'goal_path_count': len(result['goal_paths']),
            })
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='./env_data')
    ap.add_argument('--out-dir', default='./outputs')
    ap.add_argument('--rules', default='./prompts/planner_rules.yaml')
    ap.add_argument('--task-id', default=None)
    ap.add_argument('--output-dir', default='./dag_outputs')
    args = ap.parse_args()

    planner = Planner(Path(args.base_dir), Path(args.out_dir), Path(args.rules))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.task_id:
        result = planner.search_task(args.task_id)
        out = outdir / f'{args.task_id}_dag.json'
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({
            'task_id': args.task_id,
            'has_solution': result['has_solution'],
            'goal_nodes': result['goal_nodes'],
            'node_count': result['node_count'],
            'transition_edge_count': result['transition_edge_count'],
            'equivalence_edge_count': result['equivalence_edge_count'],
            'grounded_action_count': result['grounded_action_count'],
            'goal_path_count': len(result['goal_paths']),
            'output': str(out),
        }, indent=2))
    else:
        planner.search_all(outdir)
        print(str(outdir / 'summary.json'))


if __name__ == '__main__':
    main()
