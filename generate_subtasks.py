import argparse
import copy
import itertools
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

_VAR_RE = re.compile(r"\{([^{}]+)\}")
_PRED_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$")


@dataclass(frozen=True)
class Predicate:
    name: str
    args: Tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(self.args)})"


@dataclass
class ActionTemplate:
    template_id: str
    action_type: str
    pre_state: List[str]
    post_state: List[str]
    available_objects: dict
    task_description: str
    var_names: List[str]


@dataclass
class ActionInstance:
    template_id: str
    action_type: str
    var_mapping: Dict[str, str]
    preconditions: List[Predicate]
    effects_add: List[Predicate]
    effects_delete: List[str]
    description: str

    def pretty_name(self) -> str:
        parts = []
        for var in sorted(self.var_mapping):
            parts.append(self.var_mapping[var])
        if parts:
            return f"{self.template_id}::{','.join(parts)}"
        return self.template_id


class SubtaskGenerator:
    def __init__(self, atomic_path: Path, rules_path: Path):
        with open(atomic_path, 'r', encoding='utf-8') as f:
            atomic = json.load(f)
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)

        self.templates: List[ActionTemplate] = []
        self.templates_by_id: Dict[str, ActionTemplate] = {}
        self.suppressed_templates = set(self.rules.get('suppressed_templates', []))

        self.agent_scoped_predicates = set(self.rules.get('state', {}).get('agent_scoped_predicates', []))
        self.mutex_groups = [set(g) for g in self.rules.get('state', {}).get('mutex_groups', [])]
        self.binary_relation_mutex = [set(g) for g in self.rules.get('state', {}).get('binary_relation_mutex', [])]
        self.container_labels = set(self.rules.get('container_labels', []))
        self.surface_labels = set(self.rules.get('surface_labels', []))
        self.hanger_labels = set(self.rules.get('hanger_labels', []))
        self.semantics = self.rules.get('semantics', {})
        self.action_rules = self.rules.get('action_rules', {})
        self.label_aliases = self.rules.get('normalization', {}).get('label_aliases', {})
        self.disallow_consecutive_same_action_type = self.rules.get('search', {}).get('disallow_consecutive_same_action_type', True)

        for raw in atomic['templates']:
            if raw['template_id'] in self.suppressed_templates:
                continue
            action_type = self._normalize_action_type(raw['action_space'][0] if raw.get('action_space') else raw['template_id'])
            available = self._normalize_available_objects(raw['available_objects'])
            vars_in_id = _VAR_RE.findall(raw['template_id'])
            if not vars_in_id and isinstance(available, dict):
                vars_in_id = list(available.keys())
            elif not vars_in_id and not isinstance(available, dict):
                vars_in_id = []
            tpl = ActionTemplate(
                template_id=raw['template_id'],
                action_type=action_type,
                pre_state=raw.get('pre_state', []),
                post_state=raw.get('post_state', []),
                available_objects=available,
                task_description=raw.get('task_description', ''),
                var_names=vars_in_id,
            )
            self.templates.append(tpl)
            self.templates_by_id[tpl.template_id] = tpl

        self.role_counter = 0

    def _normalize_available_objects(self, obj):
        if isinstance(obj, list):
            return {'object': {self._normalize_label(x) for x in obj}}
        return {k: {self._normalize_label(x) for x in v} for k, v in obj.items()}

    def _normalize_label(self, label: str) -> str:
        s = label.lower().replace('_', '').replace('-', '').replace(' ', '')
        return self.label_aliases.get(s, s)

    def _normalize_action_type(self, action_type: str) -> str:
        mapping = {
            'look_at': 'look',
            'walk_to': 'walk',
            'pick_up': 'pick',
            'wash_object': 'wash',
            'switch_on': 'switch_on',
            'switch_off': 'switch_off',
            'plug_in': 'plug_in',
            'point_at': 'point',
            'hang_object': 'hang',
        }
        return mapping.get(action_type, action_type)

    def _parse_predicate(self, text: str) -> Predicate:
        text = text.strip()
        m = _PRED_RE.match(text)
        if not m:
            return Predicate(text, ())
        name = m.group(1)
        args_blob = m.group(2).strip()
        args = tuple(a.strip() for a in args_blob.split(',')) if args_blob else tuple()
        return Predicate(name, args)

    def _substitute(self, text: str, mapping: Dict[str, str]) -> str:
        out = text
        for k, v in mapping.items():
            out = out.replace('{' + k + '}', v)
        return out

    def _predicate_from_text(self, text: str, mapping: Dict[str, str]) -> Predicate:
        return self._parse_predicate(self._substitute(text, mapping))

    def _delete_matches(self, state: Set[Predicate], delete_pattern: str) -> Set[Predicate]:
        pat = self._predicate_from_text(delete_pattern, {}) if '{' not in delete_pattern else None
        if '*' not in delete_pattern:
            p = self._parse_predicate(delete_pattern)
            return {x for x in state if x != p}
        m = _PRED_RE.match(delete_pattern)
        if not m:
            return state
        name = m.group(1)
        args = [a.strip() for a in m.group(2).split(',') if a.strip()]
        kept = set()
        for pred in state:
            if pred.name != name or len(pred.args) != len(args):
                kept.add(pred)
                continue
            matched = True
            for a, b in zip(args, pred.args):
                if a == '*':
                    continue
                if a != b:
                    matched = False
                    break
            if not matched:
                kept.add(pred)
        return kept

    def _remove_mutex_conflicts(self, state: Set[Predicate], new_pred: Predicate) -> Set[Predicate]:
        new_state = set(state)
        if new_pred.name in self.agent_scoped_predicates:
            new_state = {p for p in new_state if p.name != new_pred.name}

        for group in self.mutex_groups:
            if new_pred.name in group:
                for other in group:
                    if other == new_pred.name:
                        continue
                    new_state = {p for p in new_state if not (p.name == other and p.args == new_pred.args)}

        for group in self.binary_relation_mutex:
            if new_pred.name in group and len(new_pred.args) == 2:
                a1 = new_pred.args[0]
                new_state = {p for p in new_state if not (p.name in group and len(p.args) == 2 and p.args[0] == a1 and p.name != new_pred.name)}
        return new_state

    def _semantic_extra_preconditions(self, action: ActionTemplate, mapping: Dict[str, str]) -> List[Predicate]:
        extra: List[Predicate] = []
        if self.semantics.get('require_exists_for_all_actions', True):
            for var in action.var_names:
                extra.append(Predicate('exists', (mapping[var],)))
        if action.template_id == 'place_{object1}_in_{object2}' and self.semantics.get('require_open_container_for_place_in', False):
            extra.append(Predicate('open', (mapping['object2'],)))
        if action.template_id == 'hang_{object}':
            extra.append(Predicate('exists', (mapping['object'],)))
        return extra

    def instantiate_action(self, template: ActionTemplate, mapping: Dict[str, str]) -> ActionInstance:
        preconditions = [self._predicate_from_text(x, mapping) for x in template.pre_state]
        preconditions.extend(self._semantic_extra_preconditions(template, mapping))

        effects_add = [self._predicate_from_text(x, mapping) for x in template.post_state]
        rule = self.action_rules.get(template.template_id, {})
        for x in rule.get('adds', []):
            effects_add.append(self._predicate_from_text(x, mapping))
        for x in rule.get('implies', []):
            effects_add.append(self._predicate_from_text(x, mapping))
        effects_add = self._dedup_preds(effects_add)

        effects_delete = [self._substitute(x, mapping) for x in rule.get('deletes', [])]

        return ActionInstance(
            template_id=template.template_id,
            action_type=template.action_type,
            var_mapping=dict(mapping),
            preconditions=preconditions,
            effects_add=effects_add,
            effects_delete=effects_delete,
            description=template.task_description,
        )

    def _dedup_preds(self, preds: List[Predicate]) -> List[Predicate]:
        seen = set()
        out = []
        for p in preds:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    def apply_action(self, state: Set[Predicate], action: ActionInstance) -> Set[Predicate]:
        s = set(state)
        for delete in action.effects_delete:
            s = self._delete_matches(s, delete)
        for pred in action.effects_add:
            s = self._remove_mutex_conflicts(s, pred)
            s.add(pred)
        return s

    def _conflicts(self, preds: Set[Predicate]) -> bool:
        by_name_args = defaultdict(set)
        for p in preds:
            by_name_args[(p.name, p.args)] = True
        # mutex groups
        for group in self.mutex_groups:
            grouped = defaultdict(set)
            for p in preds:
                if p.name in group:
                    grouped[p.args].add(p.name)
            for names in grouped.values():
                if len(names) > 1:
                    return True
        # agent-scoped singleton
        for name in self.agent_scoped_predicates:
            objs = [p for p in preds if p.name == name]
            if len(objs) > 1:
                return True
        # holding singleton via agent scoped already
        return False

    def _role_label_choices(self, template: ActionTemplate, var: str) -> Set[str]:
        return set(template.available_objects[var])

    def _new_role(self) -> str:
        self.role_counter += 1
        return f"r{self.role_counter}"

    def _candidate_mappings(self, template: ActionTemplate, existing_roles: Dict[str, Set[str]], max_new_roles: int = 2) -> List[Dict[str, str]]:
        vars_ = template.var_names
        if not vars_:
            return [{}]
        candidates: List[Dict[str, str]] = []
        choices_per_var = []
        fresh_roles = [f"NEW{i}" for i in range(max_new_roles)]
        for var in vars_:
            allowed = self._role_label_choices(template, var)
            opts = []
            for role, types in existing_roles.items():
                if types & allowed:
                    opts.append((role, types & allowed, False))
            opts.extend((fr, allowed, True) for fr in fresh_roles)
            choices_per_var.append(opts)

        for combo in itertools.product(*choices_per_var):
            mapping: Dict[str, str] = {}
            role_types: Dict[str, Set[str]] = {k: set(v) for k, v in existing_roles.items()}
            fresh_bindings: Dict[str, str] = {}
            valid = True
            for var, (role_key, allowed_types, is_new) in zip(vars_, combo):
                if is_new:
                    if role_key not in fresh_bindings:
                        fresh_bindings[role_key] = self._new_role()
                        role_types[fresh_bindings[role_key]] = set(allowed_types)
                    role = fresh_bindings[role_key]
                else:
                    role = role_key
                if var in mapping and mapping[var] != role:
                    valid = False
                    break
                mapping[var] = role
                role_types[role] = role_types[role] & set(allowed_types)
                if not role_types[role]:
                    valid = False
                    break
            if not valid:
                continue
            # object1/object2 cannot be same when binary placement / containment
            if 'object1' in mapping and 'object2' in mapping and mapping['object1'] == mapping['object2']:
                continue
            candidates.append(mapping)
        # dedup by mapping signature
        uniq = []
        seen = set()
        for m in candidates:
            key = tuple(sorted(m.items()))
            if key not in seen:
                uniq.append(m)
                seen.add(key)
        return uniq

    def _recompute_rollout(self, seq_actions: List[ActionInstance], seq_pre: Set[Predicate]) -> Optional[Tuple[List[Set[Predicate]], List[Set[Predicate]]]]:
        states_before = []
        states_after = []
        state = set(seq_pre)
        if self._conflicts(state):
            return None
        for act in seq_actions:
            states_before.append(set(state))
            if not all(p in state for p in act.preconditions):
                return None
            new_state = self.apply_action(state, act)
            if new_state == state:
                return None
            if self._conflicts(new_state):
                return None
            states_after.append(set(new_state))
            state = new_state
        return states_before, states_after

    def _dominant_effects(self, seq_actions: List[ActionInstance], seq_pre: Set[Predicate], end_state: Set[Predicate]) -> Tuple[Set[Predicate], Set[Predicate]]:
        added = {p for p in end_state if p not in seq_pre}
        deleted = {p for p in seq_pre if p not in end_state}
        return added, deleted

    def _sequence_type(self, last_action: ActionInstance, added: Set[Predicate]) -> str:
        names = {p.name for p in added}
        if 'in' in names:
            return 'store_in'
        if 'on' in names:
            return 'place_on'
        if 'holding' in names and 'clean' in names:
            return 'acquire_and_clean'
        if 'holding' in names:
            return 'acquire'
        if 'clean' in names:
            return 'clean'
        if 'open' in names:
            return 'open_container'
        if 'closed' in names:
            return 'close_container'
        if 'powered_on' in names:
            return 'power_on'
        if 'powered_off' in names:
            return 'power_off'
        if 'plugged_in' in names:
            return 'plug_in'
        if 'unplugged' in names:
            return 'unplug'
        if 'with_reach' in names:
            return 'reach'
        if 'in_view' in names:
            return 'focus'
        if 'pointed_at' in names:
            return 'point'
        return f'composite_{last_action.action_type}'



    def _rename_in_text(self, text: str, role_map: Dict[str, str]) -> str:
        out = text
        for old, new in sorted(role_map.items(), key=lambda kv: (-len(kv[0]), kv[0])):
            out = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])', new, out)
        return out

    def _canonicalize_subtask_dict(self, st: dict) -> dict:
        role_order: List[str] = []
        for step in st['primitive_sequence']:
            for _, role in sorted(step['var_mapping'].items()):
                if role not in role_order:
                    role_order.append(role)
        if not role_order:
            for role in sorted(st.get('roles', {})):
                if role not in role_order:
                    role_order.append(role)
        role_map = {old: f'r{i+1}' for i, old in enumerate(role_order)}

        out = copy.deepcopy(st)
        out['primitive_sequence'] = [
            {
                'template_id': step['template_id'],
                'action_type': step['action_type'],
                'var_mapping': {k: role_map.get(v, v) for k, v in sorted(step['var_mapping'].items())},
            }
            for step in st['primitive_sequence']
        ]
        out['roles'] = {
            role_map.get(role, role): sorted(list(types))
            for role, types in sorted(st.get('roles', {}).items(), key=lambda kv: role_map.get(kv[0], kv[0]))
        }
        out['pre_state'] = sorted(self._rename_in_text(x, role_map) for x in st['pre_state'])
        out['post_state'] = sorted(self._rename_in_text(x, role_map) for x in st['post_state'])
        out['net_add'] = sorted(self._rename_in_text(x, role_map) for x in st['net_add'])
        out['net_delete'] = sorted(self._rename_in_text(x, role_map) for x in st['net_delete'])
        return out

    def _canonical_subtask_signature(self, st: dict) -> Tuple:
        st = self._canonicalize_subtask_dict(st)
        return (
            st['subtask_type'],
            st['length'],
            tuple(
                (step['template_id'], step['action_type'], tuple(sorted(step['var_mapping'].items())))
                for step in st['primitive_sequence']
            ),
            tuple(sorted((role, tuple(types)) for role, types in st['roles'].items())),
            tuple(st['pre_state']),
            tuple(st['post_state']),
            tuple(st['net_add']),
            tuple(st['net_delete']),
        )

    def _is_valid_start_template(self, template: ActionTemplate) -> bool:
        return template.action_type == 'look'

    def _has_repeated_template(self, seq_actions: List[ActionInstance]) -> bool:
        seen = set()
        for act in seq_actions:
            if act.template_id in seen:
                return True
            seen.add(act.template_id)
        return False
    def _action_role(self, action: ActionInstance, preferred: List[str]) -> Optional[str]:
        for key in preferred:
            if key in action.var_mapping:
                return action.var_mapping[key]
        return None

    def _adjacent_compatible(self, prev: ActionInstance, nxt: ActionInstance) -> bool:
        prev_item = self._action_role(prev, ['object', 'object1'])
        prev_target = self._action_role(prev, ['object2', 'object'])
        next_item = self._action_role(nxt, ['object', 'object1'])
        next_target = self._action_role(nxt, ['object2', 'object'])

        if self.disallow_consecutive_same_action_type and prev.action_type == nxt.action_type:
            return False
        if prev.action_type == 'look' and nxt.action_type in {'walk', 'point'}:
            return prev_target == next_target
        if prev.action_type == 'point' and nxt.action_type == 'walk':
            return prev_target == next_target
        if prev.action_type == 'walk' and nxt.action_type in {'pick', 'wash', 'open', 'close', 'switch_on', 'switch_off', 'plug_in', 'unplug'}:
            return prev_target == next_target
        if prev.action_type == 'pick' and nxt.action_type in {'place', 'hang', 'wash'}:
            return prev_item == next_item
        if prev.action_type == 'open' and nxt.template_id == 'place_{object1}_in_{object2}':
            return prev_target == nxt.var_mapping.get('object2')
        if prev.action_type == 'wash' and nxt.action_type in {'look', 'walk'}:
            return True
        if prev.action_type == 'wash' and nxt.action_type == 'place':
            return prev_item == next_item
        if prev.action_type == 'place' and nxt.action_type in {'look', 'walk'}:
            return True
        if prev.action_type in {'switch_on', 'switch_off', 'plug_in', 'unplug', 'close', 'hang'} and nxt.action_type in {'look', 'walk'}:
            return True
        return False

    def _logical_filter(self, seq_actions: List[ActionInstance], seq_pre: Set[Predicate], states_before: List[Set[Predicate]], states_after: List[Set[Predicate]]) -> bool:
        for a, b in zip(seq_actions, seq_actions[1:]):
            if not self._adjacent_compatible(a, b):
                return False
            if a.template_id.startswith('open_') and b.template_id.startswith('close_') and a.var_mapping == b.var_mapping:
                return False
            if a.template_id.startswith('close_') and b.template_id.startswith('open_') and a.var_mapping == b.var_mapping:
                return False
            if a.template_id.startswith('switch_on_') and b.template_id.startswith('switch_off_') and a.var_mapping == b.var_mapping:
                return False
            if a.template_id.startswith('switch_off_') and b.template_id.startswith('switch_on_') and a.var_mapping == b.var_mapping:
                return False
        end_state = states_after[-1]
        added, _ = self._dominant_effects(seq_actions, seq_pre, end_state)
        meaningful = [p for p in added if p.name not in {'in_view', 'with_reach', 'pointed_at'}]
        if len(seq_actions) > 1 and not meaningful and seq_actions[-1].action_type not in {'look', 'walk'}:
            return False
        return True

    def enumerate_subtasks(self, max_len: int = 4) -> List[dict]:
        results: List[dict] = []
        seen_signatures = set()

        def dfs(seq_actions: List[ActionInstance], seq_pre: Set[Predicate], role_types: Dict[str, Set[str]], depth: int):
            if self._has_repeated_template(seq_actions):
                return
            rollout = self._recompute_rollout(seq_actions, seq_pre)
            if rollout is None:
                return
            states_before, states_after = rollout
            if not self._logical_filter(seq_actions, seq_pre, states_before, states_after):
                return
            end_state = states_after[-1]
            added, deleted = self._dominant_effects(seq_actions, seq_pre, end_state)
            st_type = self._sequence_type(seq_actions[-1], added)
            raw_item = {
                'subtask_id': '',
                'subtask_type': st_type,
                'length': len(seq_actions),
                'primitive_sequence': [
                    {
                        'template_id': a.template_id,
                        'action_type': a.action_type,
                        'var_mapping': a.var_mapping,
                    }
                    for a in seq_actions
                ],
                'roles': {r: sorted(list(t)) for r, t in sorted(role_types.items())},
                'pre_state': sorted(str(p) for p in seq_pre),
                'post_state': sorted(str(p) for p in end_state),
                'net_add': sorted(str(p) for p in added),
                'net_delete': sorted(str(p) for p in deleted),
            }
            canonical_item = self._canonicalize_subtask_dict(raw_item)
            signature = self._canonical_subtask_signature(canonical_item)
            if len(seq_actions) >= 2 and signature not in seen_signatures:
                seen_signatures.add(signature)
                canonical_item['subtask_id'] = f"subtask_{len(results)+1:04d}"
                results.append(canonical_item)

            if depth >= max_len:
                return

            current_state = states_after[-1]
            existing_roles = {r: set(t) for r, t in role_types.items()}
            used_template_ids = {a.template_id for a in seq_actions}
            for tpl in self.templates:
                if tpl.template_id in used_template_ids:
                    continue
                if self.disallow_consecutive_same_action_type and seq_actions[-1].action_type == tpl.action_type:
                    continue
                for mapping in self._candidate_mappings(tpl, existing_roles):
                    act = self.instantiate_action(tpl, mapping)
                    matched = [p for p in act.preconditions if p in current_state]
                    if not matched:
                        continue
                    if not self._adjacent_compatible(seq_actions[-1], act):
                        continue
                    new_pre = set(seq_pre)
                    for p in act.preconditions:
                        if p not in current_state:
                            new_pre.add(p)
                    # Subtasks are only valid when the initial state is empty-handed.
                    # If a later primitive still requires holding in the pre-state, the
                    # sequence is not self-contained enough to be a valid subtask.
                    if any(p.name == 'holding' for p in new_pre):
                        continue
                    new_role_types = {r: set(t) for r, t in role_types.items()}
                    valid = True
                    for var, role in mapping.items():
                        allowed = tpl.available_objects[var]
                        if role in new_role_types:
                            new_role_types[role] &= set(allowed)
                        else:
                            new_role_types[role] = set(allowed)
                        if not new_role_types[role]:
                            valid = False
                            break
                    if not valid:
                        continue
                    dfs(seq_actions + [act], new_pre, new_role_types, depth + 1)

        for tpl in self.templates:
            if not self._is_valid_start_template(tpl):
                continue
            existing_roles: Dict[str, Set[str]] = {}
            for mapping in self._candidate_mappings(tpl, existing_roles):
                act = self.instantiate_action(tpl, mapping)
                # All subtasks must start from an empty-handed initial state.
                seq_pre = set(act.preconditions)
                seq_pre.add(Predicate('empty', tuple()))
                if any(p.name == 'holding' for p in seq_pre):
                    continue
                role_types = {}
                for var, role in mapping.items():
                    role_types[role] = set(tpl.available_objects[var])
                dfs([act], seq_pre, role_types, 1)
        return results

    def compress_by_signature(self, subtasks: List[dict]) -> List[dict]:
        best = {}
        for st in subtasks:
            st = self._canonicalize_subtask_dict(st)
            key = (
                st['subtask_type'],
                tuple(st['pre_state']),
                tuple(st['net_add']),
                tuple(st['net_delete']),
                tuple((step['template_id'], step['action_type'], tuple(sorted(step['var_mapping'].items()))) for step in st['primitive_sequence']),
            )
            cur = best.get(key)
            if cur is None or st['length'] < cur['length']:
                best[key] = st
        out = []
        for i, st in enumerate(best.values(), 1):
            item = copy.deepcopy(st)
            item['subtask_id'] = f"subtask_{i:04d}"
            out.append(item)
        out.sort(key=lambda x: (x['subtask_type'], x['length'], x['subtask_id']))
        return out


def main():
    parser = argparse.ArgumentParser(description='Enumerate linear subtask templates from atomic task templates.')
    parser.add_argument('--atomic', default='./outputs/atomic_templates.json')
    parser.add_argument('--rules', default='./prompts/planner_rules.yaml')
    parser.add_argument('--max-len', type=int, default=8)
    parser.add_argument('--output', default='./outputs/subtask_templates.json')
    parser.add_argument('--compressed-output', default='./outputs/subtask_templates_compressed.json')
    args = parser.parse_args()

    gen = SubtaskGenerator(Path(args.atomic), Path(args.rules))
    subtasks = gen.enumerate_subtasks(max_len=args.max_len)
    compressed = gen.compress_by_signature(subtasks)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(subtasks, f, indent=2, ensure_ascii=False)
    with open(args.compressed_output, 'w', encoding='utf-8') as f:
        json.dump(compressed, f, indent=2, ensure_ascii=False)

    by_type = defaultdict(int)
    for st in compressed:
        by_type[st['subtask_type']] += 1

    summary = {
        'raw_subtask_count': len(subtasks),
        'compressed_subtask_count': len(compressed),
        'type_counts': dict(sorted(by_type.items())),
        'raw_output': args.output,
        'compressed_output': args.compressed_output,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
