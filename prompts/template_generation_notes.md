# What extra prompt content you still need

Your current `env.yaml` already does three important things:
1. It explains object metadata fields.
2. It says labels are the main grounding signal.
3. It says affordances are authoritative and hidden state must not be assumed.

But for **template generation**, that is still not enough.

You still need these extra prompt layers:

## 1. Output schema prompt
Tell the model the exact JSON schema to output.
Without this, it will drift between instance-level tasks and template-level tasks.

## 2. Template abstraction prompt
Explicitly say:
- generate **template-level** operators
- merge same `(operator_family, canonical_label)`
- keep `candidate_object_ids` for later instantiation
- do not generate one template per object instance

## 3. Dedup / normalization prompt
Explicitly ask for:
- canonical label normalization
- no duplicates like `Pick Book`, `Pick Book 2`, `PickUp Book Instance A`
- one family-level template per canonical label

## 4. Hidden-state restriction prompt
Your env prompt already says this, but repeat it in the generation prompt:
- do not assume open/closed/on/off current state
- do not assume containment/support from contact alone

## 5. Primitive vs semantic task prompt
Tell the model:
- `rotation` is a primitive exploration template
- `put_down` is inventory-level generic
- do not hallucinate `put_on(table)` or `put_in(cabinet)` unless explicit relations exist

## 6. Intent anchor prompt for later graph composition
This is not needed for atomic template extraction itself,
but you will need it immediately in the next stage.
OmniBench shows that resource-only composition creates meaningless graphs,
so you should add an intent-anchoring prompt before composition.

## 7. Validator prompt
Add a second pass that checks:
- affordance legality
- duplication
- hidden-state hallucination
- unsupported relations
