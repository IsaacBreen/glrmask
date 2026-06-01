# Runtime transition diagram

The runtime loop can be read as the following diagram:

```text
             ┌──────────────────────┐
             │ immutable Constraint │
             └──────────┬───────────┘
                        │ borrowed by
                        ▼
              ┌───────────────────┐
              │ ConstraintState S │
              └──────┬──────┬─────┘
                     │      │
             Mask_C(S)      Commit_C(S, bytes)
                     │      │
                     ▼      ▼
          original-token    successor ConstraintState S'
          bitset
```

Mask must not change `S`.  Commit must change `S` only by replacing the semantic
frontier and bumping generation.  Cache and scratch writes are permitted because
they are not part of the denotation.

The direct relation to decoding is:

```text
while not state.is_complete():
    mask = state.mask()
    token = model.sample(mask)
    state.commit_token(token)
```

Every file in `runtime/` should be explainable in terms of one node or arrow in
this diagram.
