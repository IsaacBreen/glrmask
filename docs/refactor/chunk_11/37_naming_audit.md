# Runtime naming audit

Chunk 11 makes one concrete naming correction and records several deferred ones.

## Applied

- `end_state_may_advance` became `end_state_can_advance`.

Reason: the predicate is exact.  It asks whether the parser stack can advance on
some terminal accepted from the tokenizer end state.  It is not estimating.

## Deferred

- `may_advance_ns` profile fields still contain `may`.

Reason: profile names may be consumed by benchmark scripts or Python users.  A
future diagnostics compatibility chunk should rename or alias them.

## Desired style

Use `can` when a predicate is exact:

```text
stack_can_advance_on
end_state_can_advance
token_can_scan_from_state
```

Use `maybe` for optional optimization attempts:

```text
maybe_enable_json_u_prefix_token
maybe_use_delta_replay
```

Use `try` when a function may decline and fall back:

```text
try_fill_mask_single_path_direct
try_commit_direct_linear_fast_path
```

Avoid `game`, `magic`, `special`, and `misc` names.  They hide mathematical
structure.
