# Commit runtime

Commit is the runtime transition relation for accepted bytes.

Mathematically, a live state is a finite map

```text
lexer_state -> parser frontier
```

where each parser frontier is a GSS annotated with delayed longest-match
exclusions. A Commit step scans a byte fragment from every live lexer state,
turns completed lexer matches into grammar-terminal observations, advances the
parser frontier through the stack-effect relation for each completed terminal,
and preserves lexer residual states only when the corresponding parser frontier
can still complete a future terminal.

The module is deliberately split by sub-relation rather than by optimization
level:

- `mod.rs`: routing layer and shared imports only.
- `api.rs`: public `ConstraintState::commit_*` methods.
- `general.rs`: reference queue-based Commit transition.
- `fast_path.rs`: semantic-preserving fast paths for common Commit shapes.
- `profiled.rs`: profiled variants and per-advance diagnostics.
- `acceptance.rs`: actionable-terminal filtering and normalized match sets.
- `initial_scan.rs`: first-offset scanner summary used before pruning.
- `pruning.rs`: delayed longest-match exclusion pruning and future-terminal exclusions.
- `queue.rs`: parser-frontier merge, queue, and final fusion helpers.
- `single_top.rs`: single-top parser-stack effect shortcuts.
- `terminal_advance.rs`: cached terminal-advance helper for queue processing.
- `parser_advance.rs`: dispatch between template-DFA stack effects and reference GLR advance.
- `template_advance.rs`: template-DFA execution machinery.
- `tokenizer_scan.rs`: primitive tokenizer scan bridge.
- `token_lookup.rs`: original-vocabulary token id to byte-string lookup.
- `mask_assert.rs`: optional Mask/Commit equivalence assertion.
- `options.rs`: Commit-local environment switches.
- `profile.rs`: public profiling records and private profile accumulation helpers.
- `types.rs`: local aliases, small records, and fast-path result enums.

Commit must not compute Mask. The optional mask assertion deliberately lives at
the API boundary and is not part of the transition relation.
