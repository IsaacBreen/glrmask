# Delayed longest-match pruning

## What is being pruned

Lexer longest-match exclusions can be delayed across Commit steps. A parser branch may carry a map from tokenizer states to terminals that are disallowed if those terminals later appear as completed matches from the corresponding scanner context.

## Source boundary

`pruning.rs` owns:

- `prune_initial_states`
- `prune_single_initial_state_for_exec`
- `prune_single_initial_state_for_terminal`
- `apply_future_terminal_disallow`
- `end_state_can_advance`

## Key invariant

Pruning removes parser branches only when all relevant accepted terminals are disallowed on that branch. Remapping moves delayed exclusions from old tokenizer states to residual tokenizer states. It must not invent exclusions for states that are not residual states of the current scanner execution.

## Why this is mathematically separate from parser advance

Parser advance consumes completed terminals. Longest-match pruning is a lexer-consistency side condition. Putting both in the same function made the old file harder to reason about. The new layout lets review ask two different questions:

1. Did the parser advance relation produce the correct stack effects?
2. Did delayed lexical exclusions remove exactly the branches forbidden by longest-match semantics?
