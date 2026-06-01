# Force is derived, not primitive

`ConstraintState::force` is a convenience observation.  It is not a third online
operation alongside Mask and Commit.

The implementation now lives in `runtime/state/force.rs` to make this explicit.
It uses existing public reasoning:

1. If the state is complete, no token is forced.
2. Try to infer a forced byte prefix by repeatedly consulting Mask.
3. Greedily tokenize that byte prefix when doing so is unambiguous.
4. Fall back to repeated single-token Mask/Commit reasoning.

The important publication point is that `force` should not define correctness.
It is a derived helper that must be correct because Mask and Commit are correct.
This matters because forced-token logic is heuristic-looking: it talks about
first bytes, greedy tokenization, and stopping when a longer token could exist.
Putting it in `state/force.rs` avoids making `state/mod.rs` look like the
fundamental runtime semantics depends on tokenization heuristics.

Future tests for `force` should compare it against repeated Mask calls on small
vocabularies.  They should not be used as primary tests of Mask or Commit.
