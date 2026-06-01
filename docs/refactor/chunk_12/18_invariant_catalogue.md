# Commit invariant catalogue

## Invariant 1: offset monotonicity

Every queue transition must increase the byte offset by the matched width. Zero-width terminal matches must not be enqueued into the same offset loop unless explicitly handled by parser semantics elsewhere.

## Invariant 2: ignored terminals do not advance parser stacks

Ignored terminals move through bytes and reset to the tokenizer initial state, but leave the parser frontier unchanged except for representation-level merge/fuse effects.

## Invariant 3: non-ignored terminals require parser advance

A non-ignored terminal contributes a pending state only through `advance_parser_stacks` or an equivalent shortcut.

## Invariant 4: residual tokenizer states are guarded

A residual tokenizer end state is retained only if the parser frontier can still advance on a terminal that can complete from that tokenizer state.

## Invariant 5: delayed exclusions are branch-local

Delayed longest-match exclusions are attached to parser branches. Merging and fusion must preserve their branch semantics.

## Invariant 6: profiling is observational

Profile collection may clone, summarize, or time internal data. It must not alter the produced runtime state.
