# Basic implementer instructions

This file assumes you know basic programming but not the whole project.

## Rule 1: do not put code back into `mod.rs`

When you need to add a helper, ask what it does:

- Does it build or cache the ordered vocabulary? Put it in `ordered_vocab.rs`.
- Does it decide whether tokens are equivalent for CanMatch? Put it in
  `vocab_equivalence.rs`.
- Does it build interval maps? Put it in `collector.rs`.
- Does it turn interval maps into weights? Put it in `vocab_materialize.rs`.
- Is it the old validation algorithm? Put it in `legacy_materialize.rs`.
- Is it the small sparse root path? Put it in `root_collect.rs`.
- Is it just wiring phases together? Put it in `compute.rs`.

## Rule 2: never reuse Terminal-DWA maps for CanMatch

If you see a map from Terminal DWA construction and you are tempted to use it in
scan relation construction, stop.  These maps are not the same mathematical
relation.

## Rule 3: keep runtime and compile-time separate

Runtime commit scans one token.  Compile-time scan relation scans all relevant
tokens and all relevant states.  Do not call compile-time collectors from commit.

## Rule 4: names should answer “what relation is this?”

Avoid names like:

```text
possible
pm
thing
map2
```

Prefer:

```text
can_match
scan_relation
completed_terminals
partial_lexer_state
ordered_vocab
state_terminal_label
```

## Rule 5: one file, one reason to change

If a future change touches `ordered_vocab.rs`, the reason should be vocab
ordering/cache.  If it touches `vocab_materialize.rs`, the reason should be
weight materialization.  If a change touches five scan-relation files, check
whether it is actually a boundary change.
