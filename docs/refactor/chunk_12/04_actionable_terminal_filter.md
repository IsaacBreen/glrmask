# Actionable-terminal filtering

## Definition

A matched terminal is actionable for a parser frontier when at least one active parser top state can advance on that terminal. Ignored terminals are special: they do not require parser advance, so they should not be rejected merely because the parser action table has no transition for them.

## Source boundary

`acceptance.rs` defines `ActionableTerminals`, `is_ignored_terminal`, `is_actionable_terminal`, and normalized-match collection.

## Reasoning obligation

The filter is an optimization and pruning device. It must never reject an ignored terminal, and it must never reject a non-ignored terminal if any active stack can advance on it. It is valid for it to return a conservative `ManyStates` summary of parser tops; it is not valid for it to inspect deeper stack structure except through the parser table's advance-row predicate.
