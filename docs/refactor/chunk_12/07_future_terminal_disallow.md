# Future-terminal disallow semantics

## Purpose

When scanning bytes ends in a tokenizer state that can still match some terminal, and the current Commit step also accepted a shorter completed terminal, future longest-match behavior may require recording that the shorter terminal is disallowed if the longer terminal completes later.

## Function

`apply_future_terminal_disallow` inserts delayed exclusions only when the scanner end state can still match the same terminal group in the future.

## Invariant

The delayed exclusion is indexed by the tokenizer end state, not by the parser state. Parser branches carry the exclusion because different parser branches may have different histories, but the key that will be tested on a later Commit step is the scanner state.
