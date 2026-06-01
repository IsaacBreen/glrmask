# Partial-boundary test design

This chunk does not add executable tests, but it defines the exact tests that
should be added once compilation resumes.

## Test goal

Construct a case where a vocabulary token ends inside a terminal match.  The
mask should admit the token only if the parser can accept at least one terminal
that can complete the partial lexer state.

## Minimal conceptual grammar

A useful grammar shape is:

```text
start = WORD
WORD  = "ab"
```

Vocabulary:

```text
v0 = "a"
v1 = "b"
v2 = "x"
```

Scanning `v0` from the lexer start state reaches a partial state: no complete
`WORD` has ended, but `CanMatch(q_after_a)` contains `WORD` because byte `b` can
finish it.

Expected behavior:

- At parser start, token `"a"` may be allowed because it can lead to `WORD`.
- Token `"x"` is blocked.
- After committing `"a"`, token `"b"` is allowed.
- After committing `"b"`, the parser has completed `WORD`.

## Negative companion test

Grammar:

```text
start = OTHER
WORD  = "ab"
OTHER = "x"
```

At parser start, token `"a"` should not be allowed merely because it begins
`WORD`; `WORD` is not parser-admissible.  This catches the bug where CanMatch is
computed but not intersected with parser admissibility.

## Equivalence non-reuse test

Construct two token byte strings that complete the same terminal prefix but end
in partial states with different future completions.  They may be equivalent for
completed-terminal behavior but not for CanMatch behavior.

Expected behavior: CanMatch vocabulary quotient keeps them separate.

## Serialization test

Compile the constraint, serialize it, load it, and repeat the partial-boundary
mask/commit sequence.  Expected behavior: identical masks and commits before and
after load.
