# Concrete frontier examples

Example 1: deterministic parser, boundary tokenizer state.

```text
state = { q0 -> stack [0, 4, 7] }
```

Mask walks one parser stack through the Parser DWA.  Commit scans bytes from
`q0`, emits terminals, and advances that one stack.

Example 2: partial lexer state.

```text
state = { q_string_escape -> stack [0, 5] }
```

The tokenizer is in the middle of matching a terminal.  Mask must only allow
vocabulary tokens whose bytes can complete or continue a valid terminal match.
Commit may accept bytes without emitting a grammar terminal yet.

Example 3: parser ambiguity.

```text
state = {
  q0 -> GSS with two top parser states,
  q1 -> GSS with one top parser state and delayed exclusions,
}
```

Mask must union admissible continuations across active branches while respecting
branch-local terminal exclusions.  Commit must advance every surviving branch
and merge equivalent successor frontiers.

These examples explain why the semantic state is a map, not a single stack.
They also explain why scratch buffers must not be treated as semantic state.
