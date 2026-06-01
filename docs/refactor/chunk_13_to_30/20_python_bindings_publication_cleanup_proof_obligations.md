# Chunk 20 proof obligations: python_bindings_publication_cleanup

## Equivalence obligation

The refactor is valid only if the denotation of the moved subsystem is unchanged.  For this chunk, the denotation is whatever object or relation is owned by:

```text
bindings/python
```

The proof obligation is structural rather than syntactic: moving files does not matter; changing which relation is computed does.

## Local invariants

1. Inputs and outputs must stay in the same id spaces as before.
2. Any quotient map must be carried with the artifact it interprets.
3. Any cache must be derivable from semantic fields.
4. Any fast path must have a reference path.
5. Any compatibility shim must be behaviorally inert.
6. Any profiler must not affect semantic state.
7. Any diagnostic must not trigger extra construction except explicitly requested debug work.
8. Any public API method must keep the same high-level contract.

## Suggested tests after compile repair

- Construct the smallest nontrivial object in this subsystem.
- Construct a second object that exercises nondeterminism or quotienting.
- Compare old-path shim behavior to canonical path behavior.
- Round-trip through serialization if this subsystem contributes artifact fields.
- Exercise an empty/degenerate case and a maximal small case.
- Verify deterministic output order where user-visible diagnostics are produced.

## Review proof sketch template

A reviewer should write a short proof in this form:

```text
Before the refactor, function F denoted relation R over domains A and B.
After the refactor, the canonical function F' delegates to the same operations,
with all ids interpreted in the same coordinate maps. Compatibility shim S is
only a reexport. Therefore F and F' denote R.
```

If the proof cannot be written in five sentences, the subsystem needs a stronger boundary or more tests.
