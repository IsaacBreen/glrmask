# Future refactor options unlocked by Chunk 07

This chunk is a structural prerequisite. It intentionally stops before rewriting algorithms. The following options are now feasible.

## Option A: split `Lowerer` into contexts

Current `Lowerer` is still a large mutable context. Later, it could be split into:

- `RuleEmitter`
- `TerminalRegistry`
- `NameAllocator`
- `NullabilityOracle`
- `RepeatCache`
- `GeneratedRuleSink`

That split should happen only when compile/test work begins because it will touch many method signatures.

## Option B: turn env-var shape policy into compile options

`repeat_tree_shape` and `comma_sep_shape` still read environment variables. This is not publication ideal. Later, those should be fields of `CompileOptions`, threaded into grammar lowering.

## Option C: make transforms a typed pipeline

Instead of ad hoc calls from importers, transforms could be an explicit pipeline:

```rust
NamedGrammarTransforms::default()
    .factor()
    .simplify()
    .lower_exact_subtractions()
    .promote_choice_terminals()
    .apply(grammar)
```

This would make frontend behavior more reproducible and easier to document.

## Option D: move GLRM frontend behind `frontend::glrm`

`grammar_ir::glrm::from_glrm` is currently a parser implementation. Public frontend entrypoints should eventually live under `frontend`, while grammar IR retains only representation and exchange-format support.

## Option E: property-test lowering laws

The new boundaries make it easier to write property tests:

- repeat range language equivalence for small alphabets;
- separated sequence separator placement;
- GLRM round-trip parse/render;
- transform idempotence for simplification;
- exact-subtraction partition correctness.
