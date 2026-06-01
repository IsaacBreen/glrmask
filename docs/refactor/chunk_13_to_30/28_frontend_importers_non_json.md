# Chunk 28: Frontend importers other than JSON Schema

## Purpose

This chunk completes one remaining publication-cleanup area after the first twelve structural chunks.  It is deliberately self-contained: a reader should not need the historical plan to understand what changed, what the target architecture is, and how to continue compile repair later.

## Files and directories in scope

- `src/import/README.md`
- `src/import/ebnf/README.md`
- `src/import/ebnf/mod.rs`
- `src/import/json_schema/diagnostics.rs`
- `src/import/json_schema/load/README.md`
- `src/import/json_schema/load/collect.rs`
- `src/import/json_schema/load/keywords.rs`
- `src/import/json_schema/load/mod.rs`
- `src/import/json_schema/load/pointers.rs`
- `src/import/json_schema/load/shape.rs`
- `src/import/json_schema/load/typed.rs`
- `src/import/json_schema/lower/README.md`
- `src/import/json_schema/lower/array/README.md`
- `src/import/json_schema/lower/array/mod.rs`
- `src/import/json_schema/lower/mod.rs`
- `src/import/json_schema/lower/number/README.md`
- `src/import/json_schema/lower/number/mod.rs`
- `src/import/json_schema/lower/object/README.md`
- `src/import/json_schema/lower/object/mod.rs`
- `src/import/json_schema/lower/string/README.md`
- `src/import/json_schema/lower/string/mod.rs`
- `src/import/json_schema/mod.rs`
- `src/import/json_schema/normalize/README.md`
- `src/import/json_schema/normalize/combinators.rs`
- `src/import/json_schema/normalize/mod.rs`
- `src/import/json_schema/options.rs`
- `src/import/json_schema/schema/array.rs`
- `src/import/json_schema/schema/assertions.rs`
- `src/import/json_schema/schema/document.rs`
- `src/import/json_schema/schema/mod.rs`
- `src/import/json_schema/schema/object.rs`
- `src/import/json_schema/schema/scalar.rs`
- `src/import/json_schema/tests/mod.rs`
- `src/import/lark/README.md`
- `src/import/lark/mod.rs`
- `src/import/mod.rs`
- `src/import/numeric_range/README.md`
- `src/import/numeric_range/mod.rs`

## Priority

Priority level: **publication-shaping / high**.  These changes are primarily about making the mathematical architecture visible.  They should be completed before detailed compile repair because compile errors are much easier to repair once the target module boundaries are correct.

## Target abstraction

The target abstraction for this chunk is not “a set of Rust files”.  It is a named mathematical object or policy boundary.  The source tree should encode that object directly.  Names that describe accidents of implementation, old experiments, or temporary benchmark harnesses should be demoted to compatibility shims or deleted.


## Mathematical reading discipline

For every function in this area, classify it before editing:

1. **Denotation constructor** — builds a language, relation, quotient, automaton, or transition system.
2. **Representation transformer** — changes storage while preserving denotation.
3. **Evaluator** — applies an already-built object to a state, token, byte, or stack.
4. **Policy reader** — chooses an algorithm or diagnostic mode.
5. **Reporter** — formats diagnostics or profiles without changing semantics.

Functions from different classes should not be interleaved unless a module is explicitly an orchestrator.

## Definition of done

A chunk is done when a beginner can answer these questions by looking only at file names and short module headers:

- What mathematical object does this directory own?
- What does each child file own?
- Which file is the public boundary?
- Which files are compatibility shims?
- Which operations are semantic, and which are optimizations?
- Which invariants must tests check after compile repair?


## Concrete application rules

1. Keep compatibility shims small and visibly marked with `#[doc(hidden)]` where possible.
2. Move canonical code into a module named after its denotation.
3. Update imports in canonical code to the new path.
4. Do not hide a semantic operation inside an optimization module.
5. Do not put environment-variable parsing inside a pure mathematical algorithm unless it is temporarily documented as legacy.
6. Every README must state both denotation and forbidden dependencies.
7. Old names may remain only as shims, not as the preferred path in new code.
8. Every large file left unsplit must be called out as a remaining mechanical extraction target.

## Invariants to test after compile repair

- Moving files did not change the recognized language or runtime transition relation.
- Every compatibility shim reexports exactly the canonical module and no new logic.
- Every quotient map is applied consistently to all ids in its artifact.
- Every fast path has a direct reference path with equal denotation.
- Diagnostic/profiling code cannot mutate semantic state except for cache/stat counters.

## Review checklist

- Read the directory README first.
- Confirm public names match paper names.
- Confirm no old path is used by canonical code.
- Confirm examples describe public API, not internal modules.
- Confirm future compile-repair notes are exact enough to execute mechanically.

## Deferred compile repair notes

This pass intentionally does not compile.  The repair order is: path imports, visibility, module declarations, formatting, clippy, unit tests, integration tests, serialization tests, benchmark parity.  Do not start by deleting compatibility shims; they are temporary scaffolding for the first compile repair pass.
