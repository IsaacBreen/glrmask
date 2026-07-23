# Changelog

## Unreleased

### Improved

- Large bounded JSON Schema string patterns now retain exact `maxLength`
  semantics by compiling terminal/parser automata against a certified smaller
  residual representative while keeping the full exact lexer for runtime
  state. Pathological bounded-repeat intersections no longer force the former
  multi-second terminal-DWA construction path.
- Exact runtime tokenizer finalization now overlaps terminal/parser automaton
  construction. Large protected residual products publish their synthesized
  compile tokenizer before full future analysis and byte-transition
  materialization, and runtime component assembly moves transition storage
  instead of copying it a second time.
- Bounded-terminal synthesis candidate discovery now runs before terminal
  materialization, uses cached vocabulary length/alphabet statistics, reuses
  language-canonical repeat-horizon proofs, and parallelizes only the small set
  of qualified candidates. Grammars with no viable candidate avoid synthesis
  allocations entirely, and the diagnostic opt-out shares the same eligibility
  scan so no-candidate ON/OFF builds follow the same planning path.
- Expensive constrained array items are no longer duplicated into separate
  first/next contextual terminals when one item already exceeds the importer
  product budget. The item is compiled once and its count remains enforced at
  grammar level.

### Changed

- Finite `maxLength` constraints on patterned JSON Schema strings are preserved
  by default even when their estimated pattern/length product is large. The
  complexity budget now selects lowering strategy only; it no longer permits a
  semantically weaker grammar.
- Exact bounded-terminal synthesis is enabled by default. Runtime keeps the
  full exact tokenizer while terminal/parser DWA construction may use a
  certified smaller representative. Set
  `GLRMASK_SYNTHETIC_BOUNDED_TERMINALS=0` only for diagnostics.

## 0.1.1 — 2026-07-19 — runtime, integration, and tail-latency update

### Added

- Grammar-level end-token IDs for JSON Schema, EBNF, Lark, and GLRM constructors. End tokens are exact parser terminals rather than byte spellings or metadata stored on `Vocab`.
- Bounded token-level rollback for speculative decoding, with zero retained history by default.
- Non-mutating proposal validation that returns the longest admissible token prefix.
- Explicit failed-state inspection for recovery after an invalid commit.
- A llama.cpp-oriented vocabulary construction path and expanded integration examples.

### Improved

- Dynamic masking now precompiles and caches exact residual token programs, selects overlays by structural family, and avoids redundant parser simulation and continuation-partition construction.
- Dynamic mask and artifact paths received additional indexing, cache, serialization, and tail-latency work.
- README performance figures, dark-mode assets, runtime-mode documentation, and full-corpus benchmark links were revised.

### Changed

- `Vocab` no longer owns a distinguished EOS field. Consumers pass one or more `end_token_ids` when compiling a constraint; those tokens may also retain ordinary byte semantics if present in the byte vocabulary.
- Dynamic constraint artifacts use a new format version. Older artifacts without Vocab-level EOS metadata are migrated; artifacts that depended on the removed EOS metadata fail explicitly and must be rebuilt.
- Importer-level complex anchored-pattern splitting is available through `GLRMASK_JSON_SCHEMA_SPLIT_COMPLEX_PATTERNS=1` but is disabled by default.

### Integration compatibility

- The frozen vLLM backend requires `glrmask >= 0.1.1` for bounded rollback, non-mutating validation, failed-state inspection, and grammar-level end-token support.
- Public `glrmask 0.1.0` remains installable but is not compatible with that backend.

## 0.1.0 — 2026-07-15 — Shingleback initial release

### Highlights

- Public project brand: Shingleback; the Rust crate, PyPI distribution, and Python import remain `glrmask`.
- Vocabulary-specific grammar-constrained decoding for EBNF, Lark, and a documented pragmatic subset of JSON Schema.
- Reusable compiled `Constraint` objects with incremental mask, commit, completion, and forced-prefix operations.
- GLR-based parsing for ambiguous and genuinely context-free grammars, including tokenizations that cross grammar-terminal boundaries.
- Rust and Python APIs for incremental mask, commit, completion, and forced-prefix operations.
- Constraint serialization for compile-once, load-and-run deployments, plus a smaller execution-only runtime crate for serving artifacts.
- A build-only Python wheel workflow covering Python 3.9–3.13 across manylinux x86_64/aarch64, macOS x86_64/arm64, and Windows x86_64.

### Release evidence and caveats

- The bounded v0.1 `make example-slow-all` comparison is documented in [`docs/benchmark-0.1.md`](docs/benchmark-0.1.md), including exact scope, environment, backend versions, methodology, and caveats.
- JSON Schema support is not full specification conformance; see [`docs/json-schema-semantic-deviations.md`](docs/json-schema-semantic-deviations.md).
