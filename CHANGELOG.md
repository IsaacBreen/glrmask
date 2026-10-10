# Changelog

## 0.2.0 — 2026-10-10

### Public API

- Rust and Python share the `Grammar`, `UnlinkedConstraint`, `Constraint`, and
  `ConstraintState` lifecycle. Immutable bindings support source grammars,
  reusable compiled children, and vocabulary-qualified exact tokens.
- `FastBuild` / `FAST_BUILD` selects ordinary Dynamic/O1; `Balanced` / `BALANCED`
  selects native O2; `FastRuntime` / `FAST_RUNTIME` selects Static; `Auto` / `AUTO`
  retains the existing default. Every mode returns the same public constraint
  and state types. Parser backend selectors and data-only parser providers remain
  unstable tooling under Rust's `internal-api` feature and Python's `_internal`.
- Public persistence supports compiled constraints and open pre-link artifacts.
  Python uses `Constraint.save(*, external_vocab=False)`; Rust uses
  `save_without_vocab()` for external-vocabulary saves. Loading an external
  artifact validates the exact vocabulary identity.
- Boundary queries are selected through Rust `BuildOptions.boundary_trigger` or
  Python's `boundary_trigger` keyword. Exact boundary triggers are unsupported
  by retained-LR Dynamic and return an error. Pre-link artifacts retain requests
  until final link, whose explicit options can override the root request.
- Python exposes exact state snapshots with `ConstraintState.clone()`.
- Remove the public `is_terminated()` accessor. Stop at the first complete match
  with `is_accepting()`, or continue until a configured end-token ID is sampled
  and committed. An allowed end-token commit empties the next mask, preserves
  acceptance, and rejects further commits.

### Correctness and implementation

- Retained-LR Dynamic builds executable actions for compressed direct-regular
  languages. Static declines terminal-run stabilization proofs based on
  admission-only compact tables. Valid whole-word Lark tokens remain maskable
  in both paths, including vocabularies containing only that token.
- Native O2 and Static parser representations execute without retained LR
  action/goto tables. Retained-LR artifacts stay LR and native artifacts stay
  native when loaded, independently of the compilation environment.
- Dynamic masking shares correlated lexer/parser recognizer states during a
  vocabulary-trie walk. Compiler, boundary linking, exact bounded-terminal
  synthesis, automaton minimization, and runtime storage have been improved.
  Exact recognition, token masks, commit behavior, and completion remain the
  correctness contract.
- Serialization retains exact tokenizer transitions and liveness metadata,
  bounds decompression, and preserves packed runtime data. Constraints prepare
  runtime caches during compilation/loading rather than hiding that work in
  the first state operation.
- Python uses mimalloc with delayed automatic purging and reset-based purges.
  Pages remain OS-reclaimable; set `MIMALLOC_PURGE_DECOMMITS=1` before import
  when immediate RSS reduction is preferred.

### Compatibility and evidence

- This release changes the Rust and Python APIs from 0.1.1. Use the examples and
  guides at the `v0.2.0` tag; integrations written for the older API must adapt.
  Serialized artifacts are pre-release formats: rebuild old artifacts when the
  current loader rejects them. No cross-release artifact compatibility is promised.
- Python wheels target CPython 3.9–3.13 on manylinux x86_64/aarch64, macOS
  x86_64/arm64, and Windows x86_64. Source installation requires Rust and native
  build tools. Python 3.14 and other platforms are outside this wheel matrix.
- The root Rust crate and Python distribution are 0.2.0. The extracted helper
  crates make their first registry publications at their existing 0.1.1 versions;
  root dependencies pin those exact versions.
- The README's corrected 20 August 2026, official-9,558-schema engineering run
  remains an interim historical benchmark, not a measurement of this release.
  Later reports under `docs/performance/` retain their own source, workload,
  machine, and evidence provenance. Do not combine their timings into an
  unqualified full-corpus or cross-platform performance claim.

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
