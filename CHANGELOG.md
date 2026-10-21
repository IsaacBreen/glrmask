# Changelog

## Unreleased (0.4.0 candidate) — v1 API: modules, expressions and mask strategies

- The public pipeline is `Grammar` → `Module` → `Constraint`, plus `Expr` for
  composition. `Grammar` is immutable source without settings or bindings;
  `Grammar::compile` and `compile_with(CompileOptions)` are sugar over a
  default `Module`.
- `Module` holds per-module settings, bindings and retained work.
  `Module::bind` mutates in place and is atomic; each extern is bound once on
  its declaring module, and qualified `::` paths are rejected. A bound child
  must have no unbound externs. `precompile` retains local work while externs
  are still open; `compile` requires every extern to be bound.
- `MaskStrategy` (`Auto`, `ParserDwa`, `TokenTrie`, `TokenClassTrie`, with
  aliases `FAST_BUILD` = `TokenTrie` and `FAST_RUNTIME` = `ParserDwa`) replaces
  `Optimization`; `Balanced` is removed. Explicit strategies error instead of
  silently falling back. `TokenTrie` parts cannot be combined with `ParserDwa`
  or `TokenClassTrie` parts in one composition.
- `BoundaryMaskStrategy` (`FastBuild`, the default, and `FastRuntime`) replaces
  `BoundaryTriggerDetail`. The level is a minimum: retained work at a higher
  level is reused, never downgraded.
- `Externs` and `Extern` (`name`, `kind`, `span`) describe a module's declared
  externs, with an `unbound()` view. `CompileOptions` carries `end_tokens`,
  applied only to the root.
- Saved modules preserve cold, open, partly compiled and mixed graphs,
  including exact vocabulary identity, bindings and retained work.
- `Expr.choice`, `Expr.sequence` and `Expr.literal` compose grammars without
  adding a namespace per operator; model-token spellings may cross leaves.
- Python: `Module(grammar, *, mask_strategy=..., boundary_mask_strategy=...)`
  with assignable properties, `bind(name, value)` and atomic `bind(**bindings)`,
  an `externs` `Mapping` with `.unbound`, and `copy()` replacing `clone()`.
  C/C++ and Node.js bindings expose the same model.
- Grammar language: `extern` declarations are only allowed at the top level,
  inline grammars cannot refer to outer names, `use`, `export`, `import` and
  `super` are reserved, and `::` is invalid in extern names.
- Removed: `PreparedGrammar`, `link`, `requirements`, `Grammar::bind`,
  `Grammar::exact`, `Optimization`, `BoundaryTriggerDetail`, `UnboundExtern`,
  `with_options`/`set_options`, and Python `clone()`.
- Remove native GLRM `shortest { ... }` syntax. Preserve internal prefix-minimal
  regular languages and Lark `[lazy]`, and emit reparsable byte-correct regular
  subtraction from the formatter.
- Disable debug-info stripping on language-binding shared images to avoid the
  upstream Mach-O string-table alignment issue exposed by macOS 27. Core
  optimization and canonical performance profiles are unchanged.

## 0.3.1 — 2026-10-18

- Add native C ABI and C++17 RAII bindings, static/shared libraries, CMake installation and external-consumer support.
- Add a Node-API JavaScript/TypeScript package with asynchronous compilation and persistence, synchronous packed masks and commits, owning snapshots, explicit disposal and CommonJS/ESM support.
- Cover the current public source/prepared/constraint lifecycle, exact token sets and ranges, marker observations, forcing, and cross-language artifact interchange with matching builds.
- Add language guides, native build artifacts, platform validation and links to the hosted C++ and JavaScript documentation. The release workflow distributes verified native package and installation artifacts for each platform using tagged, version-aligned source.
- Overhaul Linux portability: reduce glibc minimum for Node and C++ artifacts via portable `manylinux_2_28` build container (glibc >= 2.28, matching Node 22+ runtime requirements), add native Linux musl x64 and ARM64 builds, relocated consumer tests and C++/Node packaging, and add Python musllinux wheels across Python 3.9–3.13.

## 0.3.0 — 2026-10-17

### Components and public APIs

- Use the `Grammar` → `PreparedGrammar` → `Constraint` → `ConstraintState`
  lifecycle. Source descriptions are parsed and cached once; `prepare()` retains
  reusable vocabulary-specific machinery and `link()` creates a closed runnable
  root. The old public `UnlinkedConstraint` / `compile_unlinked` spellings and
  implicit runnable-constraint child bindings are no longer exposed.
- Make `Grammar.bind(name, value)` and `PreparedGrammar.bind(name, value)`
  single-item mutations in both Rust and Python. They return `Result<()>` and
  `None`, respectively, and leave the previous configuration intact on error.
  Cheap `clone()` creates an independent handle; Python also supports `copy.copy`
  and `copy.deepcopy`. Attaching an open child captures its current snapshot.
  Later edits do not change parents, siblings, or already compiled constraints.
  Qualified `child::NAME` requirements retain their types and source spans.
- Keep explicit `extern token NAME` and `extern grammar NAME` contracts.
  `Token(id)` / `Token::new(id)` consumes one original ID and is checked against
  the target vocabulary. Vocabulary-qualified `ExactTokens` provides compact
  sets/ranges. Exact event values can bind a grammar port directly, with the
  semantics of `Grammar.exact`; arbitrary grammars still cannot bind token ports.
  Empty sets denote the empty language, never epsilon.
- Store prepared assemblies as a bounded DAG with shared compiled definitions
  and distinct instance environments. Reject malformed, excessively deep,
  unreachable or wrong-kind manifests before decoding compiled bodies. Older
  private artifact versions are rejected rather than guessed or upgraded.

### Grammar frontends

- Make native GLRM declarations layout-independent with optional semicolons,
  no required version header, `grammar NAME { ... }` scopes, backtick-escaped
  identifiers, explicit `edge` FA statements, and canonical formatting. Preserve
  `internal t`, maximal munch, ambiguous CFG recognition, regular algebra,
  structural alternative subtraction, and separated-sequence presence semantics.
- Add explicit Unicode/byte strings and regexes, checked useful flags and
  escapes, original-source diagnostics, and regular-terminal `shortest { R }`.
  The braced operator is compositional and replaces the function-call spelling.
  Keep independent ignored-terminal identities and component-local skip policy.
  Regex-internal reluctant quantifiers and nonregular assertions fail explicitly.
- Add a documented Lark recognition profile and an explicit LLM compatibility
  profile. Fix exact/bounded repetition, common imports, Unicode ranges, flags,
  start-rule handling and directive errors. The LLM profile supports exact-ID
  sets/ranges/complements, explicit named-token maps, `@name` grammar requirements,
  isolated `%json` fragments, and regular-body `[lazy]`.
- Do not add numeric-token, named-special or inline-JSON literals to native
  GLRM. Both frontends share the component/compiler machinery; neither accepts
  priorities, captures, substring generation, sampling annotations or rule budgets.

### Correctness and packaging

- Expose only `forced_tokens()` for a safe library-canonical prefix of genuinely
  forced text. Remove `forced()` and `forced_token_ids()` while retaining the
  canonical engine as an implementation detail. Ordinary text uses longest
  admitted spelling and lowest-ID alias selection; exact events retain their IDs,
  and greedy text never crosses an intermediate exact-event opportunity. The
  correctness-first reference may construct full masks. `commit_bytes()` remains
  a separate raw-byte recognition operation, and typed resource failures remain
  errors rather than fabricated ambiguity or completion.
- Correct trailing-unmapped vocabulary-coordinate refinement, productionless
  start coordinates and zero-terminal component ownership, with independent
  byte/event and empty-language regressions.
- Include Python type stubs and `py.typed`, and exclude development interpreter
  caches from distributable wheels. Updated guides describe the canonical API
  and the deliberate native/Lark dialect differences.

- Normalize platform line endings in distribution README metadata checks,
  without accepting actual description drift.
- Add matched core-batch qualification results, portable comparison drivers,
  and explicit remaining performance/scaling limitations in
  `docs/validation/core-batch-2026-10-15.md`.

### Construction efficiency and concurrency

- Share immutable binding snapshots and declaration metadata. Incremental
  instance summaries replace repeated whole-graph discovery during binding and
  closure checks; requirement paths are materialized when requested. Full
  validation remains at artifact trust boundaries.
- Reuse exact-event adapters across prepared links instead of recompiling the
  same event set each time. Cache ownership is limited to the prepared definition;
  no new registry or lookup is added to ordinary runtime masking.
- Release Python's GIL during substantial native binding and requirement work.
  Independent concurrent edits use cloned handles; overlapping access to the
  same mutating handle reports a borrow error rather than losing updates.
- Respect caller-owned Rayon pools during compilation and composition, avoiding
  redirection of independent jobs into a capped or serial helper pool. Keep the
  documented single-build defaults outside a caller-owned execution context.

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
