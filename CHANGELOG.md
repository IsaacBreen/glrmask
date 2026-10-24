# Changelog

## 0.4.0

0.4 is a new API. Code written for 0.3 needs changes; this section lists them.

### Grammars, modules and constraints

- Four types make up the API: `Grammar` (source text, or built from an expression), `Module` (a grammar with its extern
  bindings and any precompiled work), `Constraint` (a grammar compiled for a vocabulary) and `ConstraintState` (one sequence).
- `Grammar` owns its source and has no lifetime parameter. `from_ebnf`, `from_lark`, `from_glrm`, `from_json_schema` and
  `from_expr` parse the source straight away and return a `Result`.
- `grammar.compile(&vocab)` compiles directly. Use a `Module` to compose and reuse: bind its externs, `precompile` it while
  some are still unbound, `save` and `load` it, then `compile`.
- `Expr::choice`, `Expr::sequence` and `Expr::literal` combine grammars, modules and token sets. Each grammar keeps its own
  rules, including which whitespace it ignores, and one token may span several parts.
- Removed: `PreparedGrammar`, `UnlinkedConstraint`, `BuildOptions`, `Requirement`, `BoundaryTriggerDetail`,
  `UnboundExtern`, `link`, `requirements`, `with_options`, `set_options` and `Grammar::exact`.

### Externs

- A grammar declares `extern token NAME`, bound to a token ID or `TokenSet`, or `extern grammar NAME`, bound to a grammar,
  module or expression. In Lark, `<|name|>` declares a token extern and `@name` a grammar extern.
- `bind(name, value)` binds in place. Each extern is bound once, on the grammar or module that declares it. A value that
  has unbound externs of its own can't be bound.
- `externs()` lists the declared externs in order, and `externs().unbound()` the ones not yet bound.

### Tokens and vocabularies

- `TokenSet` replaces `Token` and `ExactTokens`. Create one from IDs or from ranges. Its IDs are checked against the
  vocabulary when you compile.
- Tokens with no bytes are now called special tokens: `Vocab::with_special_tokens(ids)`, Python `special_token_ids=`,
  Node.js `specialTokenIds`.
- `Vocab` has `new`, `with_special_tokens`, `get`, `iter`, `len` and `is_empty`. `len` counts only tokens with bytes.
- The Rust `TokenId` alias is removed.

### End tokens

- There are no end-token options. To require an end token, put it after the grammar:
  `Grammar::from_expr(Expr::sequence([Expr::from(grammar), Expr::from(TokenSet::from(eos))]))`. This costs nothing extra
  to compile or run.

### Generating

- `commit_token`, `commit_bytes` and the new `commit_tokens` return `bool` instead of `Result`. A failed commit ends the
  state: later commits fail, the mask is empty and `is_terminated()` is true. Copy the state to try a token without
  committing to it. `is_rejected` is removed.
- `commit_tokens` commits IDs in order and stops at the first one that isn't allowed.
- New `is_terminated()`: whether nothing more can be committed. `is_accepting()` still says whether the input so far is a
  complete match.
- `forced_tokens()` returns the token IDs, not a `Result`.

### Mask strategies

- `MaskStrategy` replaces `Optimization`. It's a compile setting, passed in `CompileOptions` (Python: keyword arguments;
  Node.js: the options object; C and C++: `glrmask_compile_options` and `CompileOptions`):
  - `Auto`, the default, reuses any precompiled work; otherwise it uses `ParserDwa`, or `TokenTrie` when that would be too
    large.
  - `ParserDwa` is slower to compile and has the fastest masks.
  - `TokenTrie` is the fastest to compile and has slower masks. Its settings choose the parser backend (`LrTable` or
    `TemplateDfa`) and whether to merge equivalent tokens.
  - `FAST_COMPILE` and `FAST_MASKS` name the two ends of the trade-off.
- `BoundaryMaskStrategy` makes the same choice for the boundaries between composed parts.
- A strategy you choose explicitly reports an error if it can't be used. Only `Auto` falls back.
- Strategies apply to parts that aren't compiled yet. Precompiled parts are always reused as they are.

### Saving

- `Constraint::save` no longer stores the vocabulary. `Constraint::load(bytes, &vocab)` checks that the vocabulary is
  the one it was saved with. `save_without_vocab` and `load_with_vocab` are removed.
- `Module::save` and `Module::load(bytes)` keep bindings and precompiled work, including for modules with unbound externs.
- Bytes saved by 0.3 can't be loaded.

### Errors

- `Error` has `kind()` and `source_span()`, and its message includes the line and column when they're known. `Error` and
  `ErrorKind` are `#[non_exhaustive]`.
- Messages are rewritten to say what went wrong and what to do about it.

### Grammar languages

- Lark has one dialect, with every GLRMask extension. `LarkOptions` and `LarkProfile` are removed.
- GLRM: `extern` is allowed only at the top level, an inline grammar can't refer to names outside it, `use`, `export`,
  `import` and `super` are reserved, and `shortest { ... }` is removed (use regular subtraction, or Lark's `[lazy]`).

### Bindings

- Python, Node.js, C and C++ have the same model as Rust, with names in each language's style.
- Python: `Module(grammar)`; `bind` accepts a token ID, and `bind(**bindings)` binds several externs at once; `externs`
  is a read-only mapping; `copy()` replaces `clone()`.
- Node.js: `bind` accepts an object to bind several externs at once, on `Grammar` as well as `Module`.
- C and C++: new `glrmask_grammar_unbound_externs`, `glrmask_token_set_equal` (C++ `TokenSet` `==`), and the C++
  `MarkerId` type. The C++ CMake package accepts any 1.x version request.

### Fixes

- Some grammars that used to fail at the build budget now compile.
- `TokenTrie` masks no longer hang on grammars with reduce cycles.
- Compiling a deeply nested grammar on a thread with a small stack no longer overflows it.

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
