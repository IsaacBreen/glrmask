# Chunk 03 — Compile pipeline as an explicit phase graph

This document is the self-contained implementation manual for chunk 03.  It is intentionally long.  It is meant to be useful even to a future maintainer who has not read the earlier cleanup-plan zips, has not memorized the paper, and only knows Rust basics.

## 1. Purpose

The old compile pipeline was a single gravitational center.  Almost every compile-time concept passed through one file and was represented by local variables rather than named phase outputs.  That makes the code hard to publish because the implementation does not visibly match the paper.  The paper has named objects: a grammar normal form, a tokenizer/lexer, a Terminal DWA, a scan relation / CanMatch object, stack-effect template recognizers, a Parser DWA, and a final runtime constraint.  The code should show exactly those objects.

Chunk 03 does not try to make all downstream modules perfect.  It draws the boundary that future chunks can refine.  The crucial change is this:

> The compile pipeline is now a composition of typed phase functions.  Each phase has named inputs and named outputs.  Environment parsing, thread-pool policy, profile emission, tokenizer construction, artifact-size accounting, Parser-DWA/CanMatch reconciliation, and runtime-finalization layout are no longer mixed into one file.

## 2. What changed structurally

| File | Change |
| --- | --- |
| `src/compile/mod.rs` | Declares compile-object modules, making `pipeline`, `options`, `profiling`, `thread_pool`, and `tokenizer` peers of `terminal_dwa`, `scan_relation`, and `parser_dwa`. |
| `src/compile/options.rs` | Centralizes environment-backed compile decisions, including boolean flags, tokenizer-detail profiling, DWA/CanMatch reconciliation mode, and compile thread count. |
| `src/compile/profiling.rs` | Defines `CompilePhaseProfile`, profile summary rendering, explicit profile sinks, and template profile emission. |
| `src/compile/thread_pool.rs` | Owns the optional compile-specific rayon thread pool. |
| `src/compile/tokenizer.rs` | Owns grammar-terminal-to-tokenizer construction and documents tokenizer/vocab independence. |
| `src/compile/pipeline/mod.rs` | Small phase orchestrator: prepare grammar, analyze, build terminal/scan/template artifacts, reconcile, finalize. |
| `src/compile/pipeline/phases.rs` | Defines the ordered phase vocabulary and phase labels/descriptions. |
| `src/compile/pipeline/context.rs` | Defines typed intermediate outputs for phase boundaries. |
| `src/compile/pipeline/analysis.rs` | Builds tokenizer plus parser/table facts and computes disallowed follows. |
| `src/compile/pipeline/terminal_scan.rs` | Builds shared classification support, Terminal DWA, and scan relation. |
| `src/compile/pipeline/templates.rs` | Builds stack-effect templates and commit-specialized template DFAs. |
| `src/compile/pipeline/reconcile.rs` | Builds Parser DWA and reconciles Parser-DWA/CanMatch/Terminal-DWA ID spaces. |
| `src/compile/pipeline/finalize.rs` | Assembles runtime `Constraint` and rebuilds runtime caches. |
| `src/compile/pipeline/counts.rs` | Holds interned-range accounting utilities for reconciliation. |
| `src/compiler/compile.rs` | Becomes a compatibility facade into `compile::pipeline` and `compile::profiling`. |
| `src/compiler/mod.rs` | Stops declaring the old `compiler::pipeline` implementation as a central module. |
| `src/compiler/pipeline.rs` | Is reduced to a deprecated compatibility shim if included again later; it no longer contains the implementation. |
| `src/runtime/mod.rs` | Re-exports `TemplateDfasByTerminal` crate-privately so finalization can name the runtime template vector type. |
| `src/import/mod.rs` | Imports compile/profile entry points from the new compile namespace. |

## 3. Mathematical phase graph

| Order | Phase | Consumes | Produces | Mathematical meaning | Publication-facing rule |
| ---: | --- | --- | --- | --- | --- |
| 0 | ImportNormalize | frontend `GrammarDef` | prepared `GrammarDef` | choose a grammar normal form before any automaton construction | no tokenizer, DWA, parser table, or runtime cache decisions here |
| 1 | BuildTokenizer | prepared grammar terminals | lexer/tokenizer DFA | recognizer for grammar terminals over bytes | grammar object only; no vocabulary quotienting |
| 2 | AnalyzeGrammar | prepared grammar | analyzed parser facts | grammar and stack-effect facts needed by later phases | parser implementation may vary; outputs are facts, not exposition |
| 3 | BuildGlrTable | analyzed grammar | parse table | current implementation witness for stack evolution | should remain behind compile boundary |
| 4 | BuildTerminalGrammarFacts | table + analyzed grammar | terminal coloring, disallowed follows | structural facts about which terminals can coexist in parser states | no vocabulary bytes here except later consumers |
| 5 | BuildTerminalDwa | tokenizer + vocab + terminal facts | Terminal DWA over terminal strings | maps complete terminal sequences to `(lexer-state, token)` masks | name must stay Terminal DWA, not generic DWA or PM artifact |
| 6 | BuildScanRelation | tokenizer + vocab | scan relation / CanMatch | handles byte fragments that may end in a partial terminal | must not reuse Terminal-DWA equivalence as proof |
| 7 | BuildTemplates | parser stack-effect facts | template DFAs | parser-stack effect recognizers used by commit and Parser DWA | not LR-specific in wording |
| 8 | BuildParserDwa | table + templates + Terminal DWA | Parser DWA | maps parser stack prefixes to lexer-state/token masks | depends on stack-effect recognizers, not frontend parsing details |
| 9 | ReconcileArtifact | Terminal DWA, Parser DWA, CanMatch | shared internal coordinate system | proves all masks speak the same internal token/state language | the only place artifact ID spaces merge |
| 10 | FinalizeRuntime | reconciled artifacts + vocab | `Constraint` | package mathematical artifacts into runtime caches | only phase allowed to know `Constraint` field layout |

The key mathematical idea is that the phase graph is not a mere engineering pipeline.  It is a proof decomposition.  Each phase establishes a fact that later phases are allowed to assume.  The important obligations are:

1. **Tokenizer phase obligation.** The tokenizer recognizes grammar terminals over bytes.  It is independent of the LLM vocabulary.  The vocabulary only appears when we ask which token byte strings interact with the tokenizer.
2. **Terminal DWA obligation.** The Terminal DWA is a weighted automaton over complete grammar-terminal sequences.  Its weights denote the lexer-state/token pairs whose token bytes emit exactly that complete sequence.
3. **Scan relation obligation.** The scan relation is not the Terminal DWA with another name.  It records what happens when a token byte fragment may end in the middle of a terminal match.  This is why CanMatch equivalence must not be assumed from Terminal-DWA equivalence.
4. **Template obligation.** Template DFAs are stack-effect recognizers.  They are allowed to be built from the current parser implementation, but their conceptual meaning is parser-abstract: they summarize stack effects, not LR-ness.
5. **Parser DWA obligation.** The Parser DWA maps stack prefixes to masks over lexer-state/token pairs by composing parser stack-effect information with Terminal-DWA weights.
6. **Reconciliation obligation.** All weighted artifacts must eventually use one internal coordinate system.  Before reconciliation, a small internal token ID in one artifact may not denote the same original token set as the same small ID in another artifact.  After reconciliation, Parser-DWA weights and CanMatch weights must speak the same language.
7. **Finalization obligation.** Runtime caches are derived data.  They must not influence the mathematical meaning of the compiled object.

## 4. Exact phase-by-phase work already applied

### 4.1 ImportNormalize

The normalizer is still `prepare_grammar_transforms_only`.  It remains in `compiler::grammar::transforms` because this chunk is not the grammar-IR cleanup chunk.  The orchestrator calls it once in `compile_owned` or `compile_owned_profiled` and then passes a prepared grammar to the expensive phase graph.

Do not add tokenizer logic here.  Do not read vocabulary bytes here.  Do not build parser tables here.  This phase should stay about grammar normal form only.

### 4.2 BuildTokenizer / AnalyzeGrammar / BuildGlrTable / BuildTerminalGrammarFacts

These are grouped in `src/compile/pipeline/analysis.rs` because the current implementation benefits from computing tokenizer and parser facts in one phase group.  The grouping is an implementation convenience, but the output type `GrammarAnalysisOutput` makes the conceptual components explicit:

- `tokenizer`
- `analyzed_grammar`
- `table`
- `terminal_coloring`
- `disallowed_follows`

Future refactors can split `analysis.rs` further without changing downstream code, because downstream phases already depend on the typed output rather than on a huge function body.

### 4.3 BuildTerminalDwa / BuildScanRelation

These are in `src/compile/pipeline/terminal_scan.rs`.  They share the `TerminalScanSupport` precomputation:

- terminal classification cache,
- flat tokenizer transition table,
- global max-length state map.

They still produce distinct outputs.  This is not just naming pedantry.  The Terminal DWA and scan relation have different denotations.  One is about completed terminal sequences; the other is about boundary states and possible completions after partial scanning.  In other words, one is a relation over completed terminal strings and one is a relation over byte-fragment scans.

### 4.4 BuildTemplates

This is in `src/compile/pipeline/templates.rs`.  It builds two things:

- `Templates`, used by Parser-DWA construction,
- `TemplateDfasByTerminal`, used by runtime Commit.

This split matters because the Parser-DWA construction consumes the abstract template recognizers, while runtime Commit consumes specialized split DFAs.  They come from one characterization step, but they should not be treated as one opaque implementation blob.

### 4.5 BuildParserDwa / ReconcileArtifact

These are in `src/compile/pipeline/reconcile.rs`.  This is the densest phase because it preserves the old optimization modes.  The publication-facing interpretation is simple even though the code has several branches:

- choose whether Terminal DWA and CanMatch are reconciled before Parser DWA,
- choose whether the Terminal-DWA/CanMatch joint space is compacted,
- build Parser DWA over the chosen internal map,
- optionally reconcile and compact Parser DWA with CanMatch,
- compute size counters for the profile.

The important thing is that this code is no longer hidden in the same file as tokenizer construction and final `Constraint` field initialization.  It is now visibly the coordinate-system phase.

### 4.6 FinalizeRuntime

This is in `src/compile/pipeline/finalize.rs`.  It is the only compile phase that knows the full field list of runtime `Constraint`.  This is a deliberate boundary.  Earlier phases should not know about dense caches, sparse caches, word-group masks, heavy-token caches, or final mask mappings.  Those are runtime engineering details.

## 5. Symbol move table

| Old symbol / responsibility | Old location | New location | Reason | Notes for later chunks |
| --- | --- | --- | --- | --- |
| `compile_owned` | `src/compiler/pipeline.rs` via `compiler::compile` | `src/compile/pipeline/mod.rs` | the orchestration is a compile-object concern, not GLR internals | compatibility shim remains in `compiler::compile` |
| `compile_owned_profiled` | `src/compiler/pipeline.rs` | `src/compile/pipeline/mod.rs` | profile is a report on the phase graph | later can return a richer report type |
| `CompilePhaseProfile` | `src/compiler/pipeline.rs` | `src/compile/profiling.rs` | report shape is independent of orchestration implementation | still intentionally field-compatible |
| `emit_compile_profile_summary` | `src/compiler/pipeline.rs` | `src/compile/profiling.rs` | profile emission is a side effect; side effects are not phase logic | supports sink abstraction |
| `env_flag_enabled` | `src/compiler/pipeline.rs` | `src/compile/options.rs` | option resolution must be centralized | public `CompileOptions` can eventually replace env vars |
| `env_flag_enabled_by_default` | `src/compiler/pipeline.rs` | `src/compile/options.rs` | same | used for pre-reconcile CanMatch compaction |
| `DwaCanMatchMode` | `src/compiler/pipeline.rs` | `src/compile/options.rs` | this is a compile strategy decision, not a local variable | should eventually be an enum field on internal options |
| `compile_thread_count` | `src/compiler/pipeline.rs` | `src/compile/options.rs` | environment-backed compile decision | thread-pool construction moved separately |
| `run_with_compile_thread_pool` | `src/compiler/pipeline.rs` | `src/compile/thread_pool.rs` | execution policy is not phase math | keeps optional private rayon pool |
| `build_tokenizer` | `src/compiler/pipeline.rs` | `src/compile/tokenizer.rs` | tokenizer is an explicit grammar object | now documented as vocab-independent |
| `build_tokenizer_from_exprs` | `src/compiler/pipeline.rs` | `src/compile/tokenizer.rs` | same | kept crate-private |
| `terminal_expr` | `src/compiler/pipeline.rs` | `src/compile/tokenizer.rs` | terminal lowering belongs next to tokenizer construction | remains private |
| `compute_disallowed_follows` | `src/compiler/pipeline.rs` | `src/compile/pipeline/analysis.rs` | derived grammar/table fact | may later move to `compile/parser_facts` |
| interned-range accounting helpers | `src/compiler/pipeline.rs` | `src/compile/pipeline/counts.rs` | reconciliation accounting is independent utility | later can become artifact-size report module |
| `finalize_constraint` | `src/compiler/pipeline.rs` | `src/compile/pipeline/finalize.rs` | runtime layout knowledge should be isolated | finalization owns cache rebuild |
| template profile printing | inline in pipeline | `src/compile/profiling.rs` | profile side-effect centralized | line format preserved |
| old `compiler::pipeline` module | `src/compiler/pipeline.rs` full implementation | tiny compatibility shim / orphan path | prevents old file from continuing as conceptual center | remove after downstream imports are cleaned |

## 6. How to review this chunk

Review in this order:

1. Open `src/compile/pipeline/mod.rs` and verify it is readable as a phase graph.
2. Open `src/compile/pipeline/context.rs` and verify each intermediate object has a clear mathematical role.
3. Open `src/compile/options.rs` and verify no compile-pipeline module asks `std::env` directly.
4. Open `src/compile/profiling.rs` and verify profile line formats are centralized.
5. Open `src/compile/pipeline/reconcile.rs` and verify the old reconciliation cases are still represented.
6. Open `src/compile/pipeline/finalize.rs` and verify only finalization knows the `Constraint` field layout.
7. Search for `src/compiler/pipeline.rs` and verify it is no longer the implementation center.

## 7. What this chunk intentionally does not finish

This chunk does not clean up all environment variables throughout Terminal-DWA internals.  It moves the compile pipeline's own option decisions.  Later chunks should move terminal-DWA partition choices, pair-partition strategy, max-length options, and parser-DWA profiling flags into typed options.

This chunk does not make the public `CompileOptions` fully functional.  It prepares the internal seams where that can happen.

This chunk does not re-optimize parallel execution between Terminal-DWA/scan construction and template construction.  The old code mutated one shared profile across parallel branches.  The new code prioritizes phase clarity.  A later performance-preservation chunk should make phase functions return local profile deltas so that safe parallel execution can be restored without borrowing the same mutable profile twice.

This chunk does not compile or run tests.  That is deliberate: the current instruction is to get the shape right first, then work through compile errors later.
