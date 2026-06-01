# File-by-file change ledger for Chunk 05

## `builder.rs`

Lines after chunk: **218**.

Opening doc/comment:

```rust
//! High-level Parser-DWA build entrypoints.
//!
//! This file deliberately contains only phase ordering.  It should read like
//! the proof outline for the construction:
//!
//! 1. Compose the Terminal DWA with parser stack-effect templates to obtain a
//!    parser NWA.
//! 2. Resolve temporary negative labels produced by template construction.
//! 3. Determinize over parser-state labels while retaining support sets.
//! 4. Derive legal fallback/default domains from those supports.
//! 5. Normalize defaults and final weights.
//! 6. Re-determinize fallback semantics.
//! 7. Optionally minimize the resulting weighted DWA.
```

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `compose_nwa.rs`

Lines after chunk: **367**.

Opening doc/comment:

```rust
//! Composition of Terminal-DWA branches with parser stack-effect templates.
//!
//! This is the mathematical core of Parser-DWA construction.  Each productive
//! Terminal-DWA state becomes a continuation state in a parser NWA.  Each
//! outgoing Terminal-DWA branch is replaced by the template automaton(s) for
//! the terminals in that branch, weighted by the Terminal-DWA edge masks, with
//! template finals redirected to the branch continuation.
```

Local symbols:

- line 27: `fn dwa_to_nwa(dwa: &DWA) -> NWA {`
- line 48: `fn compute_productive_terminal_states(summaries: &StateSummaries) -> Vec<bool> {`
- line 90: `fn append_weighted_template_redirecting_finals(`
- line 121: `fn append_bundle_redirecting_finals(`
- line 142: `fn append_branch_fragment(`

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `determinize/epsilon.rs`

Lines after chunk: **73**.

Opening doc/comment:

```rust
//! Local weighted epsilon closure.
//!
//! The closure map is `NWA state -> accumulated pair mask`.  Multiple epsilon
//! paths to the same state union their weights, and newly added pairs cause the
//! target state to be processed again.
```

Local symbols:

- line 14: `pub(super) fn local_epsilon_closure(`

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `determinize/fallback.rs`

Lines after chunk: **211**.

Opening doc/comment:

```rust
//! Fallback/default determinization.
//!
//! After default-edge optimization, the automaton has fallback semantics that
//! are convenient for construction but inconvenient for runtime walking.  This
//! pass makes those semantics explicit in an ordinary deterministic weighted
//! automaton.
```

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `determinize/mod.rs`

Lines after chunk: **18**.

Opening doc/comment:

```rust
//! Weighted determinization phases for Parser-DWA construction.
//!
//! This submodule is split by mathematical role rather than by helper size:
//!
//! - `outgoing`: recover possible parser-state labels from NWA supports;
//! - `epsilon`: local weighted epsilon closure;
//! - `support`: first determinization, preserving source-NWA supports;
//! - `fallback`: second determinization, making default fallback semantics
//!   explicit.
```

Local symbols:

- line 11: `mod epsilon;`
- line 12: `mod fallback;`
- line 13: `mod outgoing;`
- line 14: `mod support;`

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `determinize/outgoing.rs`

Lines after chunk: **99**.

Opening doc/comment:

```rust
//! Possible outgoing parser-state labels.
//!
//! The first determinization records source-NWA supports for each DWA state.
//! This file turns those supports into the set of parser-state labels that may
//! need default fallback coverage.
```

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `determinize/support.rs`

Lines after chunk: **322**.

Opening doc/comment:

```rust
//! Support-preserving weighted subset construction.
//!
//! This is the first determinization pass.  Besides producing a DWA, it keeps
//! for each determinized state the set of source-NWA states that contributed
//! to that state.  Those supports are required by fallback/default handling.
```

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `labels.rs`

Lines after chunk: **14**.

Opening doc/comment:

```rust
//! Parser-state labels.
//!
//! Parser-DWA input symbols are parser stack states.  The underlying automata
//! package uses signed `i32` labels because it also has default and internal
//! negative labels.  This helper is the one place where a raw automaton label
//! is interpreted as a parser-state id.
```

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `mod.rs`

Lines after chunk: **64**.

Opening doc/comment:

```rust
//! Parser DWA construction.
//!
//! # Denotation
//!
//! The Parser DWA is the compile-time weighted automaton whose input word is a
//! parser-stack prefix `rho` and whose output weight is the set of lexer-state /
//! vocabulary-token pairs accepted after that stack prefix.
//!
//! In the paper's notation, this module builds the automaton `PDWA` satisfying
//!
//! ```text
//! [[PDWA]](rho)_{q,v} = 1  iff  rho ∈ E_{q,v}.
//! ```
//!
//! The important point is that the Parser DWA does not know, or expose, the
//! internal details of the parser algorithm.  It only consumes stack-effect
//! recognizers.  Today those recognizers are produced from a GLR table; the
//! construction is intentionally organized so that a future parser backend can
//! provide the same recognizer family without changing Mask or Commit.
//!
//! # Construction shape
//!
//! The construction is a pullback/composition of two finite objects:
//!
//! 1. The Terminal DWA, which maps terminal strings to lexer-state/token-pair
//!    weights.
//! 2. Template DFAs/NWAs, one per terminal, which recognize the parser-stack
//!    prefixes that can realize that terminal's stack effect.
//!
//! For each Terminal-DWA transition bundle, we splice the corresponding
//! terminal templates in front of the continuation state of the target
//! Terminal-DWA state.  The resulting weighted NWA is then determinized over
//! parser-state labels to obtain the runtime Parser DWA.
//!
//! # File guide
//!
//! - `builder.rs`: public build entrypoints and high-level phase ordering.
//! - `compose_nwa.rs`: Terminal-DWA / template composition into a parser NWA.
//! - `terminal_projection.rs`: projection of Terminal-DWA states into terminal
//!   bundles and productive continuation summaries.
//! - `determinize.rs`: weighted subse
```

Local symbols:

- line 49: `pub(crate) mod builder;`
- line 50: `mod compose_nwa;`
- line 51: `mod determinize;`
- line 52: `mod labels;`
- line 53: `mod optimize;`
- line 54: `mod options;`
- line 55: `mod profiling;`
- line 56: `mod terminal_projection;`
- line 57: `mod types;`

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `optimize.rs`

Lines after chunk: **252**.

Opening doc/comment:

```rust
//! Parser-DWA normalization and size optimization.
//!
//! These transformations preserve the weighted language of the Parser DWA.
//! They only change representation: default edges absorb repeated positive
//! parser-state edges, final weights are lifted out of outgoing transitions,
//! and empty residual transitions are removed.
```

Local symbols:

- line 18: `fn union_final_weight(slot: &mut Option<Weight>, add: Weight) -> bool {`

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `options.rs`

Lines after chunk: **55**.

Opening doc/comment:

```rust
//! Parser-DWA construction policy.
//!
//! The mathematical denotation of the Parser DWA is independent of these
//! switches.  Options may choose a faster or smaller equivalent construction,
//! but they must not change which `(lexer_state, token)` pairs appear in
//! `[[PDWA]](rho)`.
```

Local symbols:

- line 32: `fn skip_parser_dwa_minimization_env_override() -> Option<bool> {`

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `profiling.rs`

Lines after chunk: **220**.

Opening doc/comment:

```rust
//! Parser-DWA profiling records and textual emission.
//!
//! Construction code records phase timings into structs.  This file is the only
//! Parser-DWA submodule allowed to print profile lines.  Keeping profile output
//! here prevents the mathematical construction from being interleaved with
//! logging mechanics.
```

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `terminal_projection.rs`

Lines after chunk: **157**.

Opening doc/comment:

```rust
//! Projection from Terminal-DWA graph structure to Parser-DWA continuation data.
//!
//! The Terminal DWA is a weighted automaton over terminal strings.  Parser-DWA
//! construction needs to ask a different question: from this Terminal-DWA
//! state, which groups of terminals all flow to the same continuation state,
//! and can those groups actually be accepted by parser stack-effect templates?
```

Local symbols:

- line 21: `fn group_terminal_edges_by_target(`
- line 47: `fn bundle_signature(bundle: &TerminalBundle) -> BundleSignature {`
- line 54: `fn terminal_template_has_acceptance(template: &NWA) -> bool {`
- line 58: `fn terminal_bundle_has_acceptance(bundle: &TerminalBundle, templates: &Templates) -> bool {`

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

## `types.rs`

Lines after chunk: **100**.

Opening doc/comment:

```rust
//! Local data carriers for Parser-DWA construction.
//!
//! These types are deliberately not exported from the crate.  They name the
//! intermediate mathematical objects used by the construction so that the
//! builder does not collapse into a long procedural script.
```

Review question: does this file own only the concept named by its filename? If not, move the extra concept in a later chunk.

