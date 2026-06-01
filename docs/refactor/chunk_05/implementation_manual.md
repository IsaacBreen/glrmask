# Chunk 05 implementation manual: Parser DWA as a first-class subsystem

This document is intentionally self-contained.  It explains exactly what changed,
why it changed, how to review it, and how to continue from it without needing to
read the previous planning zip.

## Goal

Chunk 05 promotes Parser-DWA construction from a single monolithic file into a
subsystem whose file boundaries correspond to the mathematical construction in
the paper.

Before this chunk, almost every Parser-DWA concern lived in one `builder.rs`:

- Terminal-DWA transition grouping;
- bundle interning;
- productivity analysis;
- parser-NWA composition;
- parser stack-effect template splicing;
- local epsilon closure;
- weighted determinization;
- default-edge semantics;
- final-weight subtraction;
- minimization policy;
- profile printing;
- public build entrypoint.

That file was difficult to reason about because the reader had to infer which
parts were denotational and which parts were representation optimizations.  This
chunk makes those concepts explicit.

## The key mathematical object

The object constructed here is the Parser DWA:

```text
[[PDWA]](rho)_(q,v) = 1  iff  rho ∈ E_(q,v)
```

This says that after reading a parser stack prefix `rho`, the Parser DWA returns
the set of lexer-state/token pairs `(q,v)` for which that stack prefix is a valid
parser-side witness for the terminal strings that the lexer can scan from `q`
through token `v`.

A less formal but implementation-useful reading is:

> The Terminal DWA knows which terminal sequences a token can scan.  Template
> automata know which parser-stack prefixes can realize each terminal.  The
> Parser DWA composes those two objects into a deterministic runtime query over
> stack prefixes.

## Files after this chunk

`src/compile/parser_dwa/mod.rs`
: The module boundary and denotation.  This is the conceptual table of contents.

`src/compile/parser_dwa/builder.rs`
: The high-level phase graph.  It should remain short.  It is allowed to call
  other phases but should not implement them inline.

`src/compile/parser_dwa/terminal_projection.rs`
: Converts Terminal-DWA states into compact summaries: final weight, outgoing
  terminal bundles, and productivity.  This file lives on the Terminal-DWA side
  of the composition.

`src/compile/parser_dwa/compose_nwa.rs`
: Converts the Terminal-DWA summaries plus terminal templates into a weighted
  NWA over parser stack-state labels.  This is the categorical/compositional
  heart of the construction.

`src/compile/parser_dwa/determinize.rs`
: Owns weighted subset construction.  There are two distinct subset
  constructions; they now live together because they share representation
  machinery.

`src/compile/parser_dwa/optimize.rs`
: Owns semantics-preserving rewrites on the intermediate Parser DWA:
  default-edge normalization and final-weight subtraction.

`src/compile/parser_dwa/options.rs`
: Owns policy.  Policy is not denotation.  The environment switch that controls
  minimization no longer sits in the builder.

`src/compile/parser_dwa/profiling.rs`
: Owns all profile records and profile-line emission.  The rest of the subsystem
  records facts; it does not decide how those facts are printed.

`src/compile/parser_dwa/types.rs`
: Owns internal data carriers such as `TerminalBundle`, `StateSummary`, and
  `DetermininizedDwaWithSupports`.

`src/compile/parser_dwa/labels.rs`
: Owns the interpretation of raw automaton labels as parser-state ids.

## Exact mechanical changes

1. The old monolithic `builder.rs` was split into nine files.
2. `mod.rs` was expanded from a short comment into a denotational module header.
3. A new named input struct, `ParserDwaBuildInputs`, was added.
4. A new named output struct, `ParserDwaBuildOutput`, was added.
5. The new preferred entrypoint is
   `build_parser_dwa_from_terminal_dwa_with_templates(ParserDwaBuildInputs)`.
6. The old entrypoint,
   `build_parser_dwa_from_terminal_dwa_with_precomputed_templates`, remains as a
   compatibility wrapper so the compile pipeline does not have to change in the
   same chunk.
7. Direct `eprintln!` calls were removed from builder/composition code and moved
   into `profiling.rs`.
8. Minimization policy moved to `ParserDwaOptions`.
9. Parser-state label interpretation moved to `labels.rs`.
10. The module now has a local `README.md` explaining reading order and boundary
    rules.

## How to review this chunk manually

Read files in this order:

1. `src/compile/parser_dwa/mod.rs`.
2. `src/compile/parser_dwa/builder.rs`.
3. `src/compile/parser_dwa/types.rs`.
4. `src/compile/parser_dwa/terminal_projection.rs`.
5. `src/compile/parser_dwa/compose_nwa.rs`.
6. `src/compile/parser_dwa/determinize.rs`.
7. `src/compile/parser_dwa/optimize.rs`.
8. `src/compile/parser_dwa/profiling.rs`.
9. `src/compile/parser_dwa/options.rs` and `labels.rs`.

When reviewing, do not start by asking whether the crate compiles.  Start by
asking whether each file now has a single mathematical reason to exist.  Compile
errors from moves/imports can be fixed later; conceptual mixing is what this
chunk is trying to remove.

## Definition of done for this chunk

- `builder.rs` is under 400 lines.
- `mod.rs` states the Parser-DWA denotation explicitly.
- `builder.rs` no longer contains terminal-bundle grouping, epsilon closure,
  default optimization, or profile print strings.
- `profiling.rs` is the only file in `src/compile/parser_dwa/` containing
  `eprintln!`.
- The old compile pipeline can still call the compatibility wrapper by the old
  name.
- The subsystem has local docs and review artifacts.

## What this chunk deliberately does not do

- It does not move GLR table construction out of `compiler::glr`.
- It does not redesign terminal templates themselves.
- It does not rewrite the weighted automata crate.
- It does not touch runtime Mask traversal.
- It does not touch Commit/template-DFA acceleration.
- It does not compile or run tests.

## Why this is the right chunk after Terminal DWA cleanup

The Terminal DWA is mathematically upstream: it supplies the weighted terminal
language.  Once its construction is named and isolated, the Parser DWA can be
shown as a composition rather than as an opaque parser-stage optimization.

This chunk therefore makes the paper narrative visible in the source tree:

```text
terminal_dwa  +  template stack-effect recognizers  ->  parser_dwa
```

That is the conceptual bridge between compile-time lexer/parser analysis and
runtime mask generation.


## Additional deep split: determinization directory

After the initial split, `determinize.rs` was still the largest file and still
contained multiple mathematically distinct operations.  This chunk therefore
continues the split one level deeper:

- `determinize/mod.rs` names the determinization subphases.
- `determinize/outgoing.rs` computes possible outgoing parser-state labels from
  support sets.
- `determinize/epsilon.rs` owns local weighted epsilon closure.
- `determinize/support.rs` owns the first support-preserving subset
  construction.
- `determinize/fallback.rs` owns the second fallback/default subset
  construction.

This keeps every source file below 400 lines and makes the two subset
constructions visibly different objects rather than two long functions in a
shared helper file.
