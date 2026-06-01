# Terminology and code map

This file is the canonical publication terminology map for the implementation.
The goal is that a reader can move from the paper to the source tree without
learning a second private vocabulary.  If a name in the code does not describe a
paper object, it should either be a low-level representation detail or it should
be renamed.

## Naming rule

Use names for **denotations**, not for historical implementation accidents.

A denotation is the mathematical object computed by a module.  Examples:
Terminal DWA, Parser DWA, Scan relation, CanMatch relation, Mask, Commit,
Template DFA.  Historical names such as `id_map_and_terminal_dwa`, `l1`, `l2p`,
`possible_matches`, `pmv`, and `mask_game` hide the denotation and should not
appear in publication-facing source paths or API names.

## Core table

| Paper term | Meaning | Primary code home | Important code names | Forbidden / retired names |
| --- | --- | --- | --- | --- |
| `Scan(q, b)` | Branching lexer scan relation.  Scanning byte fragment `b` from lexer state `q` returns completed terminal sequences plus the lexer state left at the fragment boundary. | `src/compile/scan_relation/` and `src/runtime/commit/tokenizer_scan.rs` | `scan_relation`, `InitialCommitScan`, tokenizer scan helpers | `constraint_possible_matches` as a module name |
| `CanMatch(q')` | Terminals that can complete a partial lexer state `q'`.  This is needed when a token or byte fragment ends inside a terminal match. | `src/compile/scan_relation/`, especially `terminal_sequences.rs` and `collector.rs` | `CanMatchComputer`, `IntervalCanMatchMap`, `RuntimeCanMatchByTerminal` | `PossibleMatchesComputer`, `possible_matches`, `pm`, `pmv` |
| Terminal DWA | Weighted deterministic automaton over completed grammar-terminal sequences.  Its weights are masks over lexer-state/token pairs. | `src/compile/terminal_dwa/` | `build_terminal_dwa_*`, `TerminalDwaPhaseProfile`, `TerminalColoring` | `id_map_and_terminal_dwa` |
| Direct partition | Terminal-DWA construction path for terminals whose relevant token paths have length at most one. | `src/compile/terminal_dwa/direct_partition/` | `build_direct_partition_terminal_dwa`, `DirectPartition*` internal helper structs | `l1` |
| Pair partition | Terminal-DWA construction path for multi-step terminal paths, where a token byte string can cross more than one terminal boundary. | `src/compile/terminal_dwa/pair_partition/` | `build_pair_partition_terminal_dwa`, `PairPartition*` cost/config names | `l2p`, `L2P` |
| Parser DWA | Weighted deterministic automaton over parser stack prefixes.  Runtime Mask walks active stacks through it. | `src/compile/parser_dwa/` | `build_parser_dwa_from_terminal_dwa_with_precomputed_templates` | `compiler/stages/parser_dwa.rs` |
| Template DFA | Deterministic stack-effect recognizer used to accelerate commit-time parser advancement. | `src/compiler/stages/templates/` for now; target later is `src/compile/template_dfa/` | `Templates`, `CommitTemplateDfas`, `template_advance` | Generic `templates` should eventually be promoted. |
| Mask | Runtime operation that evaluates active parser stacks against the Parser DWA and materializes a vocabulary mask. | `src/runtime/mask/` | `mask`, `fill_mask`, `fill_mask_and_internal_token_ids`, `MaskProfile` | `mask_game_*` |
| Commit | Runtime operation that consumes bytes, emits completed terminals, and advances parser stacks. | `src/runtime/commit/` | `commit_token`, `commit_bytes`, `advance_parser_stacks`, `CommitProfile` | Any comment suggesting commit is merely a token operation. |
| Internal token quotient | Runtime token-space quotient induced by Terminal-DWA, Parser-DWA, and CanMatch reconciliation. | `src/runtime/mask_mapping.rs`, `src/runtime/artifact.rs` | `internal_to_original_token_ids`, `original_to_internal_token_ids` | `mask_game_mapping` |

## Mathematical invariant behind the rename

The compile pipeline computes three different objects that can all be described
as automata or masks, but they are not interchangeable:

1. **Terminal DWA** consumes sequences of completed grammar terminals.  It
   answers: for this terminal sequence, which `(lexer state, token)` pairs can
   produce it?
2. **Scan / CanMatch** consumes byte fragments and lexer states.  It answers:
   while scanning this token or byte fragment, which terminals were completed,
   and if we ended in a partial lexer state, which terminals can complete later?
3. **Parser DWA** consumes parser stack prefixes.  It answers: for this active
   parser stack prefix, which `(lexer state, token)` pairs are allowed by the
   parser language?

The old names blurred these distinctions.  In particular, `possible_matches`
sounded like a generic cache and `id_map_and_terminal_dwa` sounded like a bundle
of storage details.  The publication names identify the mathematical relation
first and let representation details remain local.

## Code-path map after chunk 02

```text
src/
  compile/
    mod.rs
    terminal_dwa/
      mod.rs
      classify.rs
      direct_partition/
      pair_partition/
      grammar_helpers.rs
      merge.rs
      partition.rs
      types.rs
    scan_relation/
      mod.rs
      collector.rs
      terminal_sequences.rs
      profile.rs
    parser_dwa/
      mod.rs
      builder.rs
  compiler/
    glr/
    grammar/
    stages/
      equiv_types.rs
      mapped_artifact/
      templates/
      resolve_negatives.rs
  runtime/
    mask/
    commit/
```

`src/compiler/` still owns the older GLR and grammar-analysis machinery.  This
is intentional for this chunk: the next structural passes can move parser and
phase-graph material separately.  Chunk 02 only promotes the paper's compiled
automata and scan relation to first-class homes.

## Environment-variable vocabulary

Publication names should also appear in diagnostics and environment variables.
The following names replaced benchmark-era abbreviations in source strings:

| Old form | New form |
| --- | --- |
| `GLRMASK_DWA_PM_MODE` | `GLRMASK_DWA_CAN_MATCH_MODE` |
| `GLRMASK_PARSER_DWA_PM_COMPACTION` | `GLRMASK_PARSER_DWA_CAN_MATCH_COMPACTION` |
| `GLRMASK_PM_*` | `GLRMASK_SCAN_RELATION_*` |
| `GLRMASK_L2P_*` | `GLRMASK_PAIR_PARTITION_*` |
| `GLRMASK_FORCE_ALL_L2P` | `GLRMASK_FORCE_ALL_PAIR_PARTITION` |
| `GLRMASK_COMPACT_POSSIBLE_MATCHES_BEFORE_RECONCILE` | `GLRMASK_COMPACT_CAN_MATCH_BEFORE_RECONCILE` |

A later configuration chunk should decide whether to preserve legacy aliases for
private benchmarks.  The publication source should not teach new readers the old
abbreviations.

## Comment style rules

Use these phrases consistently:

- “Terminal DWA” rather than “terminal automaton stage” when the object is the
  weighted terminal-sequence automaton.
- “Parser DWA” rather than “parser DWA stage” when the object is the weighted
  stack-prefix automaton.
- “Scan relation” for byte-scanning semantics.
- “CanMatch” for terminals that can complete a partial lexer state.
- “direct partition” for the single-step Terminal-DWA construction path.
- “pair partition” for the multi-step Terminal-DWA construction path.
- “internal token quotient” or “internal token mapping” for the final compact
  token space.

Avoid these phrases in new code:

- `mask_game`
- `possible_matches`
- `pm` / `pmv`
- `l1` / `l2p`
- `id_map_and_terminal_dwa`
- “token loop” when the surrounding text means the LLM generation loop.

## Acceptance checks

A terminology-aligned tree should satisfy:

```bash
rg 'id_map_and_terminal_dwa|constraint_possible_matches|mask_game|possible_matches|PossibleMatches|pmv|PMV|L2P|l2p' src bindings README.md Cargo.toml
find src -path '*id_map_and_terminal_dwa*' -o -path '*constraint_possible_matches*' -o -path '*l2p*' -o -path '*l1*'
```

The grep intentionally excludes `docs/` because this terminology file and the chunk
change log mention retired names in migration tables.  Source should not contain
the retired module/API names listed above.
The second command should be empty for publication-facing compile paths.


## GLR parser-domain vocabulary after chunk 09

`parser::glr` is the canonical home for GLR-specific parser machinery.  It is
not a paper object like Terminal DWA or Parser DWA, but it supplies the stack
effect recognizers and concrete stack advancement semantics that those paper
objects rely on.

| Term | Code home | Meaning | Retired wording |
| --- | --- | --- | --- |
| Grammar analysis | `src/parser/glr/analysis.rs` plus `src/parser/glr/analysis/` fragments | Normalization and nullable/FIRST/FOLLOW computation for flat grammars. | `compiler::glr::analysis` as the primary import path |
| GLR table | `src/parser/glr/table/` | Optimized parser transition table over states, terminals, gotos, and stack-effect actions. | Treating the table as compile-only compiler state |
| Parser stack advance | `src/parser/glr/advance/` | Exact one-terminal transition relation over a persistent parser GSS. | `compiler::glr::parser`, `stack_may_advance_on` |
| Stack applicability | `stack_can_advance_on`, `stack_can_advance_on_any` | Exact predicate for whether a concrete GSS can advance on a terminal or set of terminals. | `may_advance` for exact predicates |

The old `compiler::glr` path remains a hidden compatibility shim.  New code
should import `crate::parser::glr::*`.
