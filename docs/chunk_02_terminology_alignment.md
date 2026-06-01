# Chunk 02 changes: paper terminology alignment

This document records the exact source-tree transformation performed in chunk
02.  It is intentionally self-contained so the change can be reviewed without
reading the planning package.

## Purpose

The crate had begun to expose the same conceptual objects as the paper, but the
implementation still mixed paper names with historical names:

- `id_map_and_terminal_dwa` described a storage bundle, not the Terminal DWA.
- `constraint_possible_matches` and `possible_matches` hid the distinction
  between byte scanning, completed terminal sequences, and partial lexer-state
  completion.
- `l1` and `l2p` were local abbreviations with no mathematical meaning to a new
  reader.
- `mask_game_*` leaked benchmark-harness language into public diagnostics.
- `pm` and `pmv` appeared in profile/config names even after the surrounding
  code had a better vocabulary.

Chunk 02 makes the code tree read like the paper.

## File moves

| Old path | New path |
| --- | --- |
| `src/compiler/stages/id_map_and_terminal_dwa/` | `src/compile/terminal_dwa/` |
| `src/compiler/stages/id_map_and_terminal_dwa/l1/` | `src/compile/terminal_dwa/direct_partition/` |
| `src/compiler/stages/id_map_and_terminal_dwa/l2p/` | `src/compile/terminal_dwa/pair_partition/` |
| `src/compiler/stages/parser_dwa.rs` | `src/compile/parser_dwa/builder.rs` |
| `src/compiler/constraint_possible_matches/` | `src/compile/scan_relation/` |
| `src/compiler/possible_matches.rs` | `src/compile/scan_relation/terminal_sequences.rs` |
| `src/compiler/pm_profile.rs` | `src/compile/scan_relation/profile.rs` |

## New modules

```text
src/compile/mod.rs
src/compile/parser_dwa/mod.rs
src/compile/parser_dwa/builder.rs
src/compile/scan_relation/mod.rs
src/compile/scan_relation/collector.rs
src/compile/scan_relation/profile.rs
src/compile/scan_relation/terminal_sequences.rs
src/compile/terminal_dwa/mod.rs
src/compile/terminal_dwa/direct_partition/
src/compile/terminal_dwa/pair_partition/
```

`src/compiler/stages/` now retains only generic compiler infrastructure that has
not yet been promoted: mapped artifacts, template machinery, negative-code
resolution, and equivalence map types.

## Symbol renames

| Old symbol family | New symbol family |
| --- | --- |
| `PossibleMatchesComputer` | `CanMatchComputer` |
| `PossibleMatchesProfile` | `CanMatchProfile` |
| `PossibleMatchVocabMap` | `ScanRelationVocabMap` |
| `ConstraintPossibleMatchesConfig` | `ScanRelationConfig` |
| `ConstraintPossibleMatchesProfile` | `ScanRelationProfile` |
| `ConstraintPossibleMatchesComputation` | `ScanRelationComputation` |
| `RuntimePossibleMatchesByTerminal` | `RuntimeCanMatchByTerminal` |
| `compute_constraint_possible_matches_for_vocab` | `compute_scan_relation_for_vocab` |
| `prepare_vocab_for_possible_matches` | `prepare_vocab_for_scan_relation` |
| `build_l1_*` | `build_direct_partition_*` |
| `build_l2p_*` | `build_pair_partition_*` |
| `L2pPartition*` | `PairPartition*` |
| `mask_game_*` | removed; use internal-token mapping names |

## Runtime comment fixes

`runtime/mask` now states the central Mask operation directly:

> Mask walks the active parser stacks through the Parser DWA and combines the
> encountered transition and final weights into a vocabulary mask.

`runtime/commit` now states the central Commit operation directly:

> Commit consumes token bytes or raw bytes, lets the tokenizer emit every
> completed grammar terminal boundary, and advances the active parser stacks
> through the GLR transition relation for those completed terminals.

These comments are deliberately mathematical.  They describe the semantic
operation before the dense/sparse/template fast paths.

## Environment/config terminology

The code strings were updated to prefer:

```text
GLRMASK_DWA_CAN_MATCH_MODE
GLRMASK_PARSER_DWA_CAN_MATCH_COMPACTION
GLRMASK_SCAN_RELATION_*
GLRMASK_PAIR_PARTITION_*
GLRMASK_DIRECT_PARTITION_*
GLRMASK_FORCE_ALL_PAIR_PARTITION
```

A later configuration chunk may add compatibility aliases.  Chunk 02 chooses the
publication names as the source of truth.

## Deferred work

This chunk intentionally does not finish every architectural separation:

1. `src/compile/scan_relation/mod.rs` remains large and should be split later.
2. `src/compile/terminal_dwa/direct_partition/mod.rs` remains very large and
   should be split later.
3. `src/compile/terminal_dwa/pair_partition/mod.rs` remains very large and
   should be split later.
4. Template DFA code still lives under `src/compiler/stages/templates/` because
   the template subsystem deserves a dedicated later chunk.
5. GLR parser/table machinery still lives under `src/compiler/glr/`; moving it
   should happen in the parser cleanup chunk, not inside terminology alignment.

## Mechanical acceptance checks used for this chunk

```bash
find src -path '*id_map_and_terminal_dwa*' -o -path '*constraint_possible_matches*' -o -path '*l2p*' -o -path '*l1*'
rg 'id_map_and_terminal_dwa|constraint_possible_matches|mask_game|possible_matches|PossibleMatches|possible_match|pmv|PMV|L2P|l2p' src bindings README.md Cargo.toml
```

The source grep is empty for the retired names.  Documentation may still mention
old names only inside migration tables.  Remaining occurrences of words like
`possible_future` are lexer automaton terms, not the retired CanMatch subsystem.
