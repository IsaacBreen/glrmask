# Chunk 02 implementation manual: paper-terminology alignment

This document is intentionally exhaustive.  It is written as a hand-application
manual for this chunk, not merely as a summary of what changed.  The goal is that
someone who understands basic file editing, but not the whole crate, can inspect
or reproduce the chunk by following the sections below.

## 0. Scope of the chunk

Chunk 02 is a terminology-and-boundary chunk.  It does not try to optimize code,
change algorithms, add new parser theory, or prove runtime equivalence.  Its job
is to make the source tree say the same thing as the paper.

The target is:

1. Paper-level objects have paper-level homes.
2. Historical abbreviations disappear from source-facing names.
3. Runtime operations have names that match their denotation.
4. Compile-time relations are named as relations, not as caches or accidents of
   implementation.
5. Public and Python APIs stop exposing benchmark-era language.
6. The code tree becomes legible before deeper mechanical refactors start.

This chunk is deliberately broad but semantically shallow: it moves modules,
renames symbols, and adds explanatory documentation.  It does not compile-test the
result and it does not use compiler errors to steer design.

## 1. Mathematical target

The paper distinguishes several objects that the old file tree compressed
unnaturally:

- `Scan(q, b)`: scanning a byte fragment from lexer state `q`.
- `CanMatch(q')`: the set of terminals that can complete from a partial lexer
  state `q'`.
- Terminal DWA: weighted deterministic automaton over completed terminal
  sequences.
- Parser DWA: weighted deterministic automaton over parser stack prefixes.
- Mask: runtime operation evaluating active parser stacks against the Parser DWA.
- Commit: runtime operation consuming bytes/tokens and advancing parser stacks.
- Template DFA: precomputed stack-effect recognizer used to accelerate commit.

The old source tree blurred these objects:

- `id_map_and_terminal_dwa` mixed an object with one of its storage decisions.
- `possible_matches` sounded like a loose cache rather than the CanMatch side of
  the scan relation.
- `constraint_possible_matches` was too implementation-specific and too vague.
- `l1` and `l2p` were local performance-engineering abbreviations.
- `pm`, `pmv`, and `mask_game` leaked scratch/benchmark vocabulary into serious
  source names.

The new tree should be read as a denotational diagram:

```text
grammar + vocab
  │
  ├─ compile::terminal_dwa
  │    └─ builds the weighted automaton over completed terminal sequences
  │
  ├─ compile::scan_relation
  │    └─ builds CanMatch weights for incomplete lexer states / partial bytes
  │
  └─ compile::parser_dwa
       └─ builds the weighted automaton over parser stack prefixes

runtime::commit consumes bytes and updates parser state
runtime::mask   evaluates parser stacks and materializes token masks
```

The important design criterion is not that every module is already perfectly
small.  It is that every top-level name now points at a mathematical object.

## 2. File movement ledger

Apply these moves exactly.  Preserve file contents first, then apply imports and
renames.

| Old path | New path | Reason |
| --- | --- | --- |
| `src/compiler/stages/id_map_and_terminal_dwa/` | `src/compile/terminal_dwa/` | The module builds the Terminal DWA.  The id map is an implementation detail of that object. |
| `src/compiler/stages/id_map_and_terminal_dwa/l1/` | `src/compile/terminal_dwa/direct_partition/` | `l1` is not meaningful outside local history.  This path handles the direct/single-step partition. |
| `src/compiler/stages/id_map_and_terminal_dwa/l2p/` | `src/compile/terminal_dwa/pair_partition/` | `l2p` is not publication vocabulary.  This path handles pair/multi-step partitioning. |
| `src/compiler/stages/parser_dwa.rs` | `src/compile/parser_dwa/builder.rs` | Parser DWA is a named compiled object, not a generic stage file. |
| `src/compiler/constraint_possible_matches/` | `src/compile/scan_relation/` | The relation belongs beside the Terminal DWA and Parser DWA. |
| `src/compiler/possible_matches.rs` | `src/compile/scan_relation/terminal_sequences.rs` | The file computes terminal sequences that can complete from lexer states. |
| `src/compiler/pm_profile.rs` | `src/compile/scan_relation/profile.rs` | The profile belongs to scan-relation construction. |

After the moves, add these module files:

```text
src/compile/mod.rs
src/compile/parser_dwa/mod.rs
```

Then make `src/lib.rs` declare:

```rust
pub(crate) mod compile;
```

and remove old declarations from `src/compiler/mod.rs` and
`src/compiler/stages/mod.rs`.

## 3. New module contracts

### 3.1 `src/compile/mod.rs`

This file is not just a mechanical module list.  It is the reader's entry point
for the compile-time half of the paper.  Its module-level documentation must say:

- this module groups by denotation;
- `terminal_dwa` builds the Terminal DWA;
- `scan_relation` builds Scan/CanMatch artifacts;
- `parser_dwa` builds the Parser DWA;
- GLR table mechanics may still live in `compiler` during the transition;
- the key boundary is now paper-object code versus generic compiler plumbing.

Do not make this file re-export a broad facade.  Keep it crate-private for now.
The public facade is handled by `src/api/` from Chunk 01.

### 3.2 `src/compile/terminal_dwa/mod.rs`

This module owns Terminal DWA construction and the token-space quotient used by
that construction.  Its top comment should define the Terminal DWA as a weighted
automaton over completed grammar-terminal sequences.  It should also introduce
two sub-builders:

- `direct_partition`: direct/single-step terminal paths;
- `pair_partition`: multi-step terminal paths that may cross more than one
  terminal boundary.

The old phrase “id map and terminal DWA” should not appear in source code.  When
an id map is discussed, it should be discussed as a token quotient or internal id
map produced for a particular compiled object.

### 3.3 `src/compile/scan_relation/mod.rs`

This module owns the compile-time materialization of the CanMatch side of the
scan relation.  Its docs must explicitly warn that Terminal-DWA token equivalence
must not be reused for CanMatch equivalence.  That warning is mathematically
important:

- Terminal-DWA equivalence is equivalence under completed terminal sequences.
- CanMatch equivalence is equivalence under possible completions from partial
  lexer states.
- A token quotient correct for one relation can be unsound for the other.

Therefore names in this module should use `scan_relation` and `can_match`, not
`possible_matches`, `pm`, or `pmv`.

### 3.4 `src/compile/parser_dwa/mod.rs`

This module must be small for now.  It should define the Parser DWA as the
compile-time automaton over parser stack states/prefixes and re-export only the
builder function needed by the pipeline:

```rust
pub(crate) use builder::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
```

The detailed builder can remain in `builder.rs`.  Later chunks can split it into
language construction, determinization/minimization, and weight lifting.

## 4. Symbol rename ledger

These renames are part of the chunk.  The point is not cosmetic consistency; the
point is that the symbol name tells the reader what mathematical object it
belongs to.

| Old symbol or phrase | New symbol or phrase | Meaning |
| --- | --- | --- |
| `PossibleMatchesComputer` | `CanMatchComputer` | Computes possible terminal completions from lexer states. |
| `PossibleMatchesProfile` | `CanMatchProfile` | Profiles CanMatch computation, not a generic cache. |
| `PossibleMatchVocabMap` | `ScanRelationVocabMap` | Token quotient induced for scan-relation materialization. |
| `ConstraintPossibleMatchesConfig` | `ScanRelationConfig` | Configuration for scan-relation construction. |
| `ConstraintPossibleMatchesProfile` | `ScanRelationProfile` | Timing for scan-relation construction. |
| `ConstraintPossibleMatchesComputation` | `ScanRelationComputation` | Bundled result of scan-relation construction. |
| `RuntimePossibleMatchesByTerminal` | `RuntimeCanMatchByTerminal` | Runtime weights by terminal for CanMatch. |
| `compute_constraint_possible_matches_for_vocab` | `compute_scan_relation_for_vocab` | Builds scan-relation artifact for a vocab. |
| `compute_constraint_possible_matches` | `compute_scan_relation` | Main scan-relation builder. |
| `prepare_vocab_for_possible_matches` | `prepare_vocab_for_scan_relation` | Prepares ordered vocab/trie artifacts for scan relation. |
| `mapped_possible_matches` | `mapped_can_match` | Mapped runtime CanMatch artifact. |
| `build_l1_*` | `build_direct_partition_*` | Direct/single-step partition path. |
| `build_l2p_*` | `build_pair_partition_*` | Pair/multi-step partition path. |
| `L2pPartition*` | `PairPartition*` | Config/profile/types for pair partitioning. |
| `mask_game_*` | removed | Use internal-token mapping names instead. |

When applying a rename, update imports in the same commit/chunk.  Do not leave
compatibility aliases in source for internal names unless they are public API
stability shims.  For this chunk, the public benchmark-era `mask_game_*` shims
are removed intentionally because Chunk 01 already introduced the replacement
names and this branch is still pre-publication.

## 5. Environment-variable rename ledger

The same rule applies to environment variables.  Environment knobs are still too
numerous, but their names should at least point at the correct object.

| Old variable family | New variable family |
| --- | --- |
| `GLRMASK_L2P_*` | `GLRMASK_PAIR_PARTITION_*` |
| `GLRMASK_FORCE_ALL_L2P` | `GLRMASK_FORCE_ALL_PAIR_PARTITION` |
| `GLRMASK_PM_*` | `GLRMASK_SCAN_RELATION_*` |
| `GLRMASK_DWA_PM_MODE` | `GLRMASK_DWA_CAN_MATCH_MODE` |
| `GLRMASK_PARSER_DWA_PM_COMPACTION` | `GLRMASK_PARSER_DWA_CAN_MATCH_COMPACTION` |
| `GLRMASK_COMPACT_POSSIBLE_MATCHES_BEFORE_RECONCILE` | `GLRMASK_COMPACT_CAN_MATCH_BEFORE_RECONCILE` |

Do not add fallback support for the old names in this chunk.  Fallback support
would make sense for a released API.  This is an unpublished cleanup branch, so
keeping old names alive would only make the source harder to read.

## 6. Runtime API cleanup

Chunk 01 renamed the root-facing mapping API away from `mask_game_*`.  Chunk 02
finishes that move by removing hidden aliases from implementation and Python
bindings.

Use these names only:

```rust
internal_to_original_token_ids
original_to_internal_token_ids
fill_mask_and_internal_token_ids
```

In Python bindings, expose the same names.  Do not preserve:

```text
mask_game_mapping
mask_game_token_ids
fill_mask_and_mask_game_token_ids
```

The mathematical reason is simple: there is no “game” in the paper.  There is an
internal token quotient induced by compiled automata and scan relations.  The API
name should state that quotient directly.

## 7. Pipeline import update checklist

After moving modules, update `src/compiler/pipeline.rs` first.  It is the file
that ties all compile-time artifacts together, so it is the best smoke test for
whether the new source tree makes conceptual sense.

The imports should now include:

```rust
use crate::compile::scan_relation;
use crate::compile::terminal_dwa::classify::{...};
use crate::compile::terminal_dwa::grammar_helpers::{...};
use crate::compile::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
```

Calls should read in paper order:

1. construct / prepare vocab and tokenizer;
2. analyze grammar and GLR table;
3. classify terminal behavior;
4. build Terminal DWA;
5. compute scan relation / CanMatch artifact;
6. build template DFAs;
7. build Parser DWA;
8. reconcile runtime token spaces;
9. finalize `Constraint`.

Do not hide the Terminal DWA and scan relation under “stages” terminology after
this chunk.

## 8. Source grep acceptance checks

Run the following source-only checks.  They should produce no output:

```bash
rg -n 'id_map_and_terminal_dwa|constraint_possible_matches|mask_game|possible_matches|PossibleMatches|possible_match|pmv|PMV|L2P|l2p' src bindings README.md Cargo.toml
find src -path '*id_map_and_terminal_dwa*' -o -path '*constraint_possible_matches*' -o -path '*l2p*' -o -path '*l1*'
rg -n 'compiler::stages::(id_map_and_terminal_dwa|parser_dwa|terminal_dwa)|compiler::(constraint_possible_matches|possible_matches|pm_profile)' src bindings README.md Cargo.toml docs
```

Docs may still contain old names in migration tables.  That is intentional.  The
source tree, Python binding source, README, and manifests should not.

## 9. Things explicitly not solved by this chunk

Do not try to solve these yet:

- split the very large Terminal DWA and scan-relation files into smaller files;
- redesign GLR table ownership;
- remove environment-variable sprawl entirely;
- remove every `unwrap`, `expect`, or `panic!`;
- change template DFA algorithms;
- change serialization format;
- compile, test, benchmark, or rustfmt the crate.

Those are later chunks.  This chunk creates the language and object boundaries
needed to make those later chunks sane.

## 10. Manual review checklist

A reviewer should read the chunk in this order:

1. `docs/terminology.md` to understand canonical vocabulary.
2. `src/compile/mod.rs` to see the new source-tree boundary.
3. `src/compile/terminal_dwa/mod.rs` for the Terminal DWA builder home.
4. `src/compile/scan_relation/mod.rs` for Scan/CanMatch semantics.
5. `src/compile/parser_dwa/mod.rs` and `builder.rs` for Parser DWA naming.
6. `src/compiler/pipeline.rs` to see the compile pipeline after import updates.
7. `src/runtime/mask/mod.rs` and `src/runtime/commit/mod.rs` for runtime operation docs.
8. `bindings/python/src/lib.rs` for Python API names.
9. `docs/refactor/chunk_02/review_checklist.md` for acceptance checks.

A reviewer should not judge this chunk by whether every moved module is already
small.  The intended judgment is: “Can I now tell which mathematical object each
major file implements?”
