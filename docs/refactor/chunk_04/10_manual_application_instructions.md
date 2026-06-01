# Manual application instructions for Chunk 04

This document explains how to recreate the patch without needing to understand Rust deeply.  It is deliberately redundant with the diff because the goal is to make the desired shape unmistakable.

## Step 1: Create `src/compile/terminal_dwa/options.rs`

Move all top-level Terminal-DWA environment-variable parsing into this file.  Define `VocabPartitionScheme` so internal code stops comparing raw `GLRMASK_PARTITION_SCHEME` strings.  Keep parsing functions small and side-effect-free except for reading the process environment.

Acceptance checks:

- `rg 'GLRMASK_PARTITION_SCHEME|GLRMASK_PAIR_PARTITION|GLRMASK_USE_GLOBAL_MAX_LENGTH' src/compile/terminal_dwa` should show the policy names concentrated in `options.rs` plus profile text in strategy files.
- Raw scheme strings should be converted to `VocabPartitionScheme` before use.

## Step 2: Create `src/compile/terminal_dwa/vocab_partition.rs`

Move char-type sub-vocab caching and partition choice logic here.  The file should return `Arc<[Vocab]>`, never a DWA, id map, scan relation, parser table, or runtime artifact.

Acceptance checks:

- `choose_terminal_dwa_sub_vocabs` should be the only top-level function that decides the sub-vocab strategy.
- The file should not import `DWA`.

## Step 3: Create `src/compile/terminal_dwa/global_state_map.rs`

Move the global max-length state-map computation here.  This file may call the state-equivalence pipeline but should not choose direct or pair partitions.

Acceptance checks:

- `build_global_max_length_state_map` should remain callable through `terminal_dwa::build_global_max_length_state_map`.
- The file should mention `StateEquivalenceScope::Global`.

## Step 4: Create `src/compile/terminal_dwa/builder.rs`

Move the two top-level build functions here.  Keep timing, shared caches, parallel local partition builds, empty-vocab handling, and global merge orchestration here.

Acceptance checks:

- `build_terminal_dwa_with_precomputed_global_max_length` should call `vocab_partition::choose_terminal_dwa_sub_vocabs`.
- The global merge should remain in the top-level build algorithm.

## Step 5: Rewrite `src/compile/terminal_dwa/mod.rs`

Turn it into a boundary module: docs, module declarations, and re-exports.  It should not read env vars or contain profile `eprintln!` calls.

Acceptance checks:

- The file should be shorter than 100 lines.
- The file should contain no `fn` bodies except maybe re-export declarations; in this patch it contains none.

## Step 6: Add `src/compile/terminal_dwa/README.md`

Document the reading order and the denotation in plain text for source browsers.

Acceptance checks:

- The file exists and is referenced in the package manifest/checks.

## Step 7: Update `docs/terminal_dwa.md`

Record the new source boundary so paper/code terminology remains aligned.

Acceptance checks:

- The file exists and is referenced in the package manifest/checks.

## Step 8: Add `docs/refactor/chunk_04/*`

Include the implementation manual, mathematical contracts, inventories, and review checklists.

Acceptance checks:

- The file exists and is referenced in the package manifest/checks.

## What not to do

- Do not compile in the middle of applying this chunk.
- Do not mix runtime mask/commit cleanup into this patch.
- Do not rename the public crate API in this patch.
- Do not change the mathematical meaning of weights or id maps.
- Do not replace the partition strategies with a new algorithm; this is a boundary cleanup.

