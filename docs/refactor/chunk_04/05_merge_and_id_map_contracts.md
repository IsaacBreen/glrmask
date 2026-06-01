# Merge and id-map contracts

## Local ids

Each local builder is allowed to use local ids.  This includes local tokenizer-state ids and local internal token ids.  The local id map is the witness that explains how those ids map back to original ids.

A local DWA without its id map is not a valid compile artifact.  The pair `(id_map, dwa)` is the artifact.

## Per-partition merge

Within one vocabulary partition, direct and pair local artifacts may both exist.  The direct artifact covers direct/single-step terminals; the pair artifact covers multi-step terminals.  `partition.rs` merges those local artifacts into a single local result.

The merge is sound because the terminal masks created by terminal path-length classification are disjoint.  If a future change allows overlap, the merge proof must be revisited.

## Global merge

Across vocabulary partitions, each local artifact covers a different subset of caller tokens.  The global merge unions those local relations and reconciles token ids.

The global merge must preserve:

- original tokenizer state identity where states are not quotient-equivalent;
- original caller token ids;
- terminal labels;
- weight semantics over `(state, token)` pairs.

## Empty vocab case

`builder.rs` retains the empty-vocabulary path.  It constructs an empty DWA with a state map over tokenizer states and an empty token map.  This is a degenerate but important identity case: the relation is empty for every terminal sequence, but the tokenizer-state axis still has a defined map.

## Compaction

Compaction is permitted after local construction and after merge only if the id-map witness is updated consistently.  Compacting an automaton while forgetting to compact the corresponding id map is a semantic bug even if the DWA language looks smaller and valid.
