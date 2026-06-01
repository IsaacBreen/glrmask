# Risk register and follow-up backlog

## Risks intentionally accepted in Chunk 04

### Compile not run

Per the workflow, this patch does not compile or test.  Import mistakes are possible.  They should be repaired after the tree shape is accepted.

### Direct-partition blob remains

`direct_partition/mod.rs` is still too large.  This chunk makes the top-level Terminal-DWA boundary correct first.  A later chunk should split the direct builder.

### Pair-partition internals remain dense

The pair-partition subtree still contains large files under equivalence analysis.  That is acceptable for this chunk because the highest-value publication confusion was at the top boundary.

### Env vars still exist

The env vars are centralized, not eliminated.  A later configuration/profiling chunk should move them behind typed compile options where possible.

## Follow-up tasks

1. Split `direct_partition/mod.rs` by local algorithm stage.
2. Split pair-partition equivalence analysis by proof object: state equivalence, vocab equivalence, compatibility witness, postprocessing.
3. Move Terminal-DWA profile lines behind a profiling sink rather than raw `eprintln!` calls.
4. Replace environment variables with `CompileOptions` where feasible.
5. Add property tests comparing partition schemes on tiny grammars.
6. Add a doc diagram showing the union of local relations.
7. Add a consistency check that direct and pair active-terminal masks are disjoint.
8. Add a consistency check that vocabulary partitions are disjoint and exhaustive.
9. Add a debug-only semantic equivalence checker for small vocabularies.
10. Add a benchmark matrix that records construction route, not only total time.
