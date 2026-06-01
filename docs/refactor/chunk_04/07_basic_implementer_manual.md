# Basic implementer manual for Chunk 04

## For a basic implementer

This chunk is not about changing algorithms.  It is about moving code so that the algorithm is legible.

When applying or reviewing it manually, do the following:

1. Open `src/compile/terminal_dwa/mod.rs`.
2. Confirm it is just a module map and re-export list.
3. Open `options.rs`.
4. Confirm all environment-variable reads moved there.
5. Open `vocab_partition.rs`.
6. Confirm it produces `Arc<[Vocab]>` and does not build a DWA.
7. Open `global_state_map.rs`.
8. Confirm it contains the max-length state-map build.
9. Open `builder.rs`.
10. Confirm it calls the other files instead of owning their internals.

Do not try to fix compiler errors in the middle of this review.  The purpose is to verify shape.  Compile repair comes only after the structural target is on screen.

## Common mistakes

### Mistake: putting option reads back in `builder.rs`

Do not do that.  Builder is orchestration.  Options are policy.

### Mistake: returning more than sub-vocabs from `vocab_partition.rs`

Do not do that.  Partitioning returns construction pieces, not local automata.

### Mistake: moving global max-length into scan relation

Do not do that.  The global state map is shared support.  It should be usable by both Terminal DWA and scan relation.

### Mistake: interpreting partitioning as semantics

Do not do that.  The final denotation is independent of the partition choice, assuming every strategy is sound.
