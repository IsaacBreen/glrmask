# Chunk 04 review checklist

## Static review checklist

Use this checklist before compiling.

### Boundary files

- [ ] `src/compile/terminal_dwa/mod.rs` is under 100 lines.
- [ ] `mod.rs` contains no direct `std::env::var` reads.
- [ ] `mod.rs` contains no `eprintln!` profile lines.
- [ ] `mod.rs` re-exports the public internal entry points used by neighbouring phases.

### Options

- [ ] All top-level Terminal-DWA environment variables are named in `options.rs`.
- [ ] Raw strings are converted into enums or typed values at the boundary.
- [ ] Option functions do not build automata.
- [ ] Option functions do not mutate vocabularies or id maps.

### Vocabulary partitioning

- [ ] `vocab_partition.rs` returns sub-vocabs, not DWA artifacts.
- [ ] char-type partitioning uses `CharTypeSubVocabs` derived cache.
- [ ] pair-cost strategy profile output remains close to the strategy that computes it.
- [ ] auto strategy logs the decision reason and the quantities used to choose.

### Global state map

- [ ] `global_state_map.rs` calls `run_state_equivalence_pipeline` with `StateEquivalenceScope::Global`.
- [ ] It does not know about direct partitions.
- [ ] It does not know about pair partitions except for the state-equivalence pipeline it delegates to.

### Builder

- [ ] `builder.rs` can be summarized as choose partitions, build locals, merge locals.
- [ ] It does not parse `GLRMASK_PARTITION_SCHEME` itself.
- [ ] It does not implement char-type classification itself.
- [ ] It does not implement global state equivalence itself.

### Follow-up markers

- [ ] `direct_partition/mod.rs` remains the largest Terminal-DWA file and is explicitly noted as follow-up work.
- [ ] No legacy `l1`/`l2p` names appear in new source files except in historical docs.
- [ ] The package includes a checks file documenting what was and was not run.
