# Terminal DWA source boundary

This directory builds the Terminal DWA: the weighted deterministic automaton over grammar-terminal sequences.

The intended source reading order is:

1. `mod.rs` — names the subsystem boundary and re-exports the two build entry points.
2. `builder.rs` — orchestrates shared caches, local partition builds, and global merge.
3. `options.rs` — reads environment-controlled build policy.
4. `vocab_partition.rs` — chooses the caller-token partition used for local builds.
5. `global_state_map.rs` — computes the global tokenizer-state quotient shared with scan-relation construction.
6. `partition.rs` — builds one local partition by splitting terminals into direct and pair paths.
7. `direct_partition/` — direct/single-step Terminal-DWA construction.
8. `pair_partition/` — multi-step Terminal-DWA construction.
9. `merge.rs` — reconciles local id maps and local DWAs into one global mapped artifact.
10. `types.rs` — small shared terminal-DWA types and profile counters.

The key invariant is that partitioning changes only the construction route, not the denotation:

```text
[[TerminalDWA]](r) = { (q, v) : r is one completed terminal sequence produced by scanning beta(v) from q }
```

All local id spaces must be merged back into the original tokenizer-state space and original caller-token space before the runtime artifact is finalized.
