# Quality bar for publication

Publication cleanup is not just making code compile.  For this subsystem, the
quality bar is that a reader can answer the following questions quickly:

1. What is the scan relation?
2. What is `CanMatch`?
3. Why is `CanMatch` not Terminal-DWA equivalence?
4. Where is the byte-sorted vocabulary built?
5. Where are token classes induced by CanMatch signatures?
6. Where are runtime weights materialized?
7. Where is the slow validation oracle?
8. Where does runtime commit scan bytes?
9. Which file is the pipeline entry point?
10. Which code is compile-time only?

After Chunk 06, the intended answers are:

1. `src/scan/relation.rs` and `src/compile/scan_relation/mod.rs`.
2. `src/compile/scan_relation/types.rs` and `vocab_materialize.rs`.
3. Stated in `mod.rs`, `types.rs`, and docs.
4. `ordered_vocab.rs`.
5. `vocab_equivalence.rs` and `vocab_materialize.rs`.
6. `vocab_materialize.rs`.
7. `legacy_materialize.rs`.
8. `scan/execution.rs`.
9. `compute.rs`.
10. Everything under `compile::scan_relation` except conceptual names; runtime
    execution is under `scan::execution`.

If a future reader cannot answer these questions from filenames and module docs,
the subsystem has regressed.
