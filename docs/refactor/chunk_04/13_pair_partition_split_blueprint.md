# Detailed follow-up blueprint: pair partition

The pair-partition subtree is already more modular than direct partition, but the equivalence-analysis layer is still deep and performance-heavy.  This blueprint defines later cleanup without performing it in Chunk 04.

## Current responsibilities

- `pair_partition/mod.rs` — Local pair-partition orchestration, tokenizer simplification cache, initial-state projection, and top-level builder.
- `pair_partition/nwa_builder.rs` — Construction of the intermediate NWA via trie walk and weighted transitions.
- `pair_partition/postprocess.rs` — NWA pruning, canonicalization, collapse, and disallowed-follow constraints.
- `pair_partition/equivalence_analysis/state_equivalence/*` — Pipeline for state equivalence passes and configuration.
- `pair_partition/equivalence_analysis/state/*` — Concrete state-equivalence implementations.
- `pair_partition/equivalence_analysis/vocab/*` — Vocabulary equivalence and token-pair distinguishing logic.
- `pair_partition/equivalence_analysis/compat.rs` — Compatibility witness between tokenizer views and flat transition tables.
- `pair_partition/equivalence_analysis/combined.rs` — Composition of state and vocab equivalence into an internal id map.

## Future cleanup tasks
1. Extract pair-partition option reads from `pair_partition/mod.rs` and equivalence-analysis files into typed option modules.
2. Rename cache types so each says which coordinate space it keys: original tokenizer state, simplified tokenizer state, terminal group id, or internal token id.
3. Add explicit compatibility-witness structs instead of passing raw `flat_trans` tables through many layers.
4. Split `vocab/fast.rs` into DFA construction, batch observation, trie walk, diagnostics, and public entry point.
5. Add small proof comments before every compaction/postprocess pass explaining what relation is preserved.
6. Move profile-only structs out of algorithm files where they obscure the proof path.
7. Add debug-only semantic comparison for very small vocabularies that builds the relation naively and compares it to the pair builder.

