# Symbol ledger

This ledger maps important symbols to their new home.

| Symbol | Home | Meaning |
| --- | --- | --- |
| `ScanOutcome` | `src/scan/relation.rs` | Conceptual result of scanning one byte fragment. |
| `CompletedTerminals` | `src/scan/relation.rs` | Terminals completed inside a scanned fragment. |
| `PartialLexerState` | `src/scan/relation.rs` | Boundary state requiring future completion. |
| `CanMatchSet` | `src/scan/relation.rs` | Terminals that can complete from a partial lexer state. |
| `execute_tokenizer_from_state` | `src/scan/execution.rs` | Runtime primitive byte scan. |
| `RuntimeCanMatchByTerminal` | `src/compile/scan_relation/types.rs` | Runtime CanMatch table by terminal. |
| `ScanRelationConfig` | `src/compile/scan_relation/types.rs` | Scan-relation construction policy boundary. |
| `ScanRelationProfile` | `src/compile/scan_relation/types.rs` | Timing summary. |
| `ScanRelationComputation` | `src/compile/scan_relation/types.rs` | Complete compile-time output. |
| `OrderedVocab` | `src/compile/scan_relation/ordered_vocab.rs` | Byte-sorted vocabulary. |
| `OrderedVocabTrieArtifacts` | `src/compile/scan_relation/ordered_vocab.rs` | Cached ordered vocab plus trie. |
| `compute_scan_relation_vocab_equivalence_map_fast` | `src/compile/scan_relation/vocab_equivalence.rs` | Fast CanMatch token quotient. |
| `build_scan_relation_vocab_and_weights_from_interval_maps` | `src/compile/scan_relation/vocab_materialize.rs` | Grouped interval maps to runtime weights. |
| `build_legacy_scan_relation_vocab_and_weights_from_interval_maps` | `src/compile/scan_relation/legacy_materialize.rs` | Expanded validation oracle. |
| `collect_sparse_root_can_match` | `src/compile/scan_relation/root_collect.rs` | Small-root sparse collection. |
| `compute_scan_relation_for_vocab` | `src/compile/scan_relation/compute.rs` | Main pipeline entry point. |
| `prepare_vocab_for_scan_relation` | `src/compile/scan_relation/compute.rs` | Warm ordered vocab/trie cache. |
