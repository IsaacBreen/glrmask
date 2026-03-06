# Changelog

## Unreleased — public/runtime boundary conformance pass

### Breaking Change: `commit` is now infallible
- `ConstraintState::commit(constraint, token_id)` now returns `()` (was `Result<(), GlrMaskError>`)
- Invalid token ID (not in vocabulary) is now a silent no-op; the next mask will be empty
- All callers updated — no more `.unwrap()` / `.expect()` on `commit`

### Visibility cleanup
- `runtime::gss_acc` and `runtime::leveled_gss` changed to `pub(crate)` — not part of public API
- Removed top-level `BitSet` re-export from `lib.rs` — not in plan's public surface

### New Public API (aligned with rewrite plan)

#### `Constraint`
- `mask_len() -> usize` — number of `u32` words needed for a mask buffer

#### `ConstraintState`
- `mask(constraint) -> Vec<u32>` — allowed-token mask as `u32` words
- `fill_mask(constraint, buf: &mut [u32])` — zero-alloc mask fill
- `is_finished(constraint) -> bool` — grammar fully satisfied (alias for `is_accepting`)
- `commit_bytes(constraint, bytes: &[u8])` — infallible raw-byte commit
- `commit_tokens(constraint, tokens: &[u32])` — batch token commit
- `force(constraint) -> Vec<u32>` — greedy forced-token prefix

### Internal Compiler Improvements
- Added `non_greedy_finalizers` and `possible_future_group_ids` tracking to `Dfa` and `Nfa`
- Non-greedy terminal metadata propagated through `TokenizerDfa` → `TerminalDwa`
- `terminal_dwa.rs`: full vocabulary-trie walk replaces `possible_matches` projection
- `template.rs`: template bundle construction groups equivalent terminal characterizations
- `parser_dwa.rs`: refactored to use `build_terminal_dwa` + `build_template_bundles`
- Added `compiler/labels.rs` — shared parser-state label encoding
- Added `compiler/resolve_negatives.rs` — cancellation semantics for negative NWA labels

---

## 0.1.0 — Initial Release
### Features
- **EBNF, Lark, and JSON Schema** grammar frontends
- **GLR parser** for ambiguous grammar support
- **DWA-based mask computation** in microseconds
- **Serialization**: `save()`/`load()` via bincode
- **Force detection**: `forced_token()` and `is_dead()` utilities
- 206 tests (179 unit + 27 integration)

### Architecture
- `ds/`: Core data structures (RangeSet, U8Set, BitSet)
- `automata/`: DFA, NFA, regex, weighted automata (NWA, DWA)
- `compiler/`: Grammar → GLR table → NWA → DWA pipeline
- `frontend/`: EBNF, Lark, JSON Schema parsers
- `runtime/`: Constraint state, mask computation, force detection
