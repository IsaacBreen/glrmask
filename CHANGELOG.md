# Changelog

## Unreleased — public API cleanup

### Breaking Change: `commit` is now infallible
- `ConstraintState::commit(constraint, token_id)` now returns `()` (was `Result<(), GlrMaskError>`)
- Invalid token ID (not in vocabulary) is now a silent no-op; the next mask will be empty
- All callers updated — no more `.unwrap()` / `.expect()` on `commit`

### Visibility cleanup
- `runtime::gss_acc` and `runtime::leveled_gss` changed to `pub(crate)` — not part of public API
- Removed top-level `BitSet` re-export from `lib.rs` — not part of the supported public API

### Public API additions

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

### Highlights

- Vocabulary-specific grammar-constrained decoding for EBNF, Lark, and a documented pragmatic subset of JSON Schema.
- Reusable compiled `Constraint` objects with incremental mask, commit, completion, and forced-prefix operations.
- GLR-based parsing for ambiguous and genuinely context-free grammars, including tokenizations that cross grammar-terminal boundaries.
- Rust and Python APIs, including the public Python `glrmask` import surface and clean source installation with declared NumPy dependency.
- Constraint serialization for compile-once, load-and-run deployments, plus a smaller execution-only runtime crate for serving artifacts.
- Build-only Python wheel CI covering Python 3.9–3.13 across manylinux x86_64/aarch64, macOS x86_64/arm64, and Windows x86_64.

### Release evidence and caveats

- The bounded native release benchmark is documented in [`docs/benchmark-0.1.md`](docs/benchmark-0.1.md), including exact hardware, methodology, compile latency, runtime percentiles, and tail caveats.
- JSON Schema support is not full specification conformance; see [`docs/json-schema-semantic-deviations.md`](docs/json-schema-semantic-deviations.md).
- The Vercel benchmark remains a long compile-tail case at about 25 seconds on the measured shared 2-vCPU host.
