# Patch acceptance criteria

Accept this chunk only if these criteria hold.

## Must hold immediately

- Source tree contains `src/scan/`.
- `src/lib.rs` declares `pub(crate) mod scan;`.
- `src/runtime/commit/tokenizer_scan.rs` delegates scan execution.
- `src/compile/scan_relation/mod.rs` is below 150 lines.
- `src/compile/scan_relation/README.md` exists.
- Chunk 06 docs exist under `docs/refactor/chunk_06/`.

## Must hold after compilation repair

- All imports and module visibility errors are fixed without reversing the split.
- No file grows back into a monolith.
- Existing behavior of scan-relation construction is preserved.
- Existing benchmark harnesses can still access the compiled constraint.

## Must hold before publication

- Scan relation is described in paper-aligned terms throughout comments.
- Environment-variable policy is documented or moved into options.
- Partial-boundary tests exist.
- Serialization tests cover `can_match` artifacts.
- Performance benchmarks show no material regression from structural cleanup.
