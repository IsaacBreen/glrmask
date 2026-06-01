# Recommended patch review order

Review the patch in this order:

1. `src/runtime/state/mod.rs` to understand the new frontier core.
2. `src/runtime/state/cache.rs` and `scratch.rs` to classify nonsemantic fields.
3. `src/runtime/state/inspect.rs` and `force.rs` to verify moved methods.
4. `src/runtime/mask/dense_acc.rs` to verify DenseMaskAcc extraction.
5. `src/runtime/mask/mod.rs` to check imports and remaining phase graph.
6. `src/runtime/mask/bitset.rs` and `constants.rs` to confirm helper extraction.
7. `src/runtime/commit/options.rs` to check env reads.
8. `src/runtime/commit/parser_advance.rs` to check template/reference relation.
9. `src/runtime/commit/mask_assert.rs` to check debug oracle isolation.
10. `src/runtime/commit/mod.rs` last, because the diff there is easiest to read
    after knowing what disappeared.

Do not start by reviewing documentation.  Start with the source boundary, then
use documentation to decide whether the boundary is explained well enough.
