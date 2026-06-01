# Runtime symbol priority table

This table ranks runtime symbols by publication importance.

| Symbol | Priority | Reason | Desired prominence |
|---|---:|---|---|
| `Constraint` | P0 | immutable compiled artifact | public facade, documented |
| `ConstraintState` | P0 | live runtime frontier | public facade, documented |
| `ConstraintState::fill_mask` | P0 | paper Mask operation | public method, examples |
| `ConstraintState::commit_token` | P0 | paper Commit operation for LLM tokens | public method, examples |
| `ConstraintState::commit_bytes` | P0 | byte-level Commit operation | public method, tests |
| `MaskProfile` | P1 | runtime diagnostics | public but secondary |
| `CommitProfile` | P1 | runtime diagnostics | public but secondary |
| `DenseMaskAcc` | P2 | internal Mask representation | private to mask module |
| `CommitBuffers` | P2 | allocation reuse | private to runtime state |
| `MaskCacheData` | P2 | cache optimization | private to runtime state |
| `template_advance_enabled` | P3 | env policy | private options helper |
| `assert_mask_commit_equivalence` | P3 | debug oracle | private helper |

P0 symbols should have polished rustdoc and examples.  P1 symbols should be
stable enough for benchmarks.  P2 symbols should have comments but no public API
status.  P3 symbols should remain hidden implementation details.
