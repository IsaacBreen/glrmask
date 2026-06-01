# Chunk 21 risk register: tests_examples_benchmarks_docs

| Risk | Why it matters | Mitigation |
|---|---|---|
| Stale import | Compile repair may accidentally use old architecture. | Grep for old paths and keep matches only in shims. |
| Hidden semantic change | Moving code can invite opportunistic logic edits. | Compare reference tests before and after. |
| Visibility creep | Making modules public can leak internals. | Prefer `pub(crate)` and narrow reexports. |
| Cache inconsistency | Moved artifact/cache code may rebuild from wrong fields. | Add deterministic rebuild tests. |
| Fast-path mismatch | Optimization may not match reference relation. | Differential test fast path against reference. |
| Documentation drift | New names can diverge from paper terminology. | Keep `docs/style/naming.md` as source of truth. |
| Beginner confusion | Compatibility shims can look canonical. | Mark shims as hidden and say "Publication code should use...". |
| Benchmark noise | Structural changes can change profile labels. | Benchmark only after semantic tests pass. |

## Highest-priority manual inspection

Open the files listed in the implementation manual and verify that the first screen of each file says what mathematical object it owns.  If not, add or improve the module-level doc comment before doing any more code movement.
