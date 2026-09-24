# S50 runtime recovery ? 24 September 2026

## Source and integration ledger

Base: `811ba219a21612ce2c7179e1bd35d815ef787931` (current UnlinkedConstraint API), which includes cd7767bb0 bounded eager proofs and prior totality, word-routing, integer arithmetic, composition and build fixes.

The missing Sep23 milestone was based on 6f19 plus frozen patch SHA256 `001d7c364cd88d651004d3393c3056d8a78795207d67e97477749c194fcef2df`. This recovery ports only its reviewed runtime ideas, not the entire dirty experiment tree.

| Unit | Decision | Adaptation |
| --- | --- | --- |
| S43 bounded raw physical radius | Default on | Dedicated constraint-local cache, budget in key, exact source-coordinate checks, fail closed for virtual/epsilon states and exhaustion |
| S45 narrow-parser scheduling | Default on | Select existing exact pruning kernel only for the proved singleton-parser/broader-lexer shape |
| S48 root-admission reuse | Default on | Own immutable GSS, check constraint owner AND GSS identity, lifetime limited to one mask invocation; reset/parser/provider children cannot inherit unrelated evidence |
| Tagged/config future-cache normalization | Already upstream | Preserved; not duplicated |
| Early pruning, contiguous LR0 and transition iterators | Already upstream | Preserved |
| Old serializer/depackaging/cache-eviction diagnostics | Rejected | Not imported |
| S51 raw-row batching | Default on after separate evaluation | Exact physical row materialization; charge each eligible byte before duplicate-edge elimination |

Disable controls for diagnostics or workload comparisons:
`GLRMASK_DISABLE_RAW_TERMINAL_RADIUS`, `GLRMASK_DISABLE_PARSER_NARROW_HOT`, `GLRMASK_DISABLE_ROOT_ADMISSION_MEMO`.
The default requires none of these flags. Existing eager-disable control remains available.

## Why the changes preserve masking semantics

Raw radius is a universal bounded proof over the product of the immutable safe-atom language and the exact physical lexer. A missing/dead transition bounds the earliest counterexample. Unsupported coordinates, epsilon transitions and exhausted budgets provide no positive certificate. Tokens outside the certified radius still use the existing exact recognizer. The proof does not infer permission from a byte-class or state-count heuristic.

Parser-narrow scheduling chooses between already exact evaluators; it does not modify their accepted language. Root memoization reuses a completed exact admission set only for the same owned persistent stack and constraint, within one mask call. Lexer-only projections may share it; parser children and recursive provider roots do not.

## Validation

Frozen native executable: SHA256 `b92099d7b05e06e6f59d3fc5b3f51c03ea8eecd8d1d2c21e4f0a6bd9703f116a`.
Root library: 797 passed, 51 pre-existing ignored. This includes independent bounded atom enumeration for ASCII and UTF-8, exhausted/invalid-coordinate guards, and memo owner/frontier identity regressions.
Public and registered regressions: 45 passed (22 public API, 19 API review, 2 word-limit, 2 integer multiple). The public test fixtures were migrated from removed Module/compile_module usage to the current UnlinkedConstraint binding API, preserving test counts and semantic assertions rather than restoring obsolete API aliases.

Independent online complete-mask checks (overlapping cohorts; do not sum as unique prefixes):
- Affected9: 22,436 frames, zero differences.
- Weak10: 13,616 frames, zero differences.
- Held-out64: 18,537 frames, zero differences.
- Parser-narrow4 with memo recomputation assertions: 5,065 frames, zero differences.
Frame identities and outcomes were checked, and mismatch/error/empty-output negative controls were tested. Mask pairs were compared immediately and discarded; zero masks were persisted.

## Full same-binary Windows comparison

8,326 schema IDs in the existing common-success cohort, all 31,934 examples, exactly 2,932,614 aligned timing positions. All three disabled versus restored defaults; same frozen binary and dependencies, two independent processes per configuration, verified actual P-core affinity 0x0fff, wall clock and symmetric allocator settings. All example outcomes and lengths identical, zero runtime errors, 118 unchanged pre-existing corpus-label disagreements. This is not a claim that all corpus cases are correct.

TBM numbers are per-position minima of the two processes; build numbers are per-schema medians. Original run maxima remain retained.

| Metric | All three disabled | Restored defaults |
| --- | ---: | ---: |
| p50 TBM, ms | 0.0058 | 0.0058 |
| p99 TBM, ms | 0.1599 | 0.1531 |
| p99.9 TBM, ms | 0.5749 | 0.547539 |
| p99.95 TBM, ms | 0.763577 | 0.6742 |
| p99.999 TBM, ms | 3.406172 | 2.133667 |
| Maximum stabilized TBM, ms | 3.5550 | 3.1005 |
| Mean schema build, ms | 2.740847 | 2.731875 |

Build time is essentially neutral; the small observed reduction is not claimed as a robust build optimization. Targeted 19-schema ablations verify distinct gains: parser scheduling reduces o64977 p99 approximately 1.295 -> 0.224 ms; root memo reduces o57878 p99 approximately 1.186 -> 1.020 ms; raw proof reduces o21072/o21075 p99 approximately 3.3 -> 2.0 ms.

Per-schema screening found zero p99 regressions exceeding both 10% and 50 microseconds, but three maximum regressions exceeding both 10% and 100 microseconds. The first small recheck accidentally omitted the calling affinity environment while writing hardcoded metadata; that folder is explicitly invalidated for timing comparisons, with raw data preserved. The replacement five-process balanced recheck sets the mask internally and verifies every actual child. It did not reproduce the material regressions: Snowplow sp_342 maximum 2.1557 -> 2.1439 ms; tizen_workspace 0.7089 -> 0.6944; o12210 0.7036 -> 0.6997. The earlier noisy o8475 is 0.3242 -> 0.3514 ms (8.4%, 27.2 microseconds), retained as a small local tradeoff; o46406 improves. Original outliers and invalidation notices remain. The original full-population run explicitly set affinity and printed verification for all four children; it is unaffected by the small-wrapper mistake.

## Matched LLguidance diagnostic

Pinned LLguidance dbaf504d, same corpus/vocab/wall/P-affinity and two-run contract. Whole-example identity matching excludes 218 explicitly listed examples, retaining 30,571 nonempty examples / 2,910,543 positions. No array truncation. Both-correct-only view retains 2,893,231 positions.

Restored GLRMask is strictly below LLguidance at every examined empirical quantile knot and the p99 interpolated endpoint through p100. Worst ratio is 0.764076 (both-correct 0.745852). Maximum is 3.1005 versus 7.4097 ms. This is an observed min-of-two result, not a universal latency bound or a guarantee for every platform. LLguidance remains faster in average compilation and at p90/p95 TBM.

## Reproduction and pending publication

Evidence under `C:/gdt22/artifacts/s50-restore-1531/`: target-20260924-163723, full-20260924-163915, target-20260924-173106 (replaces the invalidated target-20260924-170104). Independent masks and matched comparison under `C:/gdt22/artifacts/s50-independent-1557/` including matched-ll-1557-20260924-165304.
Drivers: restore_validate_1531.py, restore_regressions_1531.py, s50_recovery_online_1557.py, matched_s50_ll_1557.py. Frozen lock/source hashes are in each manifest.

The earlier cd7767bb0 final-JSB run lacked these runtime optimizations. Its raw data is preserved as diagnostic evidence, not relabelled. A fresh canonical Windows make final-jsb must use the integrated clean commit and record actual source/Cargo dependency identity, all four frameworks and the same CPU class. Publication A-D and notification delivery are separate completion gates.


## Separately accepted S51 row batching

Candidate source is based on the shipped core `5fedb7881`; only the raw-radius inner loop changed. Exact physical epsilon-free transitions are expanded once into a temporary 256-entry row. Consecutive identical product targets share the liveness/distance relaxation. Each eligible input byte is still charged to the original proof budget before deduplication. No acceptance predicate, horizon, cache ownership, virtual-state guard, or fallback was relaxed.

Default is enabled, with `GLRMASK_DISABLE_RAW_RADIUS_BATCH_ROWS=1` to disable. `GLRMASK_ASSERT_RAW_RADIUS_BATCH_ROWS=1` independently compares all 256 row entries to the original step function on every visited row.

Frozen runner SHA256: `1e385c576c1e64f756eaeed440af5384522b4b52fa62397775510bc70ba129da`. All 227 tracked production source/Cargo file hashes were unchanged before/after build and validation. Complete root suite passed 797/0, 51 existing ignored, with row assertions enabled. Strict online mask comparisons with row assertions versus rows disabled: affected9 22,436 frames; weak10 13,616; held-out64 18,537; zero differing frames/bits, identities and outcomes equal, no masks persisted. These overlap and are not a summed unique population.

The 19-schema targeted balanced three-process comparison reduced stabilized p99 from 1.651896 to 1.449664 ms (12.2%). Full 8,326-schema / 31,934-example / 2,932,614-position balanced two-process comparison retained all outcomes/lengths, zero runtime errors and the same 118 known label disagreements:

| Metric | Rows disabled | Rows enabled |
| --- | ---: | ---: |
| p50 TBM, ms | 0.0058 | 0.0058 |
| p99 TBM, ms | 0.1529 | 0.1533 |
| p99.9 TBM, ms | 0.5600 | 0.557977 |
| p99.95 TBM, ms | 0.685108 | 0.684869 |
| p99.99 TBM, ms | 1.958664 | 1.905673 |
| p99.999 TBM, ms | 2.135046 | 2.133732 |
| Stabilized maximum, ms | 3.0973 | 3.1177 |
| Mean schema build, ms | 2.734947 | 2.733040 |

This is a targeted work reduction, not a uniform improvement. Whole-population p99 and maximum are fractionally higher, while p99.99 improves about 2.7%; mean build is neutral. No per-schema p99 regression exceeded both 10% and 50 microseconds. One maximum screen (o60179) exceeded both 10% and 100 microseconds; a five-process fixed-P recheck did not reproduce it (maximum 1.1166 -> 1.1002 ms). The o21073 p99 benefit repeated (1.919107 -> 1.578724 ms); o8475 differences were small/noisy. The decision is to retain this modest optimization with an opt-out, not to claim that it solves additional universal tail/build goals.

Evidence: `C:/gdt22/artifacts/s51-batch-1707/target-20260924-173117`, `full-20260924-174409`, `target-20260924-175600`, and `online/`. The earlier unpinned `target-20260924-172103` is explicitly invalidated and not used for acceptance. The driver now sets required affinity internally and verifies actual child affinity instead of relying on a caller variable. Full raw measurements, maxima, outlier screens and exact comparisons are retained.
