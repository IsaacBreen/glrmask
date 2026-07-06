# Possible-Match Reconciliation Optimization Report

Base commit: `2704e56b890088ecce5f62742d0e991becf66325`

Worktree commit:
- `7960a6ca3 Skip redundant mapped-artifact ID reconciliation`

## Summary

The optimization avoids building a synthesized common `InternalIdMap` and remapping
both mapped artifacts when reconciliation has an exact cheaper result:

- If one side has no exposed weights, adopt the non-empty side's ID map. The empty
  relation remains empty under any shared ID space.
- If one ID map refines the other on both tokenizer-state IDs and vocab-token IDs,
  use the finer ID map directly and remap only the coarser side.
- All other cases fall back to the existing generic reconciliation path.

This is implemented in `src/compiler/stages/mapped_artifact/reconcile.rs`.

## BFCL Catalog 512 Medians

Normal production TI-on mode, `GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY=1`,
`GLRMASK_PROFILE_COMPILE=1`, `GLRMASK_PROFILE_COMPILE_SUMMARY=1`.

| Mode | Metric | Baseline median | Final median |
| --- | ---: | ---: | ---: |
| 1 thread | `shared_id_reconcile_ms` | 67.545 | 1.075 |
| 1 thread | `possible_matches_pipeline_ms` | 78.165 | 11.915 |
| 1 thread | `compile_ms` | 1546.822 | 1421.785 |
| 10 threads | `shared_id_reconcile_ms` | 83.674 | 1.285 |
| 10 threads | `possible_matches_pipeline_ms` | 117.041 | 14.012 |
| 10 threads | `compile_ms` | 1023.639 | 663.521 |

Baseline logs:
- `/tmp/bfcl512-p2-p7-p8-split-t1-r1-20260706.log`
- `/tmp/bfcl512-p2-p7-p8-split-t1-r2-20260706.log`
- `/tmp/bfcl512-p2-p7-p8-split-t1-r3-20260706.log`
- `/tmp/bfcl512-p2-p7-p8-split-t10-r1-20260706.log`
- `/tmp/bfcl512-p2-p7-p8-split-t10-r2-20260706.log`
- `/tmp/bfcl512-p2-p7-p8-split-t10-r3-20260706.log`

Final timing logs:
- `/tmp/bfcl512-reconcile-emptyfast-t1-r1.log`
- `/tmp/bfcl512-reconcile-emptyfast-t1-r2.log`
- `/tmp/bfcl512-reconcile-emptyfast-t1-r3.log`
- `/tmp/bfcl512-reconcile-emptyfast-t10-r1.log`
- `/tmp/bfcl512-reconcile-emptyfast-t10-r2.log`
- `/tmp/bfcl512-reconcile-emptyfast-t10-r3.log`

## Range Counts

BFCL Catalog 512 medians:

| Count | Baseline | Final |
| --- | ---: | ---: |
| `terminal_dwa_interned_ranges_before_pm_reconcile` | 43190 | 43190 |
| `possible_matches_interned_ranges_before_pm_reconcile` | 0 | 0 |
| `terminal_pm_joint_interned_ranges` | 94006 | 43190 |
| `parser_dwa_interned_ranges` | 94225 | 55487 |
| `possible_matches_interned_ranges` | 0 | 0 |
| `parser_pm_joint_interned_ranges` | 86518 | 47780 |

The BFCL 512 possible-match artifact is empty after compaction, so the previous
generic reconciliation rebuilt terminal range tables against a synthetic joint ID
space and inflated the terminal/parser range intern tables. The final path keeps
the terminal ID map and range tables unchanged for the empty possible-match side.

## Remaining Cost

The dominant possible-match pipeline cost after this change is collection:

- 1-thread final samples: `possible_matches_collect_ms` = 10.117, 10.237, 10.737
- 10-thread final samples: `possible_matches_collect_ms` = 11.589, 11.694, 12.338

Shared-ID reconciliation is no longer material for BFCL 512 in these samples.

## Validation Commands

Rust checks:

```bash
cargo fmt --check 2>&1 | tee /tmp/glrmask_reconcile_fmt_check_current.log
CARGO_INCREMENTAL=0 cargo test -p glrmask reconcile::tests:: 2>&1 | tee /tmp/glrmask_reconcile_tests_current.log
CARGO_INCREMENTAL=0 cargo check -p glrmask 2>&1 | tee /tmp/glrmask_reconcile_cargo_check_current.log
CARGO_INCREMENTAL=0 cargo test -p glrmask possible_match 2>&1 | tee /tmp/glrmask_reconcile_possible_match_tests.log
CARGO_INCREMENTAL=0 cargo test -p glrmask --test integration 2>&1 | tee /tmp/glrmask_reconcile_integration_tests.log
```

Results:
- `cargo fmt --check`: passed.
- `cargo test -p glrmask reconcile::tests::`: passed, 4 tests.
- `cargo check -p glrmask`: passed.
- `cargo test -p glrmask possible_match`: passed, 0 matching tests.
- `cargo test -p glrmask --test integration`: failed, 75 passed and 11 failed. The inspected failure panicked in grammar precondition checking for indirect left recursion before possible-match reconciliation (`/tmp/glrmask_reconcile_integration_one_fail_nocapture.log`).

FFI build/install:

```bash
unset CONDA_PREFIX
VIRTUAL_ENV=/tmp/glrmask2-possible-match-reconcile-venv \
PATH=/tmp/glrmask2-possible-match-reconcile-venv/bin:$PATH \
CARGO_INCREMENTAL=0 \
maturin develop --release --manifest-path python/Cargo.toml \
2>&1 | tee /tmp/glrmask_reconcile_maturin_release_final.log
```

Result: passed; installed editable `glrmask-0.1.0` into
`/tmp/glrmask2-possible-match-reconcile-venv`.

BFCL validation:

```bash
GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY=1 \
GLRMASK_PROFILE_COMPILE=1 \
GLRMASK_PROFILE_COMPILE_SUMMARY=1 \
GLRMASK_COMPILE_THREADS=1 \
RAYON_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
make example-specific PROBLEM=bfcl_catalog/size_008/catalog_008_000 \
  PYTHON=/tmp/glrmask2-possible-match-reconcile-venv/bin/python \
  FRAMEWORKS="glrmask_native llguidance_native" \
  TIMING_RUNS="glrmask_native:1,llguidance_native:1" \
  BUILD_TIMEOUT=120 OUTPUT=/tmp/bfcl008-reconcile-final-t1.json \
  SHOW_TBM_ABOVE=0 \
2>&1 | tee /tmp/bfcl008-reconcile-final-t1.log

GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY=1 \
GLRMASK_PROFILE_COMPILE=1 \
GLRMASK_PROFILE_COMPILE_SUMMARY=1 \
GLRMASK_COMPILE_THREADS=1 \
RAYON_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
make example-specific PROBLEM=bfcl_catalog/size_512/catalog_512_000 \
  PYTHON=/tmp/glrmask2-possible-match-reconcile-venv/bin/python \
  FRAMEWORKS="glrmask_native llguidance_native" \
  TIMING_RUNS="glrmask_native:1,llguidance_native:1" \
  BUILD_TIMEOUT=120 OUTPUT=/tmp/bfcl512-reconcile-final-compare-t1.json \
  SHOW_TBM_ABOVE=0 \
2>&1 | tee /tmp/bfcl512-reconcile-final-compare-t1.log
```

Results:
- Catalog 008 sweep status: `attempted: 1`, `ok: 1`.
- Catalog 512 sweep status: `attempted: 1`, `ok: 1`.
- Post-sweep adjudication reported an unhandled framework disagreement for both
  008 and 512 at position 0, token `{` (`glrmask_native` accepts,
  `llguidance_native` rejects). This was also observed in the earlier post-fix
  BFCL 512 comparison run and is not introduced by the reconciliation shortcut.

Repeated BFCL 512 timing samples used the same TI/profile environment with:

```bash
make example-specific PROBLEM=bfcl_catalog/size_512/catalog_512_000 \
  PYTHON=/tmp/glrmask2-possible-match-reconcile-venv/bin/python \
  FRAMEWORKS="glrmask_native" TIMING_RUNS="glrmask_native:1" \
  BUILD_TIMEOUT=120 OUTPUT=/tmp/bfcl512-reconcile-emptyfast-...json \
  SHOW_TBM_ABOVE=0 ARGS="--allow-single-framework"
```

Results: all six single-framework timing sweeps reported `attempted: 1`, `ok: 1`.

## Correctness Argument

For the empty-side shortcut, `WeightRefs` is the only semantic surface that
reconciliation mutates. If a mapped artifact exposes no weights, its relation is
empty, and remapping an empty relation through any common ID map is still empty.
Choosing the non-empty side's ID map leaves all non-empty weights bit-identical
and gives the empty side an equivalent shared ID space.

For the refinement shortcut, if each finer tokenizer-state class and vocab-token
class maps to exactly one coarser class, the generic common map over `(left,
right)` class pairs is isomorphic to the finer map. Reusing the finer map and
remapping only the coarser side therefore produces the same artifact semantics
as the generic pair-map construction. The new tests compare the fast path against
the generic remapper for non-empty coarser weights and cover the empty-side case
after domain compaction.

Cases that are not empty and not a refinement keep using the original generic
sort/dedup/remap path.
