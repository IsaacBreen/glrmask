# Terminal-DWA construction optimization — BFCL Catalog 512

## Scope and isolation

- **Base:** `glrmask-main` at `2704e56b890088ecce5f62742d0e991becf66325`
- **Worker branch:** `perf/terminal-dwa-construction-20260706`
- **Primary worktree:** `/tmp/glrmask2-terminal-dwa-opt-20260706`
- **Measurement-only detached base worktree:** `/tmp/glrmask2-terminal-dwa-base-measure-20260706`
- **Isolated venv/FFI:** `/tmp/glrmask2-terminal-dwa-opt-venv-20260706`
- **CFA used read-only:** `/Users/isaacbreen/Projects2/constraint-framework-analysis`

No changes were made to `glrmask-main`. This branch was kept independently reviewable on the requested base and did not merge or cherry-pick:

```text
27b2d0b6d  7dbf49ce5  7960a6ca3  44a3e26fd  0125b55f0
```

The change is confined to `src/automata/weighted_u32/determinize.rs`; it does not touch classification/routing, TI policy or discovery, possible-match reconciliation, parser-DWA construction, runtime finalization, or P2 L1 profile aggregation.

## Change

The terminal-DWA P0 profile establishes a large transient exact determinization shape:

```text
terminal NWA after postprocess:          2,345 states
weighted determinization output:         5,479 states / 33,917 transitions
acyclic minimization output:                 10 states /  2,939 transitions
```

The minimized artifact remains unchanged. The optimization reduces transient determinizer allocation and lookup work without changing any automaton relation:

1. **Share each determinized subset between the map and queue.**
   - Replaced duplicate `Vec<(state, Weight)>` queue/map storage with a single immutable `Arc<[(state, Weight)]>`.
   - The queue carries the already allocated DWA state ID, eliminating the dequeue-time map lookup.

2. **Add an exact singleton-subset front cache.**
   - The cache key is `(destination state, Weight allocation identity)`.
   - It is only a fast front cache. Every miss uses the existing structural subset map, so equal weights with distinct allocations still retain the original exact behavior.

3. **Directly emit safe edges of one-entry subsets.**
   - For a grouped outgoing-weight class, edges whose label has exactly one target and whose target is epsilon-free can be emitted directly.
   - The path-weight intersection remains computed once per exact shared-weight group.
   - Mixed groups retain the old label staging path for their non-direct edges.

4. **Regression coverage.**
   - Added a determinizer test with a shared-weight group containing both directly emit-able edges and an epsilon-closure fallback edge; grouped and generic determinization remain equivalent.

## Fresh benchmark method

All measurements use the isolated venv and FFI. For the controlled base/final comparison, the base FFI was installed into that venv from the detached base worktree, measured, then the final FFI was installed from this worktree and measured with the same command shape.

```bash
PY=/tmp/glrmask2-terminal-dwa-opt-venv-20260706/bin/python
CFA=/Users/isaacbreen/Projects2/constraint-framework-analysis

PYTHONPATH="$CFA" \
RAYON_NUM_THREADS=<1|10> GLRMASK_COMPILE_THREADS=<1|10> \
GLRMASK_PROFILE_L2P_TIMING=1 \
GLRMASK_PROFILE_COMPILE=1 \
GLRMASK_PROFILE_COMPILE_SUMMARY=1 \
"$PY" -m scripts.profile_build \
  --problem bfcl_catalog/size_512/catalog_512_000 \
  --framework glrmask_native
```

Each configuration uses three fresh Python processes. Raw logs and JSON are retained outside the worktree under:

```text
/tmp/terminal-dwa-final-comparison-20260706/
```

## Stage timing results

### Single-threaded

`RAYON_NUM_THREADS=1`, `GLRMASK_COMPILE_THREADS=1`.

| Metric | Base raw ms | Base median | Final raw ms | Final median | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| split terminal-DWA stage | 772.769, 796.785, 780.034 | 780.034 | 768.049, 767.268, 780.444 | 768.049 | **-11.985 ms (-1.54%)** |
| whole compile | 1554.968, 1601.190, 1547.887 | 1554.968 | 1541.811, 1498.893, 1531.427 | 1531.427 | **-23.541 ms (-1.51%)** |
| global terminal merge | 31.058, 32.901, 32.535 | 32.535 | 31.604, 32.025, 33.177 | 32.025 | -0.510 ms |

Representative partition-wall medians:

| Partition | Base ms | Final ms | Delta |
| --- | ---: | ---: | ---: |
| P0 | 236.854 | 231.793 | **-5.061 ms** |
| P1 | 80.847 | 79.394 | -1.453 ms |
| P2 | 113.507 | 113.066 | -0.441 ms |
| P7 | 94.680 | 95.222 | +0.542 ms |
| P8 | 123.127 | 120.783 | -2.344 ms |

The main attributable movement is P0, the partition containing the 5,479-state transient determinizaton. The difference in stage medians is larger than the P0 median reduction because other partitions and stage setup vary independently between fresh compiler processes.

### Ten-threaded

`RAYON_NUM_THREADS=10`, `GLRMASK_COMPILE_THREADS=10`.

| Metric | Base raw ms | Base median | Final raw ms | Final median | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| split terminal-DWA stage | 314.279, 304.505, 291.416 | 304.505 | 308.601, 281.709, 328.560 | 308.601 | +4.096 ms (+1.35%) |
| whole compile | 821.701, 851.600, 837.907 | 837.907 | 801.953, 782.154, 845.699 | 801.953 | -35.954 ms |

The ten-thread terminal-stage samples are highly variable: P4 alone ranged from 22.029 to 187.510 ms across these six process-level runs. This sample does **not** establish a ten-thread terminal-stage speedup. The implementation does not add synchronization or alter scheduling; the demonstrated result is the modest, repeatable single-thread reduction.

## Phase attribution and artifact shape

The normal profile already separated terminal-NWA construction, follows, postprocessing, determinization, minimization, compaction, local merge, and global merge for P0/P1/P2/P7/P8. P0 was the only dominant actual terminal-DWA construction path:

```text
P0 base diagnostic profile:
  terminal-NWA build:     13.053 ms
  always allowed:          0.641 ms
  collapse:                0.573 ms
  disallowed follows:     11.777 ms
  prune:                   2.005 ms
  canonicalize:            2.735 ms
  determinize:            48.152 ms
  minimize:               36.580 ms
  compact:                11.032 ms

P0 final diagnostic profile:
  terminal-NWA build:     11.253 ms
  always allowed:          0.554 ms
  collapse:                0.231 ms
  disallowed follows:     10.846 ms
  prune:                   1.353 ms
  canonicalize:            2.483 ms
  determinize:            39.775 ms
  minimize:               32.250 ms
  compact:                10.234 ms
```

These detailed profiles are explanatory probes rather than separately paired medians; they also show unrelated id-map movement from host variation. The reliable phase evidence is structural:

```text
base and final P0 shape:
  NWA:              2,345 states
  determinized DWA: 5,479 states / 33,917 transitions
  minimized DWA:       10 states /  2,939 transitions
  compacted ranges: 11,987 -> 5,057
```

The final determinizer probe recorded:

```text
singleton subset cache: 28,437 hits / 3,367 misses
directly emitted safe grouped edges: 1,637 labels in 724 groups
```

Thus the work removes transient materialization and lookup costs while preserving every final DWA state count, transition count, and compacted range count observed in the dominant P0 case.

Other measured costs remain outside this change:

- P2 L1 direct construction/compaction is substantial but intentionally untouched because it overlaps the other worker’s P2 profile-aggregation area.
- The direct global merge is about 32 ms single-threaded, mostly exact remap/union. It already avoids generic global determinization and minimization; it was profiled but not changed.
- P0 acyclic minimization still processes 5,479 transient states to reach 10. Its main remaining internal costs are state pushing, coloring, and reconstruction. A larger exact minimizer redesign is the next credible terminal-DWA direction.

## Correctness validation

### Rust checks and tests

Passed:

```text
cargo fmt --check
cargo check -p glrmask
cargo test -p glrmask determinize --quiet                 # 7 passed
GLRMASK_ASSERT_GROUPED_DETERMINIZE_EQUIVALENCE=1 \
  cargo test -p glrmask determinize --quiet               # 7 passed
cargo test -p glrmask --lib terminal_interchangeability --quiet
                                                          # 19 passed
cargo test -p glrmask --test integration terminal_interchangeability --quiet
                                                          # 6 passed
cargo test -p glrmask --test ti_mre_final --quiet         # 1 passed
```

### BFCL production validation

Passed with the final isolated FFI:

```text
normal production mode:
  BFCL Catalog 008
  BFCL Catalog 512

strict TI reference mode:
  GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY=1
  GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE=1

  BFCL Catalog 008
  BFCL Catalog 512

additional full-workload exactness assertion:
  GLRMASK_ASSERT_GROUPED_DETERMINIZE_EQUIVALENCE=1
  plus strict TI reference on BFCL Catalog 512: PASS
```

## Remaining bottleneck and conclusion

This is an exact, low-risk reduction in determinizer transient work. It yields a **1.54% single-thread terminal-stage median reduction** on the requested Catalog 512 workload and preserves the observed terminal-DWA artifact shape exactly.

It does not solve the principal structural bottleneck: P0 still intentionally constructs a 5,479-state transient DWA before acyclic minimization collapses it to 10 states. The strongest next line of work is an exact construction/minimization fusion or a pre-determinization quotient that handles epsilon semantics correctly. The existing token-deterministic NWA minimizer cannot be applied directly without such an exact epsilon treatment, and the tested interval pointwise minimizer fallback was slower because it built and discarded its alternate representation.
