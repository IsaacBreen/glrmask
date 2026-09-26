# Static boundary terminal construction: validated profile

## Scope and activation

This checkpoint consolidates the accepted boundary compiler through N70, the subsequent terminal-side optimizations through P11, and K01's exact sole-class minimizer shortcut. It does **not** claim that the complete pre-parser stage has reached its 10 ms objective. It is also not a new LR/template/cancellation algorithm.

The measured configuration is explicitly opt-in. The complete selector set is checked in at `scripts/profiles/boundary-link-validated-20260926.json`; merely building the repository does not enable every selector in that file. Several selectors configure the previously accepted parser path as well as the terminal path, so this is a reproducibility profile for the whole benchmark, not a new public compiler API.

Use the launcher to avoid inheriting stale experiment, validation, or dump switches:

```sh
python3 scripts/run_boundary_profile.py --threads 10 --dry-run -- \
    target/release/examples/composition_build_static_artifact PREPARED_CACHE output.bin

python3 scripts/run_boundary_profile.py --threads 10 -- \
    target/release/examples/composition_build_static_artifact PREPARED_CACHE output.bin
```

The launcher modifies only its child environment, does not use a shell, and leaves unrelated environment variables intact. It removes inherited `GLRMASK_*`, `PROBE_*`, `PROFILE_*`, `PHASE_*`, and `DYNAMIC_REFERENCE*` switches before installing the recorded profile. `--threads` sets the child's Rayon count; when omitted, an existing `RAYON_NUM_THREADS` value is retained. Many Rust switches are presence-based: setting them to `0` is **not** equivalent to unsetting them.

The existing selected10 example expects `core.bin`, `dispatch-literal.bin`, and `vocab_dump.bin` in `PREPARED_CACHE`. It composes the `PROGRAMMATIC_TOOL_SUFFIX` slot. That fixture and its prepared inputs are not supplied by this profile. The separate `composition_prepare_component` example prepares one component without a future partner; preparation and certified reload costs must be accounted for separately from link latency. Old component metadata remains supported by conservative fallback.

## Exactness boundary

A change is not accepted merely because it preserves the completed weighted terminal language. The existing downstream parser can observe guard structure and arbitrary stack prefixes. The retained contract is stronger:

* Preserve tokenizer-state/token correlations, raw coordinate maps, longest-match observations, and ignore/follow/crossing behavior.
* Compare complete emitted terminal graphs and maps where the implementation promises graph identity.
* Check the existing original-coordinate arbitrary-prefix parser oracle, not only reachable runtime histories.
* Replay all 11,767 steps in both the static and dynamic reference histories for the selected10 fixture.

The native representation has explicit eligibility and resource limits. Unsupported token domains, virtual lexer shapes, unproven mappings, cycles, or resource exhaustion decline to the existing constructor rather than publish partial results.

## Accepted terminal-side construction

The expensive intermediate coefficient operations now retain raw tokenizer-state intervals and finite token bitsets through emission, follow/crossing postprocessing, determinization, and minimization. Generic weights are reconstructed for the small published result, rather than repeatedly constructed and imported between phases. This retains every correlation; it is not a Cartesian approximation or a token quotient.

The finite query lexer uses the same logical state mapping as the materialized observation view while borrowing immutable original lexer observations. The selected fixture uses 2,587 query states and 32 byte values, so its compact byte table contains 331,136 bytes instead of a state-by-256-byte table. Ordered scanner outputs are checked against the original execution path. Seed nodes are emitted directly into the native representation in the existing order.

K01 applies after the ordinary support push and exact structural class formation. When a non-leaf height contains one exact class, there is no compatibility decision to make. Copying its representative with already mapped destinations is the identity of the previous profile-expansion/reconstruction step. The optimization preserves the original logical resource budget; it does not assume that almost-uniform classes or final-DWA symbol equivalence permit early merging.

## Measurement boundaries

The acceptance metric is the existing `boundary_component_lane` **child `build_ms`** timer. It includes the selected child's prefix filtering, query mapping, terminal construction, postprocessing, finalization, and cleanup. It excludes subsequent parser-DWA consumption. Shared link setup, the parent lane, artifact loading, independent component preparation, and artifact serialization are separate costs and must not be silently counted as zero.

Consequently, a 14–16 ms child measurement is neither a 14–16 ms whole link nor a measurement of every pre-parser operation in both lanes. The 10 ms target for this measured child stage remains outstanding.

Historic matched measurements on the Mac:

| Checkpoint | Ten-thread child build | One-thread child build | Interpretation |
|---|---:|---:|---|
| P11 first screen | 17.261 → 15.9845 ms | 15.462 → 14.406 ms | Borrowed query view and direct native seeds |
| P11 independent confirmation | 17.904 → 16.392 ms | 15.109 → 14.201 ms | Same frozen binary, new fixed sample blocks |
| K01 first screen | 16.7255 → 15.7895 ms | 14.581 → 14.3315 ms | Exact sole-class reconstruction shortcut |
| K01 independent confirmation | 16.6705 → 15.8475 ms | 14.591 → 14.3875 ms | Intrinsic one-thread saving is small/noisy |

P11 retained 96 fresh calls and improved all 24 paired blocks. K01 retained 80 fresh calls; its combined median paired improvement was 0.90325 ms at ten threads and 0.19425 ms at one thread. Do not add improvements from different cohorts or attribute cross-session baseline drift to a patch. The uninstrumented whole-link data and parent/setup timings are retained separately in the local evidence.

## Development-branch integration (`glrmask-main`)

Integration starts from accepted K01 `910a55e516d7d26d999e95f439144f2338fd4694` and the active development branch `glrmask-main` at `e24e24c63`, preserving the latter's `UnlinkedConstraint` API migration and dynamic proof fixes. The validated work is merged and pushed to **`glrmask-main`**, which is checked out in the primary development worktree. GitHub's separate default branch `main` is an older, divergent line and was not modified by this task.

The textual conflict was between independent test/helper modules appended to `src/runtime/artifact.rs`; both were retained. A compile-time incompatibility in a newly added pointer-identity check was resolved by comparing `constraint.tokenizer.as_ref()` with the cached lexer pointer, matching the accepted shared-tokenizer ownership representation. This is an ownership adaptation, not a changed cache policy.

The merged checkpoint is `14047e9c7`. Validation completed on the merged source, not just on the individual branches:

| Suite | Passed | Ignored | Configuration |
|---|---:|---:|---|
| Terminal compiler | 272 | 0 | Both default and accepted profile, separately |
| Root library | 892 | 51 | Both default and accepted profile, separately |
| Lexer | 236 | 0 | Default |
| GLR | 196 | 2 | Default |
| Parser DWA | 46 | 1 | Default |
| Weighted automata | 54 | 0 | Default |
| JSON Schema | 48 | 0 | Default |
| Weight algebra | 21 | 0 | Default |
| Current public API and targeted regressions | 45 | 0 | Default |
| Profile launcher | 5 | 0 | Python standard-library tests |

All these suites completed with zero test failures. The root harness invokes several one-test subprocesses; those are already included in its 892 outer tests and must not be counted again. The Python binding also passes `cargo check --release -p _glrmask`; this is a Rust binding type-check, not a claim that wheel installation or Python runtime tests were run.

The merged implementation reproduces the accepted K01 artifact **byte-for-byte at each respective thread count**:

| Threads | Artifact bytes | SHA-256 |
|---|---:|---|
| 10 | 47,743,973 | `3f16b7149a1f000c8a4cedf5385ddf5e4d8b6422692f8cd0017da8baab638f60` |
| 1 | 47,743,805 | `84c1c90f2b7e7b519fb1aac0f0a29851c0fe5fd19c17d773a60295e6b6fd0be6` |

These artifacts are not asserted to be identical across thread counts. At both settings, complete terminal graphs/maps match K01 and the unchanged original-coordinate arbitrary-prefix parser comparison reports no difference: 8 product states / 34 compared branches for the parent, and 1,198 / 236,650 for the child. Both static and dynamic 11,767-step histories passed at each thread count, giving four full replays. The legacy prepared-component fallback still produces SHA-256 `8eb6e8a25dcf53fe63ecc1cd5a29b97c17438c198c1c1233c10a1a59a9925ab4`.

A fixed 48-call alternating comparison of frozen K01 and the merged binary used identical prepared inputs, accepted flags and coarse composition instrumentation. Every sample was retained:

| Threads | Frozen K01 child median | Merged child median | Median paired difference, old minus merged |
|---|---:|---:|---:|
| 10 | 16.1325 ms | 16.0085 ms | −0.03525 ms |
| 1 | 14.5330 ms | 14.5975 ms | +0.02625 ms |

The small paired differences do not establish a meaningful terminal performance change from main integration. The merged child remains approximately **16.0 ms / 14.6 ms**, above the 10 ms objective. The corresponding coarse whole-link medians were 268.4555 ms / 305.0265 ms; these are not uninstrumented latency claims and include the separately owned parser work.

One initial merge build failed on the pointer type mismatch described above and was corrected before testing. The additional library harness initially supplied an `internal-api` feature to the weighted-automata package, which has no such feature; Cargo rejected the command before tests. The command was corrected only for that package and all remaining suites passed. Both unsuccessful attempts are retained in the evidence, rather than silently omitted.

## S01: buffered vocabulary fingerprinting

After integration, a separate shared-setup investigation found that the first cached candidate-summary lookup still computed the full vocabulary digest. It issued three small BLAKE3 update calls per model token: token ID, byte length and token bytes. S01 batches the **identical ordered byte transcript** through a 4 KiB buffer; large byte strings pass directly to the hasher. The digest algorithm/version, canonical field encoding, cached value and cache lifetime remain unchanged. No hashing work is moved outside the measured link.

The additional switch `GLRMASK_BUFFERED_VOCAB_DIGEST=1` is included in the checked-in validated profile. It is not enabled by default outside that profile. This improves a fresh vocabulary's first digest computation; an already cached digest was already cheap and does not gain another four milliseconds on every reuse.

Two new independent test functions cover 97 vocabulary cases and 13 update-boundary sizes, comparing both the original incremental calculation and a separately assembled canonical byte transcript. The full root suite now passes **894 tests, zero failures, 51 ignored**, under both default and accepted-plus-digest-validation configurations. Both thread-count artifact/mapping/arbitrary-prefix gates, all four 11,767-step replays, and the historical legacy artifact remain exact. Only `src/compiler/compile.rs` changes executable source for S01; all other Rust source is hash-locked.

The first 48 coarse-profiled calls and an independent fixed 32-call confirmation retained all samples:

| Threads | Shared setup, first screen | Shared setup, confirmation | Combined median paired saving |
|---|---:|---:|---:|
| 10 | 23.9260 → 19.6325 ms | 23.8190 → 19.5760 ms | **4.3135 ms**, 12/12 blocks positive |
| 1 | 24.9210 → 20.4990 ms | 24.4270 → 20.5820 ms | **4.21125 ms**, 8/8 blocks positive |

A separate fixed 32-call, **uninstrumented whole-link** comparison measured ten-thread medians 253.7435 → 250.7590 ms, with median paired saving 2.54775 ms and all four blocks positive. One-thread medians were 294.3740 → 289.9330 ms, but noisy paired blocks gave only 0.04275 ms median saving and two of four positive; that run does **not** establish a reliable one-thread whole-link improvement. The consistent claim is the measured shared-setup reduction, not a universal end-to-end saving of the same size.

The child terminal build is not changed by this optimization and remains approximately 16 ms / 14–15 ms. S01 therefore **does not achieve the child-stage 10 ms target**. Its evidence, exact source/binary manifests, failed-attempt history and raw timing rows are retained in `.benchmarks/finalize-761063/s01-buffered-digest/`.

## Final release checks and cleanup

The final S01 source also passes the 45 public API/regression cases with the accepted profile and hash validator enabled. Both one-thread and ten-thread invocations through the **checked-in launcher** exercise the exact 128,256-token digest validator and produce the expected artifact sizes. The Python Rust binding check passes again on S01. The root documentation example also passes its doctest.

The remaining internal-crate suites were checked as part of closure: artifact 15, grammar 92, vocabulary 3, invariant 0, DWA merge 13, and finite automata 6 tests, all with zero failures. Across the validated suites this gives **1,941 Rust test cases**, plus **one root doctest** and **five launcher tests**; 54 Rust cases are ignored. Repeated configurations and subprocess summaries are not counted as new tests. The detailed final summary is `final-validation-summary.json` in the preserved integration evidence.

The source worktrees are left clean. Failed experiments are not installed. Cleanup is limited to byte-and-hash-verified duplicate immutable artifacts, with every original path retained, and reversible compression of this task's completed test executables. Source snapshots, raw timing rows, test logs, accepted binaries, old control flags and rejected variants remain available. Other workers' worktrees and caches are not removed. Publication records separately verify the exact local and remote `glrmask-main` SHA; no force push is used.

## Rejected directions and remaining work

The reverse suffix token-filter solver, its memoized variant, and the streaming forward-prefix solver all passed their exactness checks but lost full-stage performance. They are not enabled or retained as production source. Reducing byte-cut visits increased terminal-transfer or frontier-maintenance work. Direct native token hashing and forward-domain identity certificates likewise failed to establish a worthwhile total-cost improvement.

The remaining opportunities are in exact native event generation, postprocessing and construction, or reusable component preparation that removes rather than relocates hidden link work. Representation changes must retain the stronger downstream contract. Early LR equivalence, template expansion, cancellation and parser normalization remain a separate workstream.

Local experiment sources, failed cases, full commands, hashes, and raw samples remain in the preserved worktrees. The integration evidence is `.benchmarks/finalize-761063/` in the `glrmask-terminal-finalize-761063-20260926` worktree. Earlier accepted evidence lives under the prepared-307413 and kernel-0454 worktrees. These local evidence directories are deliberately not bundled into the source repository or crate.
