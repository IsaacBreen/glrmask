# Dynamic build optimization and crash recovery: 26 September 2026

## Release scope

The measured production tree is `7d0f388ffed2a997a4db6018a5f9f8cc619e8406`. It extends the previously published `e24e24c63664a27d26c980527439a8dea36d2a3b` by five validated commits:

| Commit | Change |
| --- | --- |
| `45002620a` | Reject ineligible observation futures before scanning byte rows. |
| `19d3c0a52` | Certify globally physical scalar lexer components without graph scans. |
| `02482137b` | Keep exact component lexers by default for Dynamic compilation instead of eagerly determinizing their union. |
| `2279c55a9` | Detect unresolved recursive schema specialization while preserving completed-memo ordering and lookup precedence. |
| `7d0f388ff` | Return projected L1 resource exhaustion as a compilation error instead of terminating the process. |

The development integration target is `glrmask-main`, the branch used by the current project worktrees and prior publications. The separately named repository-default `main` is a divergent historical line; this release does not overwrite or silently merge that line.

The finalization changes above the measured tree are validation documentation and migration of remaining public test/guide references from the removed `Module` / `compile_module` API to `UnlinkedConstraint` / `compile_unlinked`. They do not change production Rust source, dependencies, runtime policy, or the measured executable. The earlier API migration is not reversed by adding obsolete compatibility aliases.

## The pathological maxima were memory-pressure effects

The original Renovate import repeatedly expanded an unresolved recursive specialization. Increasing its already-large stack delayed, but did not solve, the failure. A captured original run reached approximately 6.42 GB of private committed memory and reduced available physical memory to approximately 366 MB. Retained Windows traces associated later 20-30 ms token intervals with hard page reads for the executable's code and read-only data. A fresh child process could therefore be affected by memory pressure created by its predecessor.

This was not an elementwise-minimum aggregation error. An independent audit of retained operands confirmed the same unusually slow positions in multiple full-corpus repetitions. Fresh isolated target replays did not reproduce the same stalls. These controls and the traces distinguish the actual observations from a universal claim that every crash must cause a later stall.

The fix removes the runaway allocation and process failure. It does not change clocks, add image prefetching, add sleeps, trim observations, substitute replay timings, or give the candidate extra warmup.

## Importer guard semantics

Completed memo entries retain the original owned canonical key, full typed collision equality, append-on-completion order, and first-match lookup. Completed results are checked before active recursion. In-flight states are kept in a separate LIFO stack; they never supply grammar expressions or alter which completed entry is chosen.

An equal unresolved canonical state revisited without additional reference bindings now returns an explicit unsupported-recursive-specialization error. New bindings count as progress, so ordinary symbolic recursive schemas remain supported. Normal error returns pop their active frames. This is a fail-closed robustness guard, not support for arbitrary recursive intersections or a universal termination proof.

The original Renovate case remains unsupported. Previously successful corpus cases retain their support, outcomes, and token trajectories. Earlier guard candidates that changed the accelerator rejection position were rejected and are not in this release.

## Static resource-limit semantics

The Static Kubernetes `kb_1161_Normalized` case exceeded the existing projected L1 limit: 2,643,046 projected states versus a configured 2,000,000. The same limit is enforced, but exhaustion now travels as a distinct typed resource-limit payload to the existing compilation error boundary. The caller receives a compilation error, never an empty-language substitute or partial artifact.

The resource payload is distinct from internal-invariant failures. Unrelated panic payloads continue to propagate, and normal unwinding runs destructors. This follows the existing unwind-based compiler boundary; it does not promise recovery under `panic=abort`. No new cyclic fallback, raised resource limit, or broad panic-string suppression was introduced. Repeated failure / successful compilation sequences were tested in one process.

## Canonical rerun

Report: `final-jsb-guard-resource-20260926`.

Executable SHA256: `6127ee209fb0e558642df1eea90403a4ba82ef17e95154ee57f51d7e16c628cd`.

Dynamic, Static, and O2 each used two independent full 9,558-schema Windows processes from that same executable. All six completed without a process crash or restart. Actual native child affinity was verified as `0x0fff`. Only llguidance was reused from the September 24 Windows report, with its raw bytes hash-verified unchanged. This is not a contemporaneous rerun of llguidance.

Both independent raw operands are retained. Every TBM and initial-mask elementwise minimum was independently checked. Build and time-to-first-mask (TTFM) retain the canonical first-run accounting; the separate experimental ABBA medians are not substituted into this report.

The exact comparison population is 8,326 common schemas, 30,788 examples, and 2,925,983 matched token positions per engine. Historical trajectory matching is unchanged.

| Engine | TBM p50, us | p99, us | p99.9, us | p100, us |
| --- | ---: | ---: | ---: | ---: |
| GLRMask Static | 2.7 | 11.5 | 18.9 | 40.4 |
| GLRMask Dynamic | 5.6 | 149.5 | 560.5 | 3,145.4 |
| GLRMask O2 | 6.7 | 59.6 | 163.802 | 496.2 |
| Archived llguidance | 8.1 | 275.0 | 893.302 | 6,979.8 |

All 29,260 audited empirical percentile points from p99 through p100 have Dynamic below llguidance. The largest Dynamic/llguidance ratio is 0.7914498982. This is rankwise distribution dominance on these samples, not a per-position guarantee or universal latency bound.

Dynamic common-schema TTFM is 1.5459 ms at the median, 4.463 ms at p90, 20.345575 ms at p99, and 120.5358 ms maximum. Archived llguidance is 0.5762 ms at the median and 6.5994 ms at p99. The build/TTFM parity goal is therefore still open; a runtime-tail win must not be represented as completion of all build goals.

## Correctness and source identity

The measured combined tree passed 799 root-library tests, 51 importer tests, and four resource-payload tests, with zero failures; 51 pre-existing root tests were ignored. The all-supported Dynamic differential compared 3,471,762 complete masks across 9,326 schemas, with zero differing frames, identical outcomes and trajectories, and aligned labels. No full-mask payloads were persisted.

Static and O2 were checked through targeted native cases and full native outcomes/trajectories. A full-corpus bitwise mask equality claim is made only for Dynamic. Existing corpus-label disagreements remain; neither pairwise agreement nor support counts constitute universal JSON Schema conformance.

The compact machine-readable evidence is [dynamic-build-and-crash-validation-2026-09-26.json](assets/dynamic-build-and-crash-validation-2026-09-26.json). It records input and raw hashes, the independent-run manifests, matched population, quantiles, and the measured-tree correctness gates. Original raw timing files and diagnostic traces remain in the retained report/artifact archive; they were not rewritten during finalization.

## Experiments deliberately not enabled

Streaming memo-key hashing, reusable serialization buffers, borrowed canonical keys, and fused canonical cloning were investigated separately. Their fixed pilots or full paired comparisons did not establish a useful broad production improvement. Their source checkpoints and negative results are retained; they are not dependencies of this release.

Lowering the sparse optional-object threshold from 64 to 16 improved the observed paired build maximum (113.478 to 88.924 ms), but mean build improved only about 0.54%, build p99 worsened, and matched TBM p99 increased from 148.4 to 170.0 us. Generated runtime coverage for previously untraced build winners also exposed costs. The production threshold remains 64. Four mask differences in that experiment were independently traced to prefixes of an impossible optional field and recorded explicitly; they were not relabelled as zero differences.

## Reproduction and continuation

Use the pinned input hashes and the same canonical final-JSB protocol, retaining both independent repetitions and the first-run build accounting. The report's A-D plots were delivered with verified notification receipts before finalization; no new notification is needed for documentation/test-only commits.

Local validation command:

```sh
cargo test --release --locked -p glrmask --tests -- --test-threads=4
```

Further work should target construction cost in runtime-artifact preparation, containment proofs, parser/lexer critical paths, and grammar representation rather than deferring eager work into the first mask. The importer-only and buffered core ledgers are diagnostic tools, not replacement publication timings. New representation policies need representative traces for schemas without corpus examples, preserved correctness evidence, and fresh build/TBM trade-off measurements.

## Finalization gates and retained experiments

The final release candidate, with production code identical to the measured tree, passed all 29 registered root/integration/regression executables: **1,062 tests passed, zero failed, 51 pre-existing ignored**. This includes 799 root-library tests and 263 registered integration/regression tests. Nine nested filtered child-process summaries were excluded from that total. The GLRM suite retains all 21 original tests and assertions after its public API migration.

The additional Rust workspace library gate (excluding the Python binding crate), all public-example compilation checks, and root documentation tests also passed. The JSON evidence records each actual suite and log hash; overlapping root tests are not presented as independent extra coverage. This finalization did not rebuild or retime the canonical publication binary, since production Rust source and dependency locks are unchanged.

The source of the negative/neutral experiments is preserved on remote `archive/*-20260926` branches, including `archive/importer-buffer-hash-20260926`, `archive/importer-borrowed-key-20260926`, `archive/importer-canonical-clone-20260926`, `archive/importer-phase-ledger-20260926`, `archive/importer-sparse-object-20260926`, and `archive/core-artifact-ledger-20260926`. Their exact commits are in the JSON evidence. They are intentionally not merged into the production branch.

The separate CFA benchmark repository now preserves independent timing operands and failure evidence by default at commit `1683b0b4eec6cef5c710855f3da7043d05052131`. Its 20 regression tests and a native 81,921-position elementwise-minimum audit passed. This harness repair keeps the same timing aggregation and first-run build/TTFM semantics; the existing publication report remains pinned to the harness revision that actually produced it.
