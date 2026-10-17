# GLRMask core API and performance qualification - 15 October 2026

## Scope and outcome

This batch implements mutable **single-item** binding in Rust and Python,
shared immutable snapshots, faster construction and requirements inspection,
reusable exact-event adapters, caller-aware compilation scheduling, strict
mask-free token forcing, and braced regular-language `shortest` syntax. It
preserves the independently merged passive-marker and benchmark-documentation
changes. No package release is part of this qualification.

The functionality and correctness gates passed. Construction of larger
component assemblies is substantially cheaper in the measured fixtures. Normal
masking retains its existing implementation and the final matched run had no
repeatable-regression review flags. **This is not a claim that every operation
is faster or that all possible mask latencies are unchanged.** Remaining slower
forcing/setup cases and the limitations of the timing study are recorded below.

## Public contracts

- `Grammar.bind(name, value)` and `PreparedGrammar.bind(name, value)` mutate one
  handle and return Rust `Result<()>` / Python `None`. There is no required
  collection syntax or consuming-Rust-only protocol. A failed bind leaves the
  binding configuration intact. `clone()`, Python `copy.copy`, and
  `copy.deepcopy` create independently editable handles sharing immutable
  definitions/snapshots. Child attachment captures a snapshot; subsequent edits
  do not change parents or already returned constraints.
- Source and prepared stages retain separate input types. Prepared binding
  accepts `PreparedGrammar`, `Token`, and `ExactTokens`; it does not silently
  compile a source child or expose the legacy runnable-child tooling overload.
  Open children and occurrence-specific `LEFT::MARK` requirements remain usable
  across persistence and independent specialization.
- `extern token` remains an exact one-ID-event contract. `extern grammar`
  remains a general scoped fragment. An exact-token value may bind a grammar
  port directly, with the semantics of `Grammar.exact`; arbitrary grammars
  cannot bind a token port. Empty event sets are empty languages, not epsilon.
- Only `forced_tokens()` is public. It returns locally inevitable **original
  model-token IDs**, not a greedy encoding of determined bytes. It is read-only,
  retains every valid byte/event successor, respects root termination, and
  does not publish passive marker events. The old `forced()` and
  `forced_token_ids()` spellings are removed. There is no public byte-forcing
  method, bounded-mask query, or forced stop-reason object. `commit_bytes`
  remains a separate raw-byte recognition operation.
- Native GLRM uses `shortest { R }`, including inside regular subexpressions.
  Its prefix-minimal-language semantics are unchanged. Lark `[lazy]` still
  lowers to the same regular operation. Backticks, optional semicolons,
  maximal munch, ambiguous CFG behavior and exact event identity are preserved.
- Packed original-ID `fill_mask` remains the performance-oriented interface.
  Python's allocating Boolean `mask()` remains available as a convenience.

Construction metadata, graph certificates and the new bounded query do not
become lookups on the ordinary mask path. Native work runs outside the Python
GIL where safe; caller-owned Rayon workers are respected rather than redirected
through a small library-owned pool. This does not make conventional Python
argument/result conversion parallel or promise unlimited hardware scaling.

## Qualified builds and reproducibility

The benchmark baseline is `b255ca31d65b8543ebfdda724d1e90d07db2a7f5`, which
already includes the passive-marker work. The candidate contains that baseline,
the completed core changes, and upstream documentation merge
`ff125cfa7f7368ddf201064f30cad39aed2aff64`, plus the private alphabet-bound fix
in this qualification commit. Exact binaries, source inputs and captured
results are identified in the accompanying JSON files; a branch name alone
is not artifact provenance.

Hardware: **Intel Core i7-13620H, 10 physical cores / 16 logical processors,
Windows x86-64**. Toolchain: Rust/Cargo 1.96.1, CPython 3.12.13. Both benchmark
revisions used matched canonical Cargo release settings and `internal-api`
qualification helpers. Ordinary development opt2 binaries were not substituted
for release timing. Native runners are `examples/core_batch_bench.rs` and
`examples/core_batch_scaling.rs`; portable controllers and independent oracle
programs are in [scripts/qualification/core_batch](../../scripts/qualification/core_batch/README.md).

The [raw records](core-batch-2026-10-15/) retain every tested case, not just
improvements. Controller identities record executable/DLL hashes. The final
wheel contains the same DLL as the final forcing benchmark, with regenerated
RECORD checksums, current Python files/stubs and no inherited stale SBOM. It was
installed and tested in a separate clean virtual environment. It is a Windows
CPython 3.12 qualification wheel, **not** a claimed complete release matrix.

During qualification, switching baseline and candidate worktrees through a
shared Cargo target exposed a stale local grammar dependency. The apparent
`shortest` parsing failure was traced to an old compiled dependency, not the
candidate source. The affected dependency closure was refreshed and all final
behavior/timing gates rerun on the correct artifacts. Earlier mixed-artifact
results were rejected. Frozen baselines were preserved; no production flags
were altered to hide the issue.

A separate dependency-only Rust consumer was type-checked with the root
`glrmask` feature set empty. It exercises the mutable source/prepared interfaces,
`forced_tokens() -> Result<Vec<u32>>`, packed masks, and braced shortest syntax.
This is a compile check, not an additional runtime test count. The final source
cleanup removes only a redundant trailing blank line in `snapshot_map.rs`;
no qualified runtime logic changed.

The Maturin source distribution was built and inspected, including exact source
content and README selection; it was not compiled a second time from an extracted
archive. Its checker now normalizes CRLF/LF metadata descriptions on Windows
while continuing to reject actual content differences. Three dedicated regression
tests cover LF, CRLF, Unicode text and a real description mismatch.

## Correctness gates

| Gate | Result |
|---|---:|
| Independent public Rust consumer (`default-features=false`, no `internal-api`) | `cargo check --locked --offline` passed |
| Full Rust workspace libraries/integration targets | **2,935 passed**, 0 failed, 58 pre-existing ignored, 74 target summaries |
| Focused bounded-query tests after correcting four test-only Unicode literals | **12 passed** |
| Independently installed final-wheel Python suite | **253 passed** |
| Public wheel smoke and installed-module/DLL identity | Passed |
| Windows/LF README metadata regression checks | **3 passed** |
| README, GUIDE and concurrent-construction Rust examples | **6 doctests passed** |
| Benchmark/guide generated-content synchronization | Passed |
| Independent byte-frontier forcing comparisons | **1,902 passed** |
| Independent marker/forcing snapshots | **312 passed** |
| Scoped/nullable/ignore/Unicode/alphabet comparisons | **4,064 passed** across 88 compiled cases |
| Public wheel conformance | **200 constraints, 1,676 states, 380 forced-ID replays, 6,144 bindings passed** |
| Release event-cache concurrency/liveness | **3,840 links passed**, including 128 concurrent workers |

The last row is an oversubscription/liveness test on the stated 16-thread host,
**not a 128-core throughput measurement**. No failing test was converted into
an ignore. Full ordinary masks are the differential reference, and cold forcing
queries are performed before the test obtains a normal mask. Tests cover all
four public optimization choices, loaded artifacts, aliases and sparse IDs,
exact-only events, nullable scopes, independent ignore policies, Unicode byte
frontiers and read-only passive markers.

## Normal mask latency

The final native run used four alternating independent-process pairs, an
8,192-ID synthetic vocabulary and identical token traces. It timed
**2,880,000 mask calls per revision** across 15 case/mode combinations. Separate
captures compared **40,704 packed original-ID words exactly**, including final
masks. Both sides retain identical checked token traces and masks.

Across the final rows, median p50 ratios ranged **0.917 to 1.036** and median
p99.9 ratios **0.922 to 1.113**. The controller flagged no case where every
paired mask quantile was more than 2% slower and the paired median more than 5%
slower. That heuristic is a review aid, not a statistical or universal proof.
All per-case p50/p95/p99/p99.9/max observations are retained in
[normal-mask/summary.json](core-batch-2026-10-15/normal-mask/summary.json).

Earlier repeated runs had inconsistent tail/max flags, including an identical
baseline-versus-itself A/A control. Its raw results remain in
[normal-mask-aa-control](core-batch-2026-10-15/normal-mask-aa-control/).
The normal mask loop, mapping and dynamic-mask files have no batch changes;
private forcing entry points, rather than optional limit branches in the
normal loop, implement the new work. Source/layout review and matched timings
support retaining the existing normal masking performance. They do not justify
claiming nanosecond-identical timings on every machine/input.

This is a bounded regression suite, **not the 9,558-schema corpus** or a
serving-model benchmark. The benchmark also retains start, commit, cold/warm
compile/link, save/load and artifact-size measurements; those stages are not
silently excluded from end-to-end review.

## Construction and linking

The following numbers are median **aggregate elapsed time per job** from
matched native trials, not individual request-latency percentiles. Each row's
persistent pool is constructed outside its timer. Full results include
1/2/4/8/16 workers and all measured interface sizes.

| Native operation, one worker | Baseline | Candidate | Candidate / baseline |
|---|---:|---:|---:|
| Bind all 128 source parameters | 6079.025 us | 471.550 us | 0.078 |
| Bind all 512 source parameters | 95168.625 us | 2393.700 us | 0.025 |
| Clone prepared 128-port template and bind one parameter | 47.809 us | 2.775 us | 0.058 |
| Clone prepared 512-port template and bind one parameter | 192.197 us | 3.413 us | 0.018 |
| List 512 prepared requirements | 143.856 us | 32.303 us | 0.225 |
| Relink four exact-event parameters | 2806.025 us | 197.875 us | 0.071 |

[setup-scaling/summary.json](core-batch-2026-10-15/setup-scaling/summary.json)
contains all rows and raw trials. The improvement comes from shared interfaces,
localized snapshot updates, reusable graph facts and event artifacts, not just
changing the return type of `bind`. A requested full requirements list still
costs at least its output size.

**Small cases are not uniformly faster.** For example, a one-slot source clone
and bind at one worker measured 1.244 -> 1.416 us. Several one-slot parallel
clone/bind rows also became slower: the 16-worker prepared case was
0.325 -> 0.475 us aggregate/job. These are retained alongside larger-template
wins rather than removed as inconvenient data.

There are also cold lifecycle trade-offs. In the normal-run fixture, first
compilation of a scoped child measured 11.20 -> 17.33 ms in Balanced and
13.69 -> 19.48 ms in FastRuntime, while separate prepared linking improved.
These small cold samples are not a universal cost model, but prevent claiming
that the batch accelerates every entry point. The complete stage measurements
remain in the normal-mask summary.

## Strict forcing

Three alternating process pairs measured 42 case/mode/cache combinations with
2,000 repeated queries per row: **252,000 queries per revision**. Every compared
forced-ID list matched. The baseline operation is its strict-ID method, not
its greedy canonical-byte encoding. First-query timings are recorded before
warming a normal mask; existing-mask queries are a separate workload.

Representative cold-state p50 values (us):

| Case | Mode | Baseline | Candidate |
|---|---|---:|---:|
| Broad identifier, no forced ID | Balanced | 42.7 | 4.5 |
| Exact event | FastRuntime | 71.4 | 4.6 |
| Nested inline grammar | FastBuild | 47.7 | 25.8 |
| Nested inline grammar | FastRuntime | 79.8 | 11.2 |
| 24-token singleton chain | Balanced | 70.7 | 52.5 |
| 24-token singleton chain | FastRuntime | 159.3 | 49.6 |
| 24-token singleton chain | FastBuild | 42.2 | 102.9 |
| Two singleton characters | FastBuild | 13.6 | 17.0 |

**The last two rows are remaining regressions**, not hidden successes.
Exact no-full-mask singleton proof and complete successor replay can cost more
than the old optimized mask-based query on some FastBuild states. There is no
claim of uniformly faster forcing. The normal masking method is independent
of this trade-off.

Qualification caught a larger nested FastBuild regression (about 307 us). A
private conservative lexical-alphabet bound eliminated impossible candidate
bytes before expensive proof, reducing it to the 25.8 us result above. The
bound includes participating child lexers, declines unsupported/large
representations conservatively, and never treats a provisional lexer candidate
as an admitted original ID. All exact-event and root-control checks remain
separate. The 6,278 independent comparisons above were rerun after this fix.

[All forcing rows, paired ratios and first-query observations](core-batch-2026-10-15/strict-forcing/summary.json)
are preserved. The implementation does not build a full mask simply to test
cardinality; it may read an already-existing exact mask. Singleton/zero proofs
can still need exhaustive work, so early termination is not a universal
complexity guarantee.

## Scaling limits and release status

Caller-aware execution removes avoidable redirection of jobs already running
in caller-owned Rayon pools. Heavy native Python construction work releases
the GIL, while argument conversion, wrapper publication and ordinary Python
object work remain subject to Python's execution model. No graph-validation
locks or new environment lookups were added to normal masking.

Real throughput qualification here reaches **16 logical workers on 10 physical
cores**. The separate 128-worker test establishes liveness/correctness only.
Actual 32/64/128-core throughput, multi-socket NUMA behavior and the full
multi-platform Python wheel matrix remain **untested by this local run**.
Future results must distinguish total throughput from per-job service latency
and hardware saturation rather than infer linear scaling from thread count.

The canonical runtime, installed wheel, ordinary masks and returned token
traces were qualified before integration. No PyPI/crates.io package or release
tag is published by this work. CI status is verified separately for the final
Git commit; the local checks above do not pretend to be remote CI results.

## Final integration audit

The final integration check reverified all 396 recorded source files, the wheel
RECORD and runtime DLL identities, and the installed Python sources/type stubs.
The only post-qualification Rust edit removed a redundant trailing blank line
from snapshot_map.rs; comparison with the archived qualified source confirmed
identical non-whitespace content. No runtime code or qualified binary changed.
The clean-wheel smoke test and all 253 Python tests passed again, together with
metadata regression checks, generated-document synchronization and the final
diff check. See [the final audit receipt](core-batch-2026-10-15/checks/final-integration-audit.json).
