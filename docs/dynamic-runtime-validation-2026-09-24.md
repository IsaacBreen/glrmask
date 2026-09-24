# Dynamic runtime correctness and default word routing

## Decisions

The virtual-residual slice totality correction is unconditional. A completion
relation records existing transitions; it does not establish that every input
atom has a transition. Invariant repetition now requires the existing universal
one-atom proof to succeed before extrapolating a self-loop.

Finite optional-space/nonspace word slicing is enabled by default. Its default
regular language uses a bound of 16 nonspace characters. The existing exact
parser-transparency proof must complete successfully, and the estimated residual
walk must contain fewer than half the operations of the already-certified
master route (or the ordinary route when no master certificate exists).
Otherwise the runtime retains the existing exact route. Neither the cost gate
nor a failed proof can admit a token.

`GLRMASK_DISABLE_DIRECT_RESIDUAL_WORD_SLICE=1` is a diagnostic opt-out for
same-binary comparisons. No experimental opt-in is necessary. Existing explicit
diagnostic length, proof-work, and margin overrides remain available.

Unconditional eager containment-quotient preparation is **not** selected as the
default. Earlier controlled full-population experiments found useful tail
improvements but approximately 8–10% additional mean schema-build cost. This
tradeoff is distinct from default word routing, which adds no schema-specific
compilation pass.

Automatic module linking also checks whether a parent or any nested compiled
child contains virtual residual coordinates. Such components require the
existing dynamic boundary implementation. Backend support is selected from
retained metadata before construction; this is not a caught-error fallback or
a request to recompile an attached source grammar.

## Correctness evidence

The lexer regression fixtures demonstrate the original failure with a missing
multibyte body atom and with a separator following the final permitted word.
Both failed against the original invariant shortcut and pass with the totality
precondition. The current-main lexer release suite passes all 231 tests.

The public API and independent API-review suites pass 22 and 19 tests. The new
`word_limit_totality` target checks public default/fast-build/fast-runtime
compilation, saved and loaded constraints, Unicode and canonical escaped-string
continuations, and compiled-child binding through saved and loaded modules.
Arbitrary Unicode escape aliases are not assumed to belong to the default
canonical JSON serialization language.

Online paired-mask checks of the integrated production-default binary compare
complete masks against a same-binary word-disabled control. They discard each
pair immediately and never save masks. Results: 22,436 identical frames on the
affected owners, 13,616 on the weak-tail cohort, and 18,537 on a held-out cohort;
zero differing token bits. Cohorts overlap and are not counted as disjoint
prefixes. Instrumented mask-stream timings are not performance measurements.

## Benchmark control and reproducibility

The earlier apparent long-process slowdown was caused by Windows scheduling on
different processor classes, not demonstrated persistent library cache
contamination. Fresh runs were observed on a performance core and long-history
runs on efficiency cores. Holding core class fixed removed the large gap.
Cache eviction and runtime repackaging experiments must not be shipped as a
solution to that scheduling artifact.

All current Windows comparisons use the same verified performance-core affinity
(`0x0fff` on the measured machine), wall-clock timing, and allocator policy for
every framework and variant. Affinity is a benchmark-process setting, not a
library change or machine-wide setting. CPU masks must be derived from topology
on other machines, not copied blindly.

The integrated baseline is `f8d478037`. The no-experiment-flags runner is frozen
under `C:/gdt22/artifacts/release-600919-20260924`, with source and dependency-lock
snapshots. Its initial binary SHA-256 is
`6ddc5f1ce1f580f6e9422576c43a190d1d77b2a9ebf9345e1d69ddfdd2266ea0`.
The source patch SHA-256 is
`5c4511591ee7cf242a9c7ebab872aa47c4e1de419455ff13c929f3ce6140bd76`.

The driver retains raw per-position scalar timings, process wall times,
commands, environment overrides, and corpus/vocabulary hashes. It requires
exact problem/example/outcome and timing-array alignment. Arrays are never
truncated to manufacture a match. Stabilized TBMs are explicitly per-position
minima across independent runs; schema builds are per-schema medians. Raw
process maxima remain available separately.

The Dynamic population of 8,326 schemas contains 2,932,617 timing positions. It
is not the historical 2,916,598-position cross-framework matched population.
Cross-framework comparisons require fresh matching under the same protocol.

## Remaining limitations

The earlier full correctness audit repaired ten invalid-example commit errors
without changing any valid-example outcome. It also retained 121 pre-existing
disagreements with corpus labels: 90 labelled-valid examples rejected and 31
labelled-invalid examples accepted. These have not all been classified as
implementation errors, intended serialization/schema limitations, or corpus
label errors. Build success alone is never reported as complete correctness.

## Integrated current-main validation

The complete root library suite passes 795 tests, with 51 existing tests
ignored and no failures. The public regression also passes after nesting a
saved/loaded linked virtual child inside another compiled host.

The current-main no-experiment-flags comparison completed on all 8,326 schemas,
31,934 examples, and 2,932,617 exactly aligned timing positions. Two independent
processes per configuration were run in alternating order. All outcome
signatures match both the word-disabled control and the previously corrected
baseline: 31,813 correctly labelled outcomes, the same 121 known disagreements,
and zero runtime errors.

| Metric | Word-disabled control | Default word route |
|---|---:|---:|
| p50 TBM (ms) | 0.0058 | 0.0058 |
| p90 TBM (ms) | 0.0562 | 0.0561 |
| p99 TBM (ms) | 0.1745 | 0.1746 |
| p99.9 TBM (ms) | 0.5995384 | 0.5989 |
| p99.95 TBM (ms) | 0.9364076 | 0.9020076 |
| p99.99 TBM (ms) | 2.09997384 | 2.08637384 |
| p99.999 TBM (ms) | 3.47084284 | 3.325669536 |
| Stabilized maximum TBM (ms) | 4.1223 | 3.8380 |
| Mean schema build (ms) | 2.5094613 | 2.5210528 |

These results support enabling the exact, profitability-gated word route,
not claiming uniform improvements at every percentile. The measured mean
build difference is approximately 0.46%; this is near-neutral, not proof of
zero overhead. A per-schema screen found one p99 increase above both 10% and
0.05 ms (`o8475`, 0.21529 to 0.26737 ms); it is retained for focused rechecking.

The full result directory is
`C:/gdt22/artifacts/release-600919-20260924/full-20260924-102437`.
This minimal current-main integration deliberately excludes the unmerged S50
experimental patches, so it must not be equated with the earlier experiment's
larger maximum reduction. Those remaining optimizations require their own
review and integration on current main.

Default word routing does not by itself establish strict p99-to-p100 dominance
over LLguidance. Remaining cold proof construction, virtual residual walks, and
schema compilation costs are measured separately rather than hidden outside
their timing intervals.
