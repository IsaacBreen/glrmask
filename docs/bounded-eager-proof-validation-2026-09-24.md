# Bounded eager proofs and build preparation

## Default decision

Ordinary Dynamic compilation now prepares exact scalar containment proofs by
default. Each physical component is limited to 3,000 states and 768,000 byte
edges during preflight. Independent components and terminal proofs use the
existing Rayon batch builder. An oversized, unsupported or unprovable component
does not receive a certificate; masking continues through the exact walker.

Set `GLRMASK_DISABLE_EAGER_CONTAINMENT_QUOTIENTS=1` before process startup to
disable this ordinary-Dynamic default. `GLRMASK_EAGER_CONTAINMENT_MAX_STATES`
overrides the ordinary component budget. Static and vocabulary-quotiented O2
retain their existing proof-preparation policies. Historical experiment flags
are not required. This does not introduce an approximate mask or a fallback
that indiscriminately admits the vocabulary.

The accompanying build optimizations use sorted contiguous LR(0) kernels and
exact-size transition iterators. Dynamic/O2 compilation additionally removes
reference-unreachable generated rules before processing their regex bodies.
Named-grammar inspection APIs still return their original complete definitions;
the optimization is confined to a hidden compiler-only entry point. Existing
public grammar/partition/provenance assertions were retained unchanged.

The earlier virtual-proof totality correction and default word16/500-permille
route remain enabled. General integer divisibility is enforced by the separately
documented exact remainder DFA, including exact finite-range arithmetic.

## Controlled validation

The Windows comparison used one frozen executable, actual P-core affinity
`0x0fff`, wall-clock runtime timing, and identical vocabulary/corpus bytes.
There were two independent processes per full configuration, with rotating run
order. Runtime figures below are per-position minima across those processes;
build figures are means of per-schema medians. They are not raw process maxima.

Frozen source: base `82f8d1f45512c460b9363af38adf57de68423ff8` plus patch
`b7c573c88066ef6fecc9a72612a5201bfe2899cc1e40cf1628a3032ea52dc20d`.
Executable SHA-256:
`342d9dd28f1753e68c30f38e29558aa8f4cd85218075200720a59df051cd6406`.

All configurations built 8,326 schemas and evaluated 31,934 examples at exactly
2,932,614 aligned mask positions. Outcomes and position lengths matched exactly:
31,816 correct against the existing labels, 118 pre-existing disagreements,
zero runtime errors. No labels were changed and no schemas were removed for
this comparison. The numeric fix accounts for three recovered invalid examples
relative to the previously published word-routing checkpoint.

| Metric, milliseconds | Build-reference control | Optimized, eager off | New defaults |
|---|---:|---:|---:|
| Mean schema build | 2.525630 | 2.486122 | 2.738077 |
| TBM p50 | 0.0057 | 0.0058 | 0.0057 |
| TBM p90 | 0.0562 | 0.0563 | 0.0563 |
| TBM p99 | 0.1739 | 0.1739 | 0.1597 |
| TBM p99.9 | 0.594239 | 0.595239 | 0.5696 |
| TBM p99.95 | 0.897969 | 0.900569 | 0.755769 |
| TBM p99.99 | 2.1504 | 2.165622 | 2.1447 |
| TBM p99.999 | 3.384717 | 3.395167 | 3.383672 |
| Stabilized maximum TBM | 3.9024 | 3.8376 | 3.5615 |

The reference disables eager proofs, early rule pruning and flat kernels; the
iterator implementation is common to all three configurations. The combined
default costs approximately 8.4% more mean build time than that reference, or
10.1% more than optimized eager-off. Thus the build work offsets part, not all,
of the eager preparation cost. This is an explicit latency/build tradeoff,
particularly relevant when a compiled constraint serves multiple sequences.
It does not establish faster end-to-end execution for every one-shot workload.

The targeted 16-schema/81-example gate used three processes per configuration.
All 29,281 timing positions and outcomes matched. Independent complete-mask
comparisons against the build-reference control found zero differing bits on
the affected nine-schema cohort (22,436 frames), weak10 (13,616 frames), and a
held-out 64-schema sample (18,537 frames). These cohorts overlap. Each pair was
compared and discarded immediately; zero mask bytes were persisted.

Tests passed: 795 root library tests with 51 existing ignored; 235 lexer tests;
195 GLR tests with two existing ignored; 48 JSON Schema tests; 22 public API,
19 public API review, two integer-divisibility and two word-totality regression
tests. These include single-worker Rayon execution and exact bounded/unbounded
certificate equality, as well as public save/load and nested compiled modules.

## Matched LLguidance diagnostic

The independently audited two-framework comparison used the same pinned
P-core/wall-clock protocol and the earlier fixed LLguidance executable. Matching
complete example outcomes and lengths yielded 2,910,543 positions in 30,571
nonempty examples; 218 examples were excluded explicitly for disagreement or
runtime errors rather than silently truncating their arrays.

Across every empirical p99-to-p100 knot and the interpolated p99 endpoint,
GLRMask was never slower. There was one exact tie at 3.2868 ms near p99.994606,
so this is not strict dominance at every recorded point. Restricting to examples
correct for both frameworks produced the same single-tie qualification.

Matched GLRMask/LLguidance figures: p50 0.0057/0.0080 ms; p99 0.1605/0.2737 ms;
p99.9 0.570346/0.886746 ms; maximum 3.5615/7.4097 ms. GLRMask remains slower
at p90/p95 and in mean schema compilation (2.738077 versus 0.715475 ms).
These observations are empirical, not latency guarantees for unseen workloads.

## Evidence and publication distinction

Windows evidence is under `C:/gdt22/artifacts/final-tail-1233/`:
`target-20260924-133133`, `full-20260924-133226`, the three streamed-mask audit
directories, and `matched-ll-0941-20260924-134540`. Raw scalar timings,
independent run maxima, manifests, driver copies and binary hashes are retained.

This 8,326-schema diagnostic must not be relabelled as the canonical full JSB
publication. The final `make final-jsb` run uses all 9,558 official schemas,
four frameworks and its documented common-success/common-present-prefix plot
population. Its raw failures and exact plot counts must be reported separately.
