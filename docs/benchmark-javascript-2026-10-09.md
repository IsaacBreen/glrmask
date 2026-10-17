# JavaScript benchmark: 9 October 2026

This is a separate source benchmark from the JSON Schema dataset. It does not
measure the published GLRMask 0.2 packages or establish language equivalence.

- GLRMask source: `5f2609ac562cb637ea9976e31e9950c47a2d306a`.
- CFA source: `159759c41891a7f7feba7972c8adee686a0ec125`.
- Comparator: official llguidance 1.8.0 macOS ARM64 wheel.
- Host: macOS 27 ARM64; Python 3.12; 128,256-coordinate vocabulary.
- Measurement interval: 9 October 2026, 05:25:09.565598–05:26:19.560011 UTC.
- Fixture: 31 JavaScript examples, 4,099 target token positions. No EOS append.

## Timing

TBM is same-pass mask plus commit at the native adapter boundary, using Python
`thread_time_ns`. It includes the native/FFI calls and commit adapter checks,
but excludes packed-array reset, Boolean conversion, and full-vocabulary oracle
comparisons. Each framework has one excluded warmup traversal. Static has 20
measured traversals; Dynamic and llguidance have one. The elementwise minimum
of each token's same-pass sum is retained. These are unequal repetition counts,
not a common cold-run distribution.

Build is grammar compiler wall time with vocabulary preparation outside the
per-grammar timer. It excludes worker startup/IPC, parent load/start,
serialization, tokenizer loading, and shared vocabulary preparation. Selected
build minima are 3,674.368791 ms (Static), 91.345875 ms (Dynamic), and 3.786584 ms
(llguidance), from 1, 3, and 3 measured builds respectively. A cumulative build
budget limited Static to one build.

The published first-mask values use those selected build minima plus the
initial trace's first-mask CPU time: 3,674.372791, 94.165750, and 5.789959 ms.
The 31 derived per-example first-mask values share the same grammar build;
they are not 31 independent compilation observations.

The README's Selected row is the initial example (index 0). Its Mean and Max
rows use all 31 derived per-example TTFM values with equal weight. The maxima
are 3,674.373791 ms (Static) and 5.973167 ms (llguidance); no separate component
maxima are summed. TBM means give each of the 4,099 retained token positions
equal weight, rather than averaging per-example means. All means are arithmetic
averages of the same samples used for the corresponding reported percentiles
and maxima.

Table quantiles use NumPy-compatible linear interpolation. Tail samples use
the inverse empirical distribution at the retained 2,000 probabilities per
mode. The figure joins every retained measured tail sample with straight
segments in logarithmic coordinates, without numerical smoothing, anchor
selection or extrapolation. Raw measurements, tables and stored tail samples
are unchanged. The rarest supported probability is one token in 4,099,
approximately 0.0244%.

## Outcomes and limits

All 31 examples completed with zero build, mask, commit, or target-token
failures. Static and Dynamic masks agree on the fixture. The canonical
cross-framework grammar policies differ: 3,311 replay steps had cross-framework
mask differences, comprising 46,729,160 disputed token events. No completed
program oracle or full-language equivalence qualification is claimed.

## Public aggregate and evidence identity

The [sanitized snapshot](benchmarks/javascript-2026-10-09.json) contains tables,
tail samples, source revisions, timing scope, outcomes and evidence digests.
Raw traces remain outside the public repository. Independent verification
reconstructed all token summaries and all plotted samples from retained arrays,
checked recorded native extension identities, and checked same-pass minima.

| Retained result file | SHA-256 |
|---|---|
| website-data.json | `2e3f19ad908dc2381e940734eca093e5e690235ff0e00ff57bbe777ff9a0c2d0` |
| per-token-distributions.json | `d3654adb3593e1ded586a65c9b1f593e72b7764570945c41f0effc656d87199e` |
| provenance.json | `ce539102b7501c8ef6b1843c2bdd52a6cd614306547f7506a396bfef667bbdbf` |
