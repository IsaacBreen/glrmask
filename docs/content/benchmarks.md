### JSON Schema

Time between masks (TBM) measures token commitment through the next ready mask. Lower is better.

{{json-plot}}

| Metric | GLRMask Static | llguidance 1.8.0 |
|---|---:|---:|
| TBM p50 (µs) | 2.5 | 8.7 |
| TBM p99 (µs) | 9.5 | 379.6 |
| TBM p99.9 (µs) | 22.7 | 1,360.9 |
| TBM maximum (µs) | 59.3 | 7,843.4 |
| Build p50 (ms) | 9.6 | 0.5 |
| TTFM p50 (ms) | 9.7 | 0.5 |
| TTFM p99 (ms) | 229.4 | 6.9 |

*8 October 2026 · GLRMask source `f9f5c84`, CFA `5d3feb4` · macOS 27 ARM64, 10 logical CPUs, Rust 1.95.0 · Llama 3.1 8B Instruct vocabulary. TBM: 2,925,983 aligned token positions, calling-thread CPU time, minimum of two native processes. Build/TTFM: 8,326 matched schemas for Static and 8,327 for llguidance. These conditional timings do not establish full-corpus correctness. [Benchmark details](docs/benchmark-cfa-2026-10-08.md).*

### JavaScript

{{javascript-plot}}

| Metric | GLRMask Static | llguidance 1.8.0 |
|---|---:|---:|
| TBM p50 (µs) | 18.8 | 938.0 |
| TBM p99 (µs) | 37.6 | 3,125.0 |
| TBM maximum (µs) | 67.0 | 4,217.4 |
| TTFM selected (ms) | 3,674.4 | 5.8 |

*9 October 2026 · GLRMask source `5f2609ac`, CFA `159759c4` · macOS ARM64 · 31 examples, 4,099 token positions. TBM: same-pass mask plus commit CPU time; minimum of 20 Static traversals and one llguidance traversal, after one warmup. TTFM is a selected shared grammar build plus the initial trace’s first mask, not an independent distribution. Grammar policies differ; language equivalence is not established. [Benchmark details](docs/benchmark-javascript-2026-10-09.md).*

Both datasets measure source builds, not the published 0.2 packages. The JSON plot shows exceedance probabilities down to 0.001%; JavaScript ends at its measured support (one token in 4,099). Maxima are retained in the tables.

### Dynamic mode

{{dynamic}}

Compare modes in the [interactive benchmarks](https://glrmask.com/#benchmarks).
