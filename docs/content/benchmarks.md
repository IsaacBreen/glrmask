### JSON Schema

Time between masks (TBM) measures token commitment through the next ready mask. Lower is better.

{{json-plot}}

**TBM (µs)**

| Percentile | GLRMask Static | llguidance 1.8.0 |
|---|---:|---:|
| p50 | 2.5 | 8.7 |
| p99 | 9.5 | 379.6 |
| p99.9 | 22.7 | 1,360.9 |
| Max | 59.3 | 7,843.4 |

**Build / time to first mask (TTFM), ms**

| Metric | GLRMask Static | llguidance 1.8.0 |
|---|---:|---:|
| Build&nbsp;p50 | 9.6 | 0.5 |
| TTFM&nbsp;p50 | 9.7 | 0.5 |
| TTFM&nbsp;p99 | 229.4 | 6.9 |

8 October 2026 · 2,925,983 aligned token positions · macOS 27 ARM64, 10 logical CPUs. [Benchmark details](docs/benchmark-cfa-2026-10-08.md).

### JavaScript

{{javascript-plot}}

**TBM (µs)**

| Percentile | GLRMask Static | llguidance 1.8.0 |
|---|---:|---:|
| p50 | 18.8 | 938.0 |
| p99 | 37.6 | 3,125.0 |
| Max | 67.0 | 4,217.4 |

**Time to first mask (TTFM), ms**

| Measurement | GLRMask Static | llguidance 1.8.0 |
|---|---:|---:|
| Selected | 3,674.4 | 5.8 |

9 October 2026 · 31 examples, 4,099 token positions · macOS ARM64. [Benchmark details](docs/benchmark-javascript-2026-10-09.md).

Both datasets measure source builds, not the published 0.2 packages.

### Dynamic mode

{{dynamic}}

Compare modes in the [interactive benchmarks](https://glrmask.com/#benchmarks).
