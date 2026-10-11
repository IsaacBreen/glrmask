### JSON Schema

Time between masks. Lower is better.

{{json-plot}}

**TBM (µs)**

| Statistic | GLRMask | llguidance |
|---|---:|---:|
| Mean | 2.9 | 27.5 |
| p50 | 2.5 | 8.7 |
| p99 | 9.5 | 379.6 |
| p99.9 | 22.7 | 1,360.9 |
| Max | 59.3 | 7,843.4 |

**Time to first mask (ms)**

| Statistic | GLRMask | llguidance |
|---|---:|---:|
| Mean | 27.4 | 0.8 |
| p50 | 9.7 | 0.5 |
| p99 | 229.4 | 6.9 |
| Max | 1,052.9 | 136.6 |

8 October 2026 · 2,925,983 aligned token positions · macOS 27 ARM64, 10 logical CPUs. [Benchmark details](docs/benchmark-cfa-2026-10-08.md).

#### Dynamic mode

Care about compile times? Try dynamic mode.

{{dynamic}}

{{json-dynamic-plot}}

**Time to first mask (ms)**

| Stat | GLRMask | GLRMask<br>dynamic | llguidance |
|---|---:|---:|---:|
| Mean | 27.4 | 2.5 | 0.8 |
| p50 | 9.7 | 1.4 | 0.5 |
| p99 | 229.4 | 21.9 | 6.9 |
| Max | 1,052.9 | 132.0 | 136.6 |

[Explore Dynamic JSON results](https://glrmask.com/?dataset=json&mode=dynamic#benchmarks).

### JavaScript

Time between masks. Lower is better.

{{javascript-plot}}

**TBM (µs)**

| Statistic | GLRMask | llguidance |
|---|---:|---:|
| Mean | 17.0 | 1,101.9 |
| p50 | 18.8 | 938.0 |
| p99 | 37.6 | 3,125.0 |
| Max | 67.0 | 4,217.4 |

**Time to first mask (ms)**

| Statistic | GLRMask | llguidance |
|---|---:|---:|
| Mean | 3,674.4 | 5.7 |
| Selected | 3,674.4 | 5.8 |
| Max | 3,674.4 | 6.0 |

9 October 2026 · 31 examples, 4,099 token positions · macOS ARM64. [Benchmark details](docs/benchmark-javascript-2026-10-09.md).

Both datasets measure source builds, not the published 0.2 packages.
