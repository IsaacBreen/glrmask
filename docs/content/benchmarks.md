### JSON Schema

Time between masks. Lower is better.

{{json-plot}}

**TBM (µs)**

| Statistic | Static / FastRuntime | Dynamic / FastBuild | Balanced / O2 | llguidance |
|---|---:|---:|---:|---:|
| Mean | 3.0 | 31.5 | 12.7 | 25.9 |
| p50 | 2.6 | 9.1 | 6.9 | 10.6 |
| p90 | 5.1 | 86.4 | 30.3 | 40.7 |
| p99 | 12.2 | 239.5 | 74.8 | 289.9 |
| p99.9 | 24.9 | 754.2 | 206.1 | 1,092.6 |
| Max | 283.1 | 3,647.8 | 2,140.9 | 6,931.5 |

**Time to first mask (ms)**

| Statistic | Static / FastRuntime | Dynamic / FastBuild | Balanced / O2 | llguidance |
|---|---:|---:|---:|---:|
| Mean | 33.7 | 3.9 | 12.3 | 1.2 |
| p50 | 12.6 | 2.3 | 6.2 | 0.8 |
| p90 | 78.9 | 6.5 | 19.0 | 1.8 |
| p99 | 302.3 | 35.9 | 125.5 | 11.1 |
| p99.9 | 854.3 | 88.8 | 445.0 | 17.9 |
| Max | 1,293.4 | 274.6 | 631.1 | 231.0 |

14 October 2026 · 11,277 complete valid trajectories accepted by all four engines · 1,734,930 TBM positions per engine · 8,326 common-built cases for TTFM. One unpinned native wall-clock timing run on AWS c8i.8xlarge (Intel Xeon 6975P-C, 32 available CPUs). Vocabulary preparation is excluded.

### JavaScript

Time between masks. Lower is better.

{{javascript-plot}}

**TBM (µs)**

| Statistic | Static / FastRuntime | Dynamic / FastBuild | Balanced / O2 | llguidance |
|---|---:|---:|---:|---:|
| Mean | 16.9 | 1,392.4 | 389.0 | 1,052.2 |
| p50 | 18.2 | 145.5 | 69.9 | 856.6 |
| p90 | 30.4 | 3,626.1 | 1,088.8 | 2,521.7 |
| p99 | 47.9 | 4,091.7 | 1,509.6 | 3,097.0 |
| p99.9 | 71.1 | 4,656.9 | 2,016.8 | 3,597.1 |
| Max | 77.4 | 5,517.4 | 2,139.0 | 3,867.9 |

**Time to first mask (ms)**

| Statistic | Static / FastRuntime | Dynamic / FastBuild | Balanced / O2 | llguidance |
|---|---:|---:|---:|---:|
| Mean | 5,089.4 | 113.3 | 458.4 | 8.0 |
| p50 | 5,099.6 | 113.0 | 457.8 | 8.0 |
| p90 | 5,152.7 | 116.9 | 464.3 | 8.1 |
| p99 | 5,223.0 | 120.7 | 470.4 | 8.2 |
| p99.9 | 5,225.9 | 121.7 | 472.3 | 8.2 |
| Max | 5,226.2 | 121.8 | 472.5 | 8.2 |

14 October 2026 · 28 complete valid trajectories accepted by all four engines · 3,805 TBM positions per engine · 55 common-built cases for TTFM. One unpinned native wall-clock timing run on AWS c8i.8xlarge (Intel Xeon 6975P-C, 32 available CPUs). Vocabulary preparation is excluded.

#### Dynamic mode

{{dynamic}}

{{json-dynamic-plot}}

[Explore Dynamic JSON results](https://glrmask.com/?dataset=json&mode=dynamic#benchmarks).

There was no CPU pinning or SMT sibling reservation, and checkpoint publication overlapped the benchmark. Saved telemetry cannot isolate scheduling effects; latency stabilization is not certified. Successful builds do not establish semantic correctness. All 9,558 JSON schemas were attempted: build errors are Dynamic 232, Balanced 232, Static 233 and llguidance 1,226. Incorrect examples are 460 for each GLRMask mode and 100 for llguidance; runtime-error examples are 0 and 6 respectively. JavaScript attempts all 55 fixtures: all builds succeed, with 1 incorrect example per GLRMask mode and 4 for llguidance. Headline TBM excludes incomplete or nonaccepted valid trajectories; all outcomes remain in the raw records. This finite corpus does not establish full-language equivalence or whole-corpus dominance.

[Native AWS benchmark details](docs/benchmark-aws-2026-10-14.md). Historical [October 8 Mac JSON](docs/benchmark-cfa-2026-10-08.md) and [October 9 Mac JavaScript](docs/benchmark-javascript-2026-10-09.md) reports are retained with their original measurement protocols. These are source builds, not measurements of the published 0.2 packages.
