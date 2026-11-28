### JSON Schema

Time between masks. Lower is better.

{{json-plot}}

**Time between masks (µs)**

| Statistic | GLRMask | llguidance | XGrammar |
|---|---:|---:|---:|
| mean | 3.2 | 25.3 | 35.9 |
| p50 | 2.9 | 11.4 | 9.1 |
| p90 | 5.1 | 40.6 | 38.2 |
| p99 | 10.1 | 267.8 | 534.0 |
| p99.9 | 17.0 | 1,070.7 | 2,736.8 |
| p99.99 | 25.2 | 3,627.2 | 7,901.3 |
| max | 43.8 | 8,276.3 | 12,660.3 |

**Time to first mask (ms)**

| Statistic | GLRMask | llguidance | XGrammar |
|---|---:|---:|---:|
| mean | 37.3 | 1.3 | 84.5 |
| p50 | 16.8 | 0.9 | 44.3 |
| p90 | 75.0 | 2.1 | 203.2 |
| p99 | 306.4 | 13.3 | 834.3 |
| max | 1,350.1 | 48.6 | 4,066.2 |

### JavaScript

Time between masks. Lower is better.

{{javascript-plot}}

**Time between masks (µs)**

| Statistic | GLRMask | llguidance | XGrammar |
|---|---:|---:|---:|
| mean | 17.2 | 1,152.4 | 116,561.8 |
| p50 | 18.4 | 938.6 | 47,894.1 |
| p90 | 30.1 | 2,661.0 | 621,935.7 |
| p99 | 46.7 | 3,389.5 | 679,617.9 |
| p99.9 | 63.6 | 3,761.8 | 805,342.2 |
| p99.99 | 69.3 | 3,819.2 | 825,890.1 |
| max | 69.6 | 3,825.7 | 826,881.4 |

**Time to first mask (ms)**

| Statistic | GLRMask | llguidance | XGrammar |
|---|---:|---:|---:|
| JavaScript | 5,593.7 | 8.9 | 4,175.5 |

19 October 2026 campaign · AWS c7i.16xlarge, Llama 3.1 8B Instruct vocabulary; GLRMask 3d65c7a, llguidance 1.9.1, XGrammar 0.2.8; matched valid trajectories (JSON: 1,570,228 positions; JavaScript: 2,370); JavaScript TTFM is a single cold observation. [Measurement details](https://glrmask.com/#benchmarks).
