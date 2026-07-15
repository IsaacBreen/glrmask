# Shingleback v0.1 benchmark

This document is the publication target for the bounded Shingleback v0.1 performance comparison. The v0.1 benchmark scope is exactly the CFA harness target:

```text
make example-slow-all
```

The comparison covers these four benchmark backends:

- `llguidance`
- `glrmask`
- `glrmask-native`
- `xgrammar`

It does **not** cover the full benchmark dataset. A broader corpus run is separate follow-up work and must not be implied by the v0.1 results.

<!--
RELEASE INTEGRATION BLOCKER:
Worker 1 must replace the Worker 2 placeholders/comments in this document with the
exact contents of 33-example-slow-all-four-backend-pc-benchmark.md before public
release. Do not invent values, versions, commits, failures, or machine details.
-->

## Environment

The benchmark records the exact machine, CPU, OS/WSL environment, relevant toolchain versions, CFA commit, Shingleback/glrmask release-equivalent commit, and exact versions or commits of the other three backends.

<!-- WORKER 2: insert exact environment and version/commit block here. -->

## Methodology

The comparison uses one stable machine state and the same benchmark harness for all four backends. It preserves the harness's natural metrics and warmup/repetition behavior, records any backend failures or timeouts explicitly, and does not change timeouts or semantics merely to improve a result.

The benchmark is a **performance comparison**. Neither `llguidance` nor `xgrammar` is treated as semantic ground truth, and a raw token-mask disagreement is not by itself evidence that either backend is correct or incorrect. Correctness differences require a separate, language-level analysis.

Before final measurement, obvious heavy background activity should be avoided so that one backend is not measured under a materially different machine load from another.

<!-- WORKER 2: insert exact commands, repetition policy, timeout policy, and any contamination checks here. -->

## Results

The results table reports the central metrics naturally emitted by `make example-slow-all` and preserves timeout or failure markers rather than dropping unfavorable rows.

<!--
WORKER 2 / WORKER 1: replace this comment with the docs-ready Markdown table from
33-example-slow-all-four-backend-pc-benchmark.md. Suggested shape, to be adapted
to the metrics the harness actually emits:

| backend | version / commit | central build metric | central mask/runtime metric | status |
|---|---|---:|---:|---|
| llguidance | ... | ... | ... | ... |
| glrmask | ... | ... | ... | ... |
| glrmask-native | ... | ... | ... | ... |
| xgrammar | ... | ... | ... | ... |

Do not publish this file with invented or placeholder numbers.
-->

## Interpretation boundaries

The v0.1 results support only claims tied to the exact benchmark target, machine, backend versions, and measured metrics recorded above.

They do not establish:

- performance on the full CFA corpus;
- hardware-independent latency or throughput guarantees;
- universal superiority of one constrained-decoding backend;
- semantic equivalence between the four backends;
- correctness of one backend merely because another backend disagrees with it.

The README should quote only a small number of headline measurements that are directly supported by this report. Detailed rows, backend versions, machine information, failures/timeouts, and methodology belong here.
