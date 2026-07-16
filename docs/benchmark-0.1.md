# GLRMask v0.1 benchmark

GLRMask v0.1 uses a bounded, deliberately difficult performance comparison based on the CFA harness target:

```text
make example-slow-all
```

The target expands to 195 selected slow or problematic schemas and compares four backends:

- `llguidance`
- `glrmask`
- `glrmask-native`
- `xgrammar`

This is **not** the full benchmark dataset. A broader corpus run is separate follow-up work. The later [10,263-problem CFA full-corpus report](benchmark-full-corpus-2026-07-16.md) is an engineering run with different framework coverage and multiple GLRMask revisions; it does not replace this bounded release benchmark.

## Benchmark status

The final four-backend comparison is being completed separately from the v0.1.0 package release. Final aggregate performance results are therefore not claimed in the v0.1.0 release documentation. They will be added here in a documentation-only follow-up after the completed result payload has been parsed and checked.

Backend failures and hard timeouts will be reported rather than discarded. No result will be reconstructed from partial console output or earlier aborted runs.

## Environment

The final run uses this machine and software environment:

```text
Host: MSI Windows PC
CPU: 13th Gen Intel(R) Core(TM) i7-13620H
Physical cores: 10
Logical processors: 16
Host RAM: 15.7 GiB
WSL: Ubuntu 24.04 under WSL2
Kernel: 6.18.33.2-microsoft-standard-WSL2
WSL memory: 10 GB
WSL swap: 8 GB
Python: 3.12.13
```

Backend versions and release provenance:

| Backend | Version / provenance |
|---|---|
| `llguidance` | `1.7.6` |
| `glrmask` | `0.1.0`, frozen release RC `a97e9bc18756590c74d01e1e06cca04176f71d52` |
| `glrmask-native` | same `glrmask 0.1.0` release code, native adapter path |
| `xgrammar` | `0.2.3` |

The CFA benchmark snapshot is based on commit `d753fb7403e63106ddecb22d7829b2cf669307fd`. The benchmark copy contains harness and compatibility changes needed to expose exactly the four requested adapters, use the local tokenizer cache and corpus, preserve compatibility with the frozen, then-branded Shingleback RC, and enforce a true parent-process hard timeout for native xgrammar compilation. Those changes do not alter GLRMask release code.

## Methodology

The effective final command is:

```bash
make example-slow-all \
  PYTHON=/home/isaac/release-bench/venv/bin/python \
  FRAMEWORKS='llguidance glrmask glrmask_native xgrammar' \
  BUILD_RUNS=3 \
  BUILD_TIMEOUT=60 \
  TIMING_RUNS='glrmask_native:50,default:100,llguidance:1' \
  OUTPUT=/home/isaac/release-bench/results/example-slow-all-four-backend.json.zst \
  V=1
```

Key benchmark semantics are:

```text
195 selected slow/problematic schemas
--max-examples-per-problem -1
--discrepancy-sample-budget 500
--max-dispute-scan 0
--on-build-error continue
--build-timeout-seconds 60
--build-runs 3
--timing-runs glrmask_native:50,default:100,llguidance:1
```

The comparison is a **performance comparison**. Neither `llguidance` nor `xgrammar` is treated as semantic ground truth, and a raw token-mask disagreement is not by itself evidence that either backend is correct or incorrect. Correctness differences require separate language-level analysis.

A pre-measurement five-second machine-load sample observed low background activity, with approximately 3.9% average total CPU utilization. The benchmark runs on Linux-native WSL paths rather than `/mnt/c` to avoid mounted-Windows-filesystem overhead.

## Results

Final aggregate results are not included in the v0.1.0 package release. The completed result payload will be parsed to report, for each backend, build successes, build failures, hard 60-second timeouts, central build-time metrics, runtime or token-mask metrics, tail metrics where clearly defined, and the exact number of contributing problems, examples, and tokens.

## Interpretation boundaries

Any later v0.1 benchmark results support only claims tied to the exact benchmark target, machine, backend versions, and measured metrics recorded here.

They do not establish:

- performance on the full CFA corpus;
- hardware-independent latency or throughput guarantees;
- universal superiority of one constrained-decoding backend;
- semantic equivalence between the four backends;
- correctness of one backend merely because another backend disagrees with it.
