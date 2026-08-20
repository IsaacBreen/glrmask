# Official JSONSchemaBench engineering benchmark — 20 August 2026

This is the latest corrected **engineering** benchmark currently shown in the GLRMask README. It is an official-population view of the historical CFA run, not the final native-Rust publication sweep.

## Population

All figures and statistics in this report use exactly the **9,558 schemas in JSONSchemaBench's official `data/` tree** at commit:

```text
ba103c73756198dd9b149ddc7db7867da7a077f6
```

The corresponding MaskBench payloads supply examples/tests. The historical CFA sweep originally also contained 705 MaskBench-only additions:

```text
147  Handwritten---*
413  Synthesized---*
100  JME_*
 45  MCPspec---*
```

Those 705 cases are **excluded from every number and graph shown here**. BFCL was already separate and was not part of the historical 10,263-case sweep.

The official-population view contains 9,558 problems. GLRMask built 8,826; llguidance built 8,828. Runtime measurements exist on 7,976 GLRMask problems and 7,977 llguidance problems; the paired runtime population contains 7,970 problems.

## Historical run and correction provenance

The original full sweep ran on AWS `m8azn.3xlarge` in `us-east-1c`, with 12 physical vCPUs and 48 GiB RAM on the M8azn AMD EPYC 9R05 family. Relevant settings were:

- one measured timing traversal;
- zero per-schema warmup traversals;
- one build attempt;
- Linux thread-CPU runtime timing;
- `MIMALLOC_PURGE_DELAY=-1`;
- CFA `895d8816b6edd5757e7afa398f47c2787f901312`;
- original GLRMask tree `42f4d1cc4a1f9e5672c3c01f976b1d36db3152a9`;
- llguidance 1.6.1.

A deterministic first-commit latency bug was subsequently found in GLRMask's post-deserialization path. After `Constraint::load()`, an empty wide-frontier cache could still force deferred dynamic-vocabulary materialization during the first commit. The fix is GLRMask commit `86a9b8ba06e58f4e2750e1f27894c19253f2464b` (`Avoid lazy vocab materialization for empty wide frontiers`).

To preserve the original build measurements while correcting that runtime artifact, every problem whose original pre-fix maximum GLRMask TBM was at least 25 µs was rerun on a same-family AWS `m8azn.xlarge` (4 physical vCPUs / 16 GiB, same AZ). The two targeted reruns cover 1,292 unique problems. Only GLRMask runtime mask/commit/TBM arrays were replaced. Original build/TTFM fields, llguidance data, semantic results, token identities, and all non-target runtime records were retained.

The splice was audited against the rerun artifacts and the original pre-splice backup. All original build/framework dictionaries remained unchanged; target runtime arrays matched the reruns exactly; non-target records were unchanged.

The underlying historical timing chunks remain the expanded 10,263-problem artifact. The official 9,558 statistics and plots are a verified structural filter using the exact official JSONSchemaBench ID set. No duplicate filtered timing copy is retained.

This is therefore intentionally a **composite engineering result**, not a single-host, single-revision publication run.

## Corrected TBM distribution — official 9,558 only

Microseconds. TBM is the old CFA metric: measured mask time plus commit time for one token transition.

| TBM | GLRMask | llguidance |
|---|---:|---:|
| mean | **3.448 µs** | 21.168 µs |
| p50 | **3.051 µs** | 10.451 µs |
| p90 | **4.730 µs** | 27.578 µs |
| p95 | **5.650 µs** | 43.640 µs |
| p99 | **10.310 µs** | 222.980 µs |
| p99.9 | **17.820 µs** | 787.850 µs |
| p99.99 | **23.943 µs** | 2.290 ms |
| maximum | **70.080 µs** | 14.426 ms |

Measured TBM sample counts:

```text
GLRMask:     3,066,674
llguidance:  3,068,495
```

GLRMask had five samples at or above 50 µs and **zero at or above 75 µs**. llguidance had 17,787 samples at or above 500 µs, 864 at or above 1 ms, 366 at or above 2 ms, and 17 at or above 5 ms.

The reciprocal of mean constraint-only TBM is approximately 290,062 transitions/s for GLRMask and 47,242 transitions/s for llguidance. This is not end-to-end model throughput; it is only a convenient reciprocal service-time quantity.

## TTFM / build distribution — official 9,558 only

TTFM:

| TTFM | GLRMask | llguidance |
|---|---:|---:|
| mean | 25.737 ms | **1.652 ms** |
| p50 | 10.421 ms | **1.109 ms** |
| p90 | 59.350 ms | **2.738 ms** |
| p95 | 102.590 ms | **3.909 ms** |
| p99 | 229.809 ms | **12.023 ms** |
| p99.9 | 674.319 ms | **36.839 ms** |
| maximum | 779.588 ms | **81.104 ms** |

Constraint build time itself is nearly identical to TTFM in this old CFA run:

| Build | GLRMask | llguidance |
|---|---:|---:|
| p50 | 10.417 ms | **1.062 ms** |
| p90 | 59.346 ms | **2.715 ms** |
| p99 | 229.809 ms | **11.946 ms** |
| p99.9 | 674.317 ms | **36.832 ms** |
| maximum | 779.586 ms | **81.104 ms** |

The static GLRMask design is deliberately trading substantially more up-front compilation for a much smaller runtime tail.

## Paired-problem tail incidence

Among the 7,970 paired runtime problems:

| Event observed in a problem | GLRMask | llguidance |
|---|---:|---:|
| any TBM ≥ 50 µs | 4 / 7,970 (0.0502%) | 7,967 / 7,970 (99.9624%) |
| any TBM ≥ 100 µs | 0 | 6,906 / 7,970 (86.6499%) |
| any TBM ≥ 200 µs | 0 | 1,952 / 7,970 (24.4918%) |
| any TBM ≥ 500 µs | 0 | 855 / 7,970 (10.7277%) |
| any TBM ≥ 1 ms | 0 | 142 / 7,970 (1.7817%) |

## Interpretation limits

This result supersedes the July README figures, but it is not the final publication benchmark:

1. The displayed population is correctly restricted to the 9,558 official JSONSchemaBench schemas, but the source timing chunks came from the older expanded CFA sweep.
2. GLRMask build/TTFM remains from the original 12-vCPU M8azn run; selected fixed-code runtime timing was refreshed on a 4-vCPU M8azn machine from the same CPU family.
3. The old CFA path uses a Python benchmark adapter and Linux thread-CPU timing. The final benchmark runner is native Rust and measures wall-clock `commit -> next mask ready` directly.
4. The old CFA replay tokenizer is not the canonical real Hugging Face Llama-3.1 tokenizer used by the final native runner.
5. This run used llguidance 1.6.1. The final native benchmark runner is pinned to llguidance 1.8.0 commit `dbaf504d498b6aeede06ae57adc6f7c2c4848c59`.

The final native 9,558-schema sweep should replace these engineering figures once it is intentionally run.

## Figure provenance

The README figures are copied byte-for-byte from the current official-population plots in CFA's canonical result directory:

- `docs/assets/benchmark-tbm-tail-2026-08-20.webp` ← `c_tbm_tail_smooth_plot.webp`
- `docs/assets/benchmark-tbm-2026-08-20.webp` ← `b_maskbench_tbm.webp`
- `docs/assets/benchmark-ttfm-2026-08-20.webp` ← `a_maskbench_ttfm.webp`

CFA preserves the previous expanded-population plots separately under `plots/expanded-10263/`; the current top-level plots and `plots/official-jsb-9558/` use the 9,558 official schemas only.
