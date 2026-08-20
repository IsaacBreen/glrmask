# CFA full-corpus engineering benchmark — 20 August 2026

This is the latest corrected **engineering** benchmark currently shown in the GLRMask README. It is not the final publication benchmark. The final native-Rust benchmark runner is being prepared separately and will use only the 9,558 official JSONSchemaBench schemas, a frozen real Llama-3.1 token stream, and current native framework versions.

## Scope

The run contains **10,263 MaskBench cases** from JSONSchemaBench commit `ba103c73756198dd9b149ddc7db7867da7a077f6`:

| Population | Cases |
|---|---:|
| Official JSONSchemaBench `data/` schemas | 9,558 |
| `Handwritten---*` additions | 147 |
| `Synthesized---*` additions | 413 |
| `JME_*` additions | 100 |
| `MCPspec---*` additions | 45 |
| **Total in this engineering run** | **10,263** |

The separate 1,043 BFCL MaskBench cases were not part of this run.

GLRMask built 8,966/10,263 cases; llguidance built 8,968/10,263; both built on 8,956 cases.

## Timing protocol and machine

The original full sweep ran on AWS `m8azn.3xlarge` in `us-east-1c`, with 12 physical vCPUs and 48 GiB RAM on the M8azn AMD EPYC 9R05 family. The relevant settings were:

- one measured timing traversal;
- zero per-schema warmup traversals;
- one build attempt;
- Linux thread-CPU runtime timing;
- `MIMALLOC_PURGE_DELAY=-1`;
- CFA `895d8816b6edd5757e7afa398f47c2787f901312`;
- original GLRMask tree `42f4d1cc4a1f9e5672c3c01f976b1d36db3152a9`;
- llguidance 1.6.1.

A deterministic first-commit latency bug was subsequently found in GLRMask's post-deserialization path. After `Constraint::load()`, an empty wide-frontier cache could still force deferred dynamic-vocabulary materialization during the first commit. The fix is GLRMask commit `86a9b8ba06e58f4e2750e1f27894c19253f2464b` (`Avoid lazy vocab materialization for empty wide frontiers`).

To preserve the expensive original build measurements while correcting that runtime artifact, every problem whose original pre-fix maximum GLRMask TBM was at least 25 µs was rerun on a same-family AWS `m8azn.xlarge` (4 physical vCPUs / 16 GiB, same AZ) using the original benchmark/corpus plus that fix. The two targeted reruns cover **1,292 unique problems**. Only GLRMask runtime mask/commit/TBM arrays were replaced. Original build/TTFM fields, llguidance data, semantic results, token identities, and all non-target runtime records were retained.

The resulting directory was audited against both the rerun artifacts and the original pre-splice backup. All 10,263 original build/framework dictionaries remained unchanged; target runtime arrays matched the reruns exactly; non-target records were unchanged.

This means the artifact is intentionally **hybrid**. Do not describe it as a single-host, single-revision publication run.

## Corrected runtime distribution

TBM is CFA's measured mask time plus commit time for a token transition. Microseconds:

| TBM | GLRMask | llguidance |
|---|---:|---:|
| p50 | **3.060 µs** | 10.458 µs |
| p90 | **4.740 µs** | 27.600 µs |
| p95 | **5.651 µs** | 43.740 µs |
| p99 | **10.360 µs** | 222.900 µs |
| p99.9 | **17.841 µs** | 787.640 µs |
| p99.99 | **24.001 µs** | 2,290.526 µs |
| maximum | **70.080 µs** | 14,426.103 µs |

Measured TBM sample counts were 3,074,417 for GLRMask and 3,076,198 for llguidance. GLRMask had five samples at or above 50 µs and **zero at or above 75 µs** in the corrected artifact. llguidance had 865 samples at or above 1 ms.

The GLRMask maximum was 70.080 µs on `Github_hard---o1184`, example 0, step 51 (5.030 µs mask + 65.050 µs commit).

## Build distribution

Milliseconds. These are the untouched original `m8azn.3xlarge` build measurements:

| Build | GLRMask | llguidance |
|---|---:|---:|
| p50 | 10.373 ms | **1.060 ms** |
| p90 | 58.274 ms | **2.682 ms** |
| p95 | 102.067 ms | **3.839 ms** |
| p99 | 228.690 ms | **11.945 ms** |
| p99.9 | 668.449 ms | **36.660 ms** |
| maximum | 779.586 ms | **81.104 ms** |

The static GLRMask design is deliberately trading substantially more up-front compilation for a much smaller runtime tail.

## Interpretation

This result is useful for engineering comparison and for showing the corrected runtime shape, but it has several limits:

1. The corpus is the expanded 10,263-case old CFA selection, not only the official 9,558 JSONSchemaBench schemas.
2. The GLRMask runtime tail is a targeted same-family refresh while build/TTFM remains the original 12-core run.
3. The old CFA path uses a Python benchmark adapter and Linux thread-CPU timing. The final benchmark runner is native Rust and uses wall-clock `commit -> next mask ready` timing.
4. The old CFA replay tokenizer is not the canonical real Hugging Face Llama-3.1 tokenizer used by the final native runner.
5. This run used llguidance 1.6.1; the final native benchmark runner is pinned to llguidance 1.8.0 commit `dbaf504d498b6aeede06ae57adc6f7c2c4848c59`.

Accordingly, these figures supersede the July 16 README figures, but they should themselves be replaced by the final 9,558-schema native run once that run is intentionally performed.

## Figures

The README figures are copied byte-for-byte from the regenerated canonical result directory after the final splice audit:

- `docs/assets/benchmark-tbm-tail-2026-08-20.webp` ← `c_tbm_tail_smooth_plot.webp`
- `docs/assets/benchmark-tbm-2026-08-20.webp` ← `b_maskbench_tbm.webp`
- `docs/assets/benchmark-ttfm-2026-08-20.webp` ← `a_maskbench_ttfm.webp`

The eight CFA plots plus `slow_step_reports.json.zst` and `readable_report.txt` were regenerated after the final runtime splice, so the obsolete 9–14 ms first-commit cliff and intermediate 245 µs tail are not present in these figures.
