# CFA full-corpus benchmark — 16 July 2026

This report covers the completed **10,263-problem primary CFA corpus** run from 16 July 2026. It is separate from the bounded 195-problem [`example-slow-all` v0.1 benchmark](benchmark-0.1.md), and it does not replace that release report.

The main result is that GLRMask native had a low and tightly bounded runtime tail over its valid measured population: **3.497 µs median TBM, 10.521 µs p99, 15.694 µs p99.9, and 49.539 µs maximum**, with zero measured TBM samples above 1 ms. This came with materially higher ahead-of-time build cost than LLGuidance native.

This was a resumed engineering run over three `glrmask-main` revisions, not an immutable-tag benchmark. Results should be interpreted with the exact cohort and revision denominators below.

## Scope and coverage

The raw completed chunks contain exactly **10,263 unique primary problems**. The coverage-aware aggregate found no duplicate, missing, or extra problem IDs relative to `problems.txt`. The optional phase was not run.

| Coverage cohort | Problems | Frameworks |
|---|---:|---|
| Three-framework cohort | 4,425 | GLRMask native, GLRMask dynamic, LLGuidance native |
| Two-framework cohort | 5,838 | GLRMask native, LLGuidance native |
| Total primary corpus | 10,263 | Coverage varies as shown above |

The original uniform-framework merge failed because the framework list changes after problem 4,425. The underlying chunks remained valid. They were aggregated with explicit per-problem coverage instead of filling the absent dynamic results.

The preserved transition evidence is explicit. `fixed100-0035` started with all three frameworks; on its first problem, GLRMask dynamic reported roughly 17–18 ms TBM on two retained examples while native remained around 10–11 µs. The process then received `SIGTERM`, wrote a one-problem partial payload, and that payload was moved to `primary/quarantine-switch-no-dynamic-20260716-165550/`. The authoritative run resumed from problem 4,425 as `nodyn100-*` without dynamic. The quarantined partial result is preserved and listed in the coverage manifest, but is not counted as an additional corpus record.

### Source revisions

The runner pulled and rebuilt `glrmask-main` during controlled resumptions. The coverage manifest maps every authoritative problem to its measured revision:

| GLRMask revision | Problems | Notes |
|---|---:|---|
| `8735fed6c8d30ce2b957d0d97c3c5088b632d311` | 725 | `Forbid lazy lexer cache initialization in workers` |
| `158b394448480203e301843e5802948861e7391c` | 3,700 | `Revert "Restore recursive additionalProperties anyOf lowering"` |
| `f2e6a03b865e7032179019399d0535e4c441a0f2` | 5,838 | `Fail closed on invalid analysis-state coordinates`; two-framework cohort |

The package installed into the benchmark environment reported version `0.1.0`, but it was rebuilt from these local `glrmask-main` revisions. It was **not** the exact public `v0.1.0` tag. The run logs did not record a CFA Git commit, so this report does not invent one.

## Methodology

The benchmark ran on the MSI Windows PC under Ubuntu 24.04 / WSL2, using Python 3.12.13 and the cached Llama 3 vocabulary. The benchmark setup allocated 10 GB of WSL memory and 8 GB of swap on a 13th-generation Intel Core i7-13620H host.

For each configured framework/problem pair:

- one build was attempted;
- the hard build deadline was 60 seconds;
- at most three examples were retained per problem;
- GLRMask native and LLGuidance native used 20 configured timing traversals;
- GLRMask dynamic used 2 configured timing traversals;
- very fast examples could be repeated further to satisfy the harness minimum measurement duration;
- each recorded token timing is the elementwise minimum across the traversals;
- timing values after the first effective-death token are null and excluded.

TBM is the measured mask-generation time plus commit time for one token position. TTFM here follows the CFA report definition: successful build time plus the median first-mask time across the retained examples. TTFM is reported only when a successful build has at least one finite first-mask sample; successful builds with no retained measurable example remain in the build-time counts but are excluded from TTFM.

The aggregate covers **21,018 examples** and **3,168,113 token positions before timing truncation**. Of those examples, 10,131 were labelled expected-valid and 10,887 expected-invalid. Effective death occurred in 11,098 examples: 10,836 expected-invalid examples and 262 expected-valid examples. The latter is a material corpus/harness outcome and is not hidden.

## Build coverage

Build failures and unsupported schemas remain in the outcome counts; they are not removed from the corpus denominator.

| Framework | Configured problems | Successful builds | Success rate | Failed builds | Timeouts |
|---|---:|---:|---:|---:|---:|
| GLRMask native | 10,263 | 8,966 | 87.36% | 1,297 | 0 |
| GLRMask dynamic | 4,425 | 4,083 | 92.27% | 342 | 0 |
| LLGuidance native | 10,263 | 8,967 | 87.37% | 1,296 | 0 |

For the full configured native-versus-LLGuidance cohort, the build outcome matrix was:

| Outcome | Problems |
|---|---:|
| Both built successfully | 8,956 |
| GLRMask native only | 10 |
| LLGuidance native only | 11 |
| Both failed | 1,286 |

The 1,286 joint failures are concentrated in unsupported comparison-boundary schemas, not timeouts. The largest category is JSON Schema `not`: the four late chunks `nodyn100-0054` through `nodyn100-0057` contain 400 such problems, and both adapters reject all 400. Other common boundaries include `dependencies`, `uniqueItems`, unsupported `format` values, conditional keywords, `contains`, and `propertyNames`. GLRMask's CFA comparison adapter deliberately preflights against LLGuidance's supported-schema boundary unless explicitly configured otherwise.

## Build time and TTFM

Times are milliseconds over successful builds for each framework's actual coverage.

### Build time

| Framework | Problems | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLRMask native | 8,966 | 50.963 | 151.503 | 257.564 | 565.006 | 2,217.617 | 6,440.287 |
| GLRMask dynamic | 4,083 | 4.550 | 12.937 | 17.381 | 44.531 | 249.956 | 549.673 |
| LLGuidance native | 8,967 | 0.905 | 2.510 | 3.952 | 11.810 | 42.986 | 239.964 |

### TTFM

| Framework | Problems | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLRMask native | 8,112 | 49.976 | 126.751 | 233.860 | 449.047 | 1,748.917 | 6,440.288 |
| GLRMask dynamic | 3,577 | 4.405 | 11.162 | 14.978 | 31.690 | 230.266 | 549.949 |
| LLGuidance native | 8,109 | 0.923 | 2.093 | 3.129 | 9.885 | 14.883 | 49.805 |

LLGuidance native built faster on 8,955 of the 8,956 joint-success problems using a 1% tie tolerance. The median per-problem LLGuidance/GLRMask build-time ratio was 0.019; equivalently, GLRMask native paid roughly a 52× larger paired median build cost. This is the expected ahead-of-time/runtime tradeoff, not a runtime result.

GLRMask dynamic reduced its median build time to 4.550 ms over its 4,083 successful builds, but its runtime distribution was much slower and substantially less stable.

## Runtime latency

### TBM by framework

Microseconds. These are each framework's valid measured token populations, so the dynamic denominator is smaller.

| Framework | Token samples | p50 | p90 | p95 | p99 | p99.9 | max | >1 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLRMask native | 2,116,613 | 3.497 | 6.533 | 7.778 | 10.521 | 15.694 | 49.539 | 0 |
| GLRMask dynamic | 1,170,139 | 59.216 | 8,166.025 | 15,925.355 | 23,121.957 | 55,642.735 | 323,608.543 | 219,251 |
| LLGuidance native | 2,118,045 | 13.709 | 41.918 | 58.758 | 249.276 | 956.856 | 8,050.274 | 1,560 |

### Mask generation

| Framework | Token samples | p50 | p90 | p95 | p99 | p99.9 | max | >1 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLRMask native | 2,127,718 | 1.440 | 2.954 | 3.629 | 5.172 | 7.675 | 28.565 | 0 |
| GLRMask dynamic | 1,175,725 | 53.549 | 8,150.759 | 15,903.389 | 23,097.303 | 55,387.264 | 323,576.983 | 219,233 |
| LLGuidance native | 2,129,162 | 12.197 | 35.854 | 48.600 | 246.788 | 950.309 | 8,041.301 | 1,478 |

### Commit

| Framework | Token samples | p50 | p90 | p95 | p99 | p99.9 | max | >1 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLRMask native | 2,116,613 | 1.660 | 3.908 | 4.846 | 7.296 | 10.906 | 34.410 | 0 |
| GLRMask dynamic | 1,170,139 | 4.927 | 12.237 | 18.300 | 39.903 | 68.620 | 140.166 | 0 |
| LLGuidance native | 2,118,045 | 1.407 | 3.297 | 4.958 | 22.590 | 98.632 | 310.312 | 0 |

GLRMask native's measured tail remained below 50 µs. LLGuidance native had 1,560 TBM samples above 1 ms and a maximum of 8,050.274 µs. GLRMask dynamic had 219,251 TBM samples above 1 ms, including 77,797 above 10 ms, with a maximum of 323,608.543 µs.

## Pairwise runtime comparisons

### GLRMask native vs LLGuidance native

The paired runtime population contains **2,111,184 shared token positions** from **8,103 problems** with at least one finite shared TBM value. The separate discrepancy analysis below has 8,116 mask-evaluable problems.

| Framework | Paired TBM samples | p50 µs | p90 µs | p95 µs | p99 µs | p99.9 µs | max µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLRMask native | 2,111,184 | 3.498 | 6.536 | 7.781 | 10.522 | 15.692 | 49.539 |
| LLGuidance native | 2,111,184 | 13.718 | 41.964 | 58.823 | 250.070 | 957.235 | 8,050.274 |

At shared token positions and a 1% tie tolerance, GLRMask native was faster at 2,096,608 positions (99.31%), LLGuidance native was faster at 13,067 (0.62%), and 1,509 tied. The median per-position LLGuidance/GLRMask TBM ratio was **3.870×**; the ratio reached 54.191× at p99.

### GLRMask native vs GLRMask dynamic

Dynamic coverage exists only in the first 4,425 configured problems. The paired runtime population contains **1,170,139 shared token positions** from **3,576 problems** with at least one finite shared TBM value. The separate discrepancy analysis below has 3,579 mask-evaluable problems.

| Framework | Paired TBM samples | p50 µs | p90 µs | p95 µs | p99 µs | p99.9 µs | max µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLRMask native | 1,170,139 | 3.403 | 6.201 | 7.334 | 10.145 | 15.694 | 49.539 |
| GLRMask dynamic | 1,170,139 | 59.216 | 8,166.025 | 15,925.355 | 23,121.957 | 55,642.735 | 323,608.543 |

GLRMask native was faster at 1,152,805 shared positions (98.52%). The median per-position dynamic/native TBM ratio was 20.340×. Dynamic mode's faster build therefore came with a large runtime and tail penalty in this corpus.

For dynamic versus LLGuidance, the paired population contains 1,167,293 token positions from 3,575 problems with at least one finite shared TBM value. The median paired TBM values were 59.097 µs for dynamic and 13.516 µs for LLGuidance. The discrepancy analysis remains based on 3,579 mask-evaluable problems.

## Mask discrepancies

A raw token-mask discrepancy is not a correctness verdict. Systems may expose different token-admissibility policies while still accepting overlapping languages. These counts identify where masks differed and where targeted semantic investigation is warranted.

The conservative result artifacts retain all-framework union/intersection disagreement counts, but omit per-framework disputed-token vote details. They also retain a dedicated exact problem-level native-versus-dynamic flag. Consequently:

- **GLRMask native vs dynamic:** exact problem-level count of 36 discrepancies among 3,579 evaluable problems (1.01%).
- **GLRMask native vs LLGuidance, two-framework cohort:** exact count of 3,884 among 4,537 evaluable problems (85.61%), spanning 198,442 token steps and 35,566,059 disputed-token events.
- **GLRMask native vs LLGuidance, full run:** bounded rather than falsely exact: 6,894 to 6,930 discrepant problems among 8,116 evaluable problems, or 84.94% to 85.39%. The 36-problem width is exactly the three-framework cases where native and dynamic differed and conservative artifacts cannot uniquely attribute the LLGuidance pair.
- **Dynamic vs LLGuidance, three-framework cohort:** bounded at 3,010 to 3,046 among 3,579 evaluable problems.

The high native-versus-LLGuidance disagreement rate is a major semantic finding, but not evidence that either backend is wrong. The retained conservative artifacts are insufficient for exact full-run pair attribution or adjudication.

## Tail and state-dependent observations

- GLRMask native's worst measured TBM was 49.539 µs; no native TBM sample exceeded 1 ms.
- LLGuidance native's worst measured TBM was 8,050.274 µs, on `Github_hard---o62060`.
- GLRMask dynamic's worst cluster was on `Github_medium---o28130`, reaching 323,608.543 µs. It used only two configured timing traversals, and many token positions on that example remained in the hundreds of milliseconds.
- GLRMask native's maximum successful build was 6,440.287 ms; LLGuidance native's was 239.964 ms; dynamic's was 549.673 ms.
- The four 100% no-build chunks near the end are a contiguous `not`-keyword block, not an infrastructure failure.

## Plot audit

The eight pre-existing plots were generated in CFA's global chunk mode over all 130 chunks. Their labels do not expose the framework-coverage change or dynamic denominator. They were preserved unchanged but **quarantined from publication**. This report uses coverage-labelled tables generated from the validated aggregate. No replacement plot was necessary.

## Reproduction and artifacts

The raw run remains unchanged. On the machine holding the run directory, regenerate the complete aggregate with:

```bash
cd constraint-framework-analysis
python -m scripts.aggregate_full_corpus_run 'results/run 16-jul-26'
```

Derived artifacts are under `results/run 16-jul-26/derived/`:

- `aggregate-problems.jsonl.zst`: one compact machine-readable record per problem;
- `aggregate-summary.json`: all reported distributions, coverage, failures, outliers, and discrepancy bounds;
- `coverage-manifest.json`: authoritative problem-to-chunk, cohort, and revision mapping;
- `input-manifest.json` and `input-chunks.sha256`: hashes for 130 summary chunks and 69 raw timing sidecars;
- `derived-manifest.sha256`: hashes for every generated artifact other than the manifest itself;
- `plot-audit.json` and `plot-audit.md`;
- `aggregate_full_corpus_run.py`: exact regeneration script.

Independent validation passed for all 199 raw input hashes, all generated artifact hashes, all JSON files, the Zstandard JSONL stream, and exactly 10,263 unique aggregate records.

## Interpretation limits

This run supports claims only for its recorded machine, corpus, framework coverage, timing protocol, and three measured GLRMask revisions. In particular:

- it is not the bounded release-tag v0.1 benchmark;
- it is not a single-revision benchmark;
- dynamic results cover 4,425 configured problems, not 10,263;
- runtime percentiles exclude null post-death positions;
- 262 expected-valid examples reached effective death;
- unsupported schemas and failed builds remain in coverage counts but not successful-build timing distributions;
- discrepancies are not correctness judgements;
- the optional corpus phase was not run;
- the run artifacts did not record the CFA Git commit.
