# Native AWS benchmark — 14 October 2026

The completed run measured all four engines once on an AWS c8i.8xlarge, Linux x86_64, Intel Xeon6975P-C,32 available CPUs. GLRMask source5657d277854d1a3d1bf8ea760bae4d50a7b0167e and llguidancef0971424ec072d3e4d4196bcc7f31a2f60527df9 were frozen before timing. The Llama3.1 8B Instruct vocabulary has128,256 entries; tokenizer revisiona2856192dd7c25b842431f39c179a6c2c2f627d1. The source archive SHA256 is0267240e2b842ccbe4408d7144d1ccb4898b310468798488f630fed4df577ba7.

The current `make jsb` invokes the native Rust reporter. JSON headline TBM uses11,277 complete valid trajectories accepted by every engine,1,734,930 positions each. TTFM uses8,326 common-built schemas. The full9,558-schema corpus and every error are retained separately. JavaScript has55 attempted fixtures,28 common complete valid trajectories and3,805 headline TBM positions; its TTFM distribution contains55 cases. These finite validation examples do not establish whole-language equivalence.

TBM is native wall time from committing the previously allowed token through the next ready mask. TTFM is wall time from schema-dependent build start through first ready mask. Vocabulary preparation is excluded. EOS is accounted for inside native ready-mask timing; no external postfilter is used. One problem runs at a time, with60-second build and120-second runtime limits. There are no real-schema warmups; a synthetic global warmup is enabled. Percentiles are linear interpolated; means are arithmetic. Interactive tails sample the empirical inverse survival function with straight segments, without smoothing or extrapolation.

The initial A–D delivery accidentally used the legacy Python shared-prefix TBM population. It was superseded before this publication. The assets below use the exact native valid-common export and match native summary quantiles. Static TTFM maximum is1.293389335s; Static TBM maximum is283.068µs. No measurement was rerun during export recovery or plot correction.

![A: time to first mask, 8,326 common-built schemas](assets/aws-2026-10-14-a_maskbench_ttfm.webp)

![B: time between masks, complete valid common trajectories](assets/aws-2026-10-14-b_maskbench_tbm.webp)

![C: TBM tail, native valid-common population](assets/aws-2026-10-14-c_tbm_tail_smooth_plot.webp)

![D: TTFM tail, common-built schemas](assets/aws-2026-10-14-d_ttfm_tail_smooth_plot.webp)

## Timing qualification

One wall-clock pass without CPU pinning, cpuset isolation or SMT sibling reservation. Checkpoint staging, verification and publication overlapped the living benchmark child. Retained telemetry cannot separate intrinsic work, scheduling, steal-time or frequency effects. Latency isolation/stabilization is not certified.

The Static TBM hotspot recurs at the same token in the same-source October11 Linux run; this identifies a recurring position, not its scheduling or algorithmic cause.

## Practical JavaScript native figures

These use55 common-built cases for TTFM and28 complete valid accepted trajectories (3,805 positions) for TBM. Cold construction repeats one grammar; only51 fixtures are correctly evaluated by all engines. The one-pass timing qualifications above also apply.

![JS A: TTFM of repeated cold builds](assets/aws-js-2026-10-14-a_maskbench_ttfm.webp)

![JS B: native complete valid-common TBM](assets/aws-js-2026-10-14-b_maskbench_tbm.webp)

![JS C: native valid-common TBM tail](assets/aws-js-2026-10-14-c_tbm_tail_smooth_plot.webp)

![JS D: common-built TTFM tail](assets/aws-js-2026-10-14-d_ttfm_tail_smooth_plot.webp)

## Full outcomes

| Engine | JSON build OK | JSON build errors | JSON incorrect examples | JSON runtime errors | JS build OK | JS incorrect examples |
|---|---:|---:|---:|---:|---:|---:|
| Dynamic | 9,326 | 232 | 460 | 0 | 55 | 1 |
| Balanced | 9,326 | 232 | 460 | 0 | 55 | 1 |
| Static | 9,325 | 233 | 460 | 0 | 55 | 1 |
| llguidance | 8,332 | 1,226 | 100 | 6 | 55 | 4 |

All220 JavaScript build records succeed, but semantic disagreements remain. JSON totals are38,232 records with1,923 build errors; combined with JavaScript there are38,452 records. Build success and native runtime qualification are distinct from corpus correctness. Performance on the selected cohort is not a whole-corpus dominance claim.

[JSON snapshot](benchmarks/json-schema-2026-10-14.json) and [JavaScript snapshot](benchmarks/javascript-2026-10-14.json) retain exact source hashes, outcome counts, mode statistics and tail samples. Historical Mac snapshots and reports remain unchanged. Cross-machine differences are observations, not causal code regressions.
