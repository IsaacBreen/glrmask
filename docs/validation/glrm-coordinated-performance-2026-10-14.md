# Coordinated redesign: matched release-profile measurements

Native measurements taken on 14 October 2026 compare the pre-redesign commit
`54e1bb741d7b51e8fd8da7e4d0016a23f49eb878` with the coordinated GLRM/Lark implementation.
The same public-API driver, vocabulary and token traces were used for both binaries.
All **15 case/optimization combinations produced identical masks** on the checked traces,
including separately linked, runtime-loaded and prepared-loaded constraints.

## Method

Two isolated batches used opposite alternating order, six process trials per version. Each case ran 2,000 traces per trial, yielding **3,456,000 timed mask computations**, the same number of commits, and **360,000 timed traces** in total.
The machine was an Apple M1 Pro (10 cores), macOS 27, native Rust 1.95.0.
Both consumers used Cargo release defaults and two Rayon workers. All 90 registry
dependency versions/checksums matched each other and the original lockfile.
No diagnostic runtime environment overrides were used. The machine execution gate
excluded other participating heavy jobs during timing. Swap allocation was recorded
and unchanged during the first batch; this is not a claim of a dedicated laboratory machine.

The 8,192-entry synthetic vocabulary contains single bytes, shared corpus substrings
and ASCII identifiers. It exercises multi-byte model tokens and component-boundary crossings.
The cases are JSON Schema, recursive Lark arithmetic, a GLRM list, an independently
scoped child grammar and a compact exact-token set. FastBuild, Balanced and
FastRuntime are reported separately.

Vocabulary allocation, source creation/binding, compilation, preparation, linking,
save/load, state creation, masks and commits have separate timers. Runtime masks
reuse their output buffer and have an explicit warm-up. Preparation is not hidden
inside the reported link time. Warm compile/link measurements share vocabulary caches.

## Results

Values below are **baseline → candidate**. Each cell is the median of six process
statistics, not a combined percentile over dissimilar grammars.

| Case | Mode | Mask p50 (µs) | Mask p99 (µs) | Warm compile (ms) | Warm link (ms) |
|---|---|---:|---:|---:|---:|
| json_schema | FastBuild | 0.459 → 0.459 | 0.875 → 0.875 | 0.402 → 0.362 | 0.022 → 0.022 |
| lark_arithmetic | FastBuild | 0.583 → 0.583 | 1.020 → 0.959 | 0.412 → 0.539 | 0.032 → 0.033 |
| glrm_list | FastBuild | 0.209 → 0.209 | 0.938 → 0.938 | 0.127 → 0.127 | 0.008 → 0.008 |
| scoped_child | FastBuild | 0.541 → 0.542 | 0.917 → 0.916 | 0.417 → 0.461 | 0.052 → 0.052 |
| exact_token_set | FastBuild | 0.666 → 0.667 | 0.917 → 0.917 | 0.057 → 0.072 | 0.004 → 0.005 |
| json_schema | Balanced | 0.459 → 0.479 | 0.875 → 0.875 | 1.613 → 1.657 | 0.018 → 0.018 |
| lark_arithmetic | Balanced | 0.542 → 0.542 | 0.917 → 0.938 | 0.930 → 1.050 | 0.032 → 0.032 |
| glrm_list | Balanced | 0.209 → 0.209 | 0.917 → 0.917 | 0.668 → 0.741 | 0.008 → 0.009 |
| scoped_child | Balanced | 31.396 → 31.166 | 61.250 → 61.376 | 2.533 → 2.689 | 0.049 → 0.055 |
| exact_token_set | Balanced | 1.104 → 1.083 | 1.917 → 1.770 | 0.310 → 0.393 | 0.004 → 0.004 |
| json_schema | FastRuntime | 0.583 → 0.583 | 1.605 → 1.583 | 4.658 → 4.721 | 0.018 → 0.018 |
| lark_arithmetic | FastRuntime | 1.167 → 1.125 | 2.021 → 2.000 | 4.276 → 4.123 | 0.031 → 0.031 |
| glrm_list | FastRuntime | 0.250 → 0.250 | 1.333 → 1.292 | 1.380 → 1.432 | 0.008 → 0.008 |
| scoped_child | FastRuntime | 6.604 → 6.646 | 11.000 → 11.084 | 4.156 → 4.051 | 1.363 → 1.404 |
| exact_token_set | FastRuntime | 1.125 → 1.083 | 2.000 → 1.792 | 0.409 → 0.322 | 0.004 → 0.004 |

Across these cases, candidate/baseline mask p50 ratios range from 0.963 to 1.045; mask p99 ratios range from 0.896 to 1.022. Per-case p99.9 ratios range from 0.802 to 1.141.
This small matched corpus shows no broad mask-latency regression. It is **not a
statistical proof of unchanged tails across arbitrary schemas**; sub-microsecond
timers are quantized and individual scheduling outliers remain visible in the JSON.

**Build and link costs are not uniformly faster.** Some small warm-build cases
are slower under the redesigned frontend/component contract. The table retains
those costs rather than moving them outside the timer or claiming zero overhead.
For example, the arithmetic frontend and several small exact/list cases can have
noticeable relative differences despite small absolute times. The lifecycle sample
counts are deliberately reported: five warm compiles and seven warm links per trial
are descriptive, not cold-build or link-tail qualification. All preparation and
persistence times, artifact sizes and individual trial values remain inspectable
in the adjacent JSON.

## Reproduction

The adjacent `glrm-coordinated-performance-driver.rs` is the exact measured driver.
Create two small Cargo consumer projects pointing at the baseline and candidate
worktrees, with that file as their binary source and dependencies `glrmask` (path,
default features only) and `serde_json = "1"`. Declare an empty `redesign` feature;
enable it only for the candidate. Its only conditional adaptations are typed
external declaration spelling and `prepare` versus `compile_unlinked`.
Copy the original lockfile into each consumer and verify matching registry versions.
Build both with `cargo +1.95.0 build --release --offline`, no custom optimization flags.
Run each binary as `RAYON_NUM_THREADS=2 <binary> 8192 2000`, in alternating order,
without competing builds. Preserve the emitted JSON and compare mask fingerprints
and token traces before interpreting timing differences.

Driver SHA-256: `b255c27d941f0a154ac21a4cd879cccb277324bbbd541d332b64af976ede80ce`.
Candidate binary SHA-256: `89d225451abda53fd2c85631911746f4263138c2d412ccc7b4d03fcee080330c`.
Baseline binary SHA-256: `443c14229ebd76f4652f30cf31e62f74e0062ce6b5af556b0ccb167f751135b7`.

The initial new benchmark fixture used compact JSON, which is outside the existing
JSON Schema frontend’s canonical single-space comma/colon serialization. That fixture
was corrected identically for both versions before the recorded runs. No library
grammar or runtime policy was relaxed to make the benchmark pass.

## Scope of acceptance

Correctness qualification is documented separately in
`glrm-coordinated-2026-10-14.md`. This performance check qualifies the bounded
matched sample above, not the full 9,558-schema benchmark set, model sampling
probabilities under forcing, or final Windows execution. Existing published
benchmark assets are unchanged. No packages or release tags were published.
