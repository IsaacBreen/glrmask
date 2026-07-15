# glrmask 0.1 native benchmark snapshot

This note records the bounded native benchmark used for the 0.1 release candidate. It is a hardware-specific snapshot, not a universal performance claim or a comparison establishing superiority over another backend.

## Exact measured code and machine

The performance-relevant production code was exactly:

```text
89559ad2600056e730439031ff2525c6b2c86632
```

The benchmark ran on:

- Hetzner CX23 in Helsinki (`hel1`);
- 2 shared AMD EPYC-Rome vCPUs;
- 4 GB RAM;
- Ubuntu 24.04 x86_64;
- Rust 1.97.0;
- Python 3.12.3;
- native release build with `RUSTFLAGS=-C target-cpu=native`.

These numbers are therefore specific to this machine and build. They are not generic-wheel timings.

The benchmark used CFA source commit `465a922c77529eee7155a42b90d3a546e1c20a94` with the cached Llama 3 vocabulary used by the benchmark environment.

## Workloads

| short name | workload |
|---|---|
| `bfcl-008` | `bfcl_catalog/size_008/catalog_008_000` |
| `bfcl-512` | `bfcl_catalog/size_512/catalog_512_000` |
| `json-glrm` | `grammar_glrm/json/json` |
| `github-o62060` | `jsb/data/Github_hard---o62060` |
| `vercel` | `jsb/data/JsonSchemaStore---vercel` |

CFA preprocessing that would alter source semantics was disabled:

```text
--no-strip-pattern-max-length
--no-coerce-one-of-to-any-of
```

## Methodology

Build timing used three repetitions per workload, except Vercel, which used one full-suite build because it is the deliberately long regression sentinel. The build table reports median, minimum, and maximum over those runs.

A separate Vercel sentinel used a 30-second build timeout and completed in `25.277667773 s`. The full-suite Vercel build took `24.925028667 s`.

Runtime timing requested 51 complete example traversals per successfully built workload. Run 0 was excluded as warmup. Token-step timings from runs 1 through 50 were pooled. Mask, commit, and combined mask+commit distributions were recorded separately.

For expected-valid labeled examples, glrmask had to replay the full example without mask rejection, commit rejection, sequence death, or validity mismatch. The checked-in JSON GLRM example has no explicit validity label; its replay completed under glrmask.

## Compile/build latency

| workload | runs | median | min | max | replay status |
|---|---:|---:|---:|---:|---|
| `bfcl-008` | 3 | 136.50 ms | 129.10 ms | 139.77 ms | valid gold replay passed |
| `bfcl-512` | 3 | 985.44 ms | 929.13 ms | 1,022.42 ms | valid gold replay passed |
| `json-glrm` | 3 | 63.11 ms | 59.70 ms | 81.82 ms | unlabeled example replay completed |
| `github-o62060` | 3 | 1,595.23 ms | 1,592.83 ms | 1,621.43 ms | valid gold replay passed |
| `vercel` | 1 | 24,925.03 ms | 24,925.03 ms | 24,925.03 ms | valid gold replay passed |

Vercel remains a long compile-tail case. On this shared 2-vCPU machine it completed in about 25 seconds, rather than reproducing the older greater-than-120-second timeout.

## Runtime distributions

Run 0 is excluded. Percentiles pool token steps from runs 1 through 50.

| workload | metric | samples | p50 µs | p95 µs | p99 µs | p99.9 µs | max µs |
|---|---|---:|---:|---:|---:|---:|---:|
| `bfcl-008` | mask | 2,550 | 2.355 | 3.898 | 5.215 | 26.676 | 35.036 |
| `bfcl-008` | commit | 2,550 | 2.195 | 6.923 | 9.595 | 28.029 | 32.981 |
| `bfcl-008` | mask+commit | 2,550 | 4.785 | 9.448 | 12.354 | 31.995 | 44.313 |
| `bfcl-512` | mask | 2,550 | 2.224 | 4.033 | 10.435 | 18.425 | 35.186 |
| `bfcl-512` | commit | 2,550 | 2.535 | 8.461 | 12.816 | 19.737 | 23.534 |
| `bfcl-512` | mask+commit | 2,550 | 4.893 | 11.106 | 17.970 | 25.753 | 40.597 |
| `json-glrm` | mask | 750 | 2.089 | 4.293 | 7.721 | 16.648 | 18.374 |
| `json-glrm` | commit | 750 | 3.526 | 10.220 | 17.583 | 25.370 | 27.171 |
| `json-glrm` | mask+commit | 750 | 5.646 | 14.665 | 21.914 | 32.855 | 33.462 |
| `github-o62060` | mask | 66,200 | 2.224 | 4.719 | 6.372 | 18.343 | 139.241 |
| `github-o62060` | commit | 66,200 | 4.198 | 9.067 | 14.707 | 24.760 | 1,420.941 |
| `github-o62060` | mask+commit | 66,200 | 6.743 | 12.224 | 20.229 | 31.735 | 1,423.737 |
| `vercel` | mask | 12,750 | 2.746 | 5.436 | 7.750 | 20.133 | 40.707 |
| `vercel` | commit | 12,750 | 5.410 | 12.233 | 17.734 | 27.606 | 114.285 |
| `vercel` | mask+commit | 12,750 | 8.657 | 15.539 | 22.899 | 34.956 | 116.680 |

Four of the five workloads recorded no runtime sample above 1 ms in any measured mask, commit, or mask+commit distribution. The exception was one isolated O62060 commit and combined mask+commit sample:

```text
O62060 mask+commit p99.9: 31.735 µs
O62060 mask+commit max:   1,423.737 µs
samples above 1 ms:      1 / 66,200
```

A claim that maximum latency is universally below 1 ms would therefore be false.

## Correctness and comparison caveats

All four expected-valid labeled workloads replayed fully under glrmask. The unlabeled checked-in recursive JSON GLRM example also replayed fully.

The benchmark also collected raw token-mask disagreements with `llguidance_native`, but those disagreements were not adjudicated as ground truth. They do not establish that either backend is correct or incorrect on a disputed token, and successful gold replay does not prove equality of the full token-mask language.

This benchmark does not support a claim of universal superiority over another backend.

## Public interpretation

The bounded release snapshot supports the following limited statements:

- the measured small BFCL tool schema compiled in about 0.14 seconds on this machine;
- the 512-tool BFCL catalog compiled in about 0.99 seconds;
- the recursive JSON GLRM grammar compiled in about 63 ms;
- O62060 compiled in about 1.6 seconds;
- Vercel remained the clear compile-tail case at about 25 seconds;
- combined mask+commit p99.9 was between about 25.8 and 35.0 µs across the five workloads;
- one isolated 1.42 ms combined runtime outlier occurred in 66,200 O62060 samples.

These are measurements of the exact commit and environment above, not hardware-independent guarantees.
