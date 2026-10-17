# JSON Schema benchmark

Measured **8 October 2026**, using native Rust implementations at GLRMask
[`f9f5c84f4de5d0d0098c5b68b9733f66727dc456`](https://github.com/IsaacBreen/glrmask/commit/f9f5c84f4de5d0d0098c5b68b9733f66727dc456)
and CFA `5d3feb4d1f06dca1bbfd0199fe5803b4e02cb80b`. These results describe
that source, not the published 0.2.0 packages. They match the JSON dataset on
the [public website](https://glrmask.com/#benchmarks).

## Measurements

- **TBM (µs):** commitment of a token allowed by the previous ready mask through
  the next ready mask. Calling-thread CPU time; each aligned position takes the
  minimum across two independent native timing processes.
- **Build (ms):** schema-dependent compilation wall time from the first process.
- **TTFM (ms):** schema-dependent build start through the first ready mask,
  measured with wall time from the first process. This includes compilation.

The machine ran macOS 27 ARM64 with 10 logical CPUs and Rust/Cargo 1.95.0.
Vocabulary preparation is outside the timer. There are no real-schema warmups
or prior-schema cache carryover; a synthetic global warmup is used. Production
validation is enabled. One problem runs at a time, with a 60-second build-only
timeout and no memory cap. Both frameworks use mimalloc with
`MIMALLOC_PURGE_DELAY=-1`. No experimental performance result is included.

The measured APIs are native Rust `Constraint::compile(Grammar::json_schema)` /
`ConstraintState::fill_mask` / `commit_token` for Static, and
`DynamicConstraint::compile(Grammar::json_schema)` /
`DynamicConstraintState::fill_mask` / `commit_token` for Dynamic. These are
the names at the measured revision; current public callers select
`Optimization::FastRuntime` and `Optimization::FastBuild`, respectively.
The comparator uses official llguidance **1.8.0**, native Rust
`ParserFactory` + `Matcher`, commit
`dbaf504d498b6aeede06ae57adc6f7c2c4848c59`, with native default forcing enabled.

## Corpus and cohorts

The full sweep contains all **9,558 official JSONSchemaBench `data/` schemas**
at `ba103c73756198dd9b149ddc7db7867da7a077f6`, with corresponding replay examples.
Handwritten, Synthesized, JME, MCPspec and BFCL additions are excluded.
The vocabulary is `unsloth/Meta-Llama-3.1-8B-Instruct` at
`a2856192dd7c25b842431f39c179a6c2c2f627d1`, with 128,256 token coordinates.
Replay excludes special-token IDs.

Examples preserve JSON values, but recursively order object properties by
schema declaration and serialize with comma-space and colon-space separators.
The same lexical policy applies to both frameworks. This is a controlled replay
comparison, not unrestricted end-to-end LLM generation. Timing excludes model
inference. The complete corpus has 35,339 examples (12,654 valid, 22,685 invalid);
1,072 schemas have no examples.

Build/TTFM uses joint-success schema cohorts across the reference/current
GLRMask pair and the fresh same-Mac reference/current llguidance pair. Static
has **8,326** matched schemas; Dynamic and the displayed llguidance cohort have
**8,327**. These cohorts differ by one schema; they are not all-9,558-schema
timing distributions.

TBM joins exact problem/example/step identities across each pair, omits examples
with runtime errors in any arm, and uses the shared replay prefix. Valid and
invalid trajectories are included up to that prefix. The three displayed
distributions have the same **2,925,983 aligned positions**, with prefix identity
SHA-256 `d88697aefe4ef77060aa8c5bda7cf0a6dae0a0141acd0b45e65a794eafe00020`.

Means are arithmetic averages: each of the 2,925,983 aligned positions has equal
weight for TBM; each matched schema has equal weight for TTFM. TTFM maxima are
the largest recorded sample-wise TTFM values in those same cohorts, not sums of
separate component maxima.

The table uses nearest-rank percentiles, with the conventional median averaging
the two central observations for even-sized samples. The SVG samples the
empirical inverse survival function; its percentile spacing expands the upper
tail logarithmically. Both axes and the measured maximum are labeled. Sampling
does not extrapolate a latency guarantee beyond observed data.

## Full-run outcomes

| Mode | Build OK | Build errors | Incorrect examples | Runtime-error examples |
|---|---:|---:|---:|---:|
| Static | 9,325 | 233 | 460 | 0 |
| Dynamic | 9,326 | 232 | 460 | 0 |
| llguidance 1.8.0 | 8,332 | 1,226 | 100 | 6 |

Incorrect examples are semantic failures. Build success and conditional replay
timings do not establish full-corpus correctness, full-mask equivalence or
language equivalence. The statistics are observations under the recorded
minimum-of-two policy, not confidence intervals or worst-case bounds. Separate
JavaScript and composition datasets on the website have different sources,
cohorts and measurement policies and are not pooled into these numbers.

## Provenance and reproduction

The [public aggregate snapshot](benchmarks/json-schema-2026-10-08.json) records
the exact source, tokenizer, corpus and membership hashes, measurement intervals,
native APIs, raw-file digests, statistics in nanoseconds and sampled tail data.
It contains no private raw records or local evidence paths.

Current measurement intervals (UTC):

| Mode | Start | End |
|---|---|---|
| Static | 2026-10-08 07:37:59.352444 | 2026-10-08 07:48:10.668505 |
| Dynamic | 2026-10-08 07:48:10.759349 | 2026-10-08 07:51:44.190365 |
| llguidance | 2026-10-08 07:56:24.227309 | 2026-10-08 07:59:39.658529 |

Regenerate documentation and SVGs without running benchmarks:

```bash
python scripts/sync-benchmarks.py
python scripts/sync-guide.py
```

See [documentation synchronization](documentation-sync.md) for the optional
website consistency check and the distinction between shared content and
website presentation.
