# Composition validation — 23 September 2026

Validated production source: `2033f8cab8851eb488d92bfc573fdb4180491cc0`. This record concerns **composition only**,
not the independent global dynamic-tail/llguidance or Windows build experiments.
Machine-readable results are in
[`assets/composition-validation-2026-09-23.json`](assets/composition-validation-2026-09-23.json).

## Delivered architecture and exactness

Static ambiguous-boundary evaluation uses the shared ordinary GSS/static-DWA
queue, with correlated accumulator groups preserved. The separate boundary-DAG
evaluators are compiled only as test oracles. Static component masks are copied
wordwise and mask-only component shadows avoid unused commit parser pools.

Dynamic composition uses the ordinary shared vocabulary traversal with scoped
finite/configuration providers, including virtual and epsilon structures. The
old independent recursive full walker is test-only, not a silent production
fallback. Exact parser admission and lexer futures are batched/lazily intersected;
product and result caches do not approximate token acceptance.

The subtree shortcut requires an accepting exact product state and already
computed exact self-loops for **every byte in the remaining trie subtree**.
Every suffix therefore remains in that same product. Missing transitions or an
exhausted cache decline the shortcut. Product identity includes correlated
parser/lexer state and guards. Both adaptive output polarities and original
duplicate token IDs are tested. No link-aware preprocessing was moved outside
the measured linking call.

The final storage correction constructs the scoped product cache only for
providers that use it. An optional heap-owned cache removes the unused large
cache from ordinary masking's stack frame. This changes storage, not a state
transition, admission condition, or exact fallback.

Integration also preserves current public-API zero-byte token semantics:
ordinary empty byte aliases cannot act as no-progress model tokens; an ID can
still be admitted through a live exact special-token path. This is applied in
the composition adapter, not by changing the internal no-op meaning of
`commit_bytes(empty)`.

## Validation and baseline identity

Release validation passed: **794 root library tests, 259 external integration/API
and regression tests, 194 GLR library tests, and one doctest**. There were no
failures; 51 root and two GLR tests remained ignored. The 129 older integration/
end-token tests are a subset of the 259, not an additional count.

The real selected10 fixture has 128,256 vocabulary entries and three traces,
covering 11,767 mask positions. Each full replay checks masks, token admission,
acceptance and commits immediately in memory. CSV files retain timings and
hashes, not full mask streams. Sparse IDs, aliases, empty/special IDs, both output
polarities, cache-disabled/capacity-limited paths and save/reload are additionally
covered by focused tests.

The ordinary performance baseline is `6784b190f` **plus only the existing
lexer-lane and unfinished-ignore commit correctness corrections**. Unmodified
main fails the previously documented trace 0, step 420 case. Its original binary
and failure were retained; timings on an incorrect mask are not treated as a
valid reference. The corrected control does not contain composition masking,
product-cache, parser-batching or allocator optimisations. Prior integrated
runtime comparisons use separately frozen source/binary identities.

Benchmarks used native release builds with Rust 1.95, `RUSTFLAGS="-C
target-cpu=native"`, incremental compilation disabled, and
`RAYON_NUM_THREADS=1`. Heavy work was isolated; sources were frozen during each
build and run. Four-run A/B/B/A controls retain every original cold observation;
paired-min summaries take the minimum per matching position across the two runs
of each version. They are noise-reduced summaries, not formal upper bounds.

## Dynamic composition results

Times below are milliseconds. Ordinary and uncached composition are from one
joint run using the **same final binary and token trajectory**. The live-cache
row is a separate exact run and is not presented as uncached traversal.

| Runtime | P50 | P90 | P99 | P99.9 | Observed maximum |
|---|---:|---:|---:|---:|---:|
| Ordinary dynamic, same binary | 3.090 | 12.294 | 13.954 | 17.339 | 246.944 |
| Composed dynamic, uncached | 2.480 | 5.348 | 7.137 | 8.134 | 58.278 |
| Composed dynamic, live cache enabled | 1.751 | 5.060 | 6.830 | 7.456 | 60.842 |

Composition was faster at 11,458 of 11,767
positions (97.37%) in the joint run.
The historical comparison of roughly 3.5 ms composed versus 0.97 ms monolithic
median used an older runtime/reference context; it is not the final same-source
comparison. Relative to the pre-subtree integrated baseline, the earlier balanced
uncached run reduced total masking time by about 28%. The subsequent storage
correction preserved that gain (paired total-time ratio
0.997615 versus the prior integrated runtime).

Fresh cold first-mask medians were
60.509 ms before and
59.720 ms after the storage correction.
All eight individual process observations are retained in the JSON record.
Cold maxima from different runs vary; none are advertised as formal bounds.

## Ordinary-path controls and the rejected regression

Final paired total-time ratios versus the correctness-aligned ordinary baseline
were 0.999988 for static masking and
0.999204 for actual dynamic masking:
effectively neutral overall. Individual wall-time positions still vary; aggregate
neutrality alone was not used to waive a repeatable regression.

An earlier candidate caused a confirmed approximately 9.7% thread-CPU slowdown
at trace 2, step 3128. That candidate was held rather than merged. The optional
scoped-cache storage correction removed its large unused ordinary stack-frame
cost. Final focused A/B/B/A measurements use the exact full prior history and
retain the original live call separately from 100 raw cloned-state calls.
Numbers below are milliseconds, before / after:

| Trace / step | Original live thread CPU | Raw-clone median thread CPU |
|---|---:|---:|
| 0/2047 | 7.226 / 6.942 | 6.852 / 6.795 |
| 0/2805 | 7.146 / 7.170 | 6.751 / 6.785 |
| 2/3128 | 11.955 / 11.780 | 11.452 / 11.422 |

The raw focused measurements, including maxima and any off-CPU delays, are in
the JSON record. Cloning occurs outside these timers; dynamic bridge mask-vector
allocation is included equally on both sides. Static first-call outliers were
also rechecked and retained rather than discarded. No uniform pointwise speedup
is claimed.

## Linking and static composition

The final pure-static trace matched every mask and commit. Composition P50 was
43.208 microseconds and P99 was
64.167 microseconds. Its raw wall maximum
was 2.318 ms at trace 0, step 535.
That isolated wall spike did not recur in the exact-history static CPU check:
original live CPU was 53.000 microseconds before / 54.042 after, and the raw-clone
median was 50.250 / 51.531 microseconds. The original 2.318 ms wall observation is
retained, not replaced by an older or best-case maximum. Focused raw measurements
and the full static summary are in the JSON record.

The fully static selected10 linking comparison fell from 11.970 / 12.158 seconds
on the correctness-aligned main control to 2.894 / 2.809 seconds on the integrated
implementation. The final storage-only check measured 2.811 seconds and emitted
the identical 40,139,213-byte static artifact. All before/after artifacts were
checked on the real trace under `GLRMASK_STRICT_STATIC_TRAP_DYNAMIC=1`.

The root-CALL/child-entry candidate proof retains only necessary boundary pairs
at the same valid cut and does not skip full exact child-boundary equivalence.
Remaining static linking time includes genuine boundary equivalence and
normalisation. It is not claimed to be a microsecond static compiler.

Standalone component loading/preparation can precede linking; no earlier step
knows the caller's link graph. The measured static link contains the entire
composition call. Serialization is measured separately. The real dynamic bind
probe measured 6.978 ms; seven-repeat controls covered all four prepared/raw and
shared/borrowed combinations. Warm raw modes were about 2.68–2.96 ms and prepared/
shared about 0.246 ms, with preparation and parent-clone costs reported separately.
The previously pathological raw dynamic binding path is not hidden by reporting
only prepared/shared timing.

## Reproduction and evidence

The committed examples are `composition_real_dynamic_control`,
`composition_real_binding_control`, `composition_static_history_probe`, and
`composition_build_static_artifact`. They require the selected10 component cache,
vocabulary dump, monolithic/composed artifacts and trace file used for this
measurement; they do not download fixtures automatically.

```sh
# Build with the measured compiler/toolchain and native release flags.
cargo build --release --example composition_real_dynamic_control \
  --example composition_real_binding_control \
  --example composition_static_history_probe \
  --example composition_build_static_artifact

# True dynamic / dynamic comparison, LEFT ordinary, RIGHT composition.
DYNAMIC_REFERENCE=1 GLRMASK_DISABLE_DYNAMIC_MASK_CACHE=1 RAYON_NUM_THREADS=1 \
  target/release/examples/composition_real_dynamic_control replay \
  "$VOCAB" "$MONOLITHIC" "$DYNAMIC_COMPOSED" "$TRACES" joint.csv

# Explicit actual dynamic history control (not a static profile).
PROBE_DYNAMIC=1 GLRMASK_DISABLE_DYNAMIC_MASK_CACHE=1 RAYON_NUM_THREADS=1 \
  target/release/examples/composition_static_history_probe \
  "$VOCAB" "$MONOLITHIC" "$TRACES" 2 3128 history.csv 101
```

Legacy CSV column names describe positions, not guaranteed engines:
`static_mask_ns` is LEFT and `dynamic_mask_ns` is RIGHT. `DYNAMIC_REFERENCE=1`
forces LEFT dynamic. The replay's `PROFILE_STEP` mode profiles RIGHT; do not use
it to claim a forced-dynamic LEFT measurement. Use `PROBE_DYNAMIC=1` in the
history control when that distinction matters.

Final replay SHA-256: `7b8f09a631c92203949bfd4d4104ef37a23f8eb5c804b13c288c59297dc2a61b`. Source hashes and structured final results
are included in the JSON record. Full local timing CSVs, commands, logs, frozen
binary manifests and failed/rejected attempts remain under
`.benchmarks/composition-final-584582/` in the integration worktree and the
composition handoff note. They are intentionally not replaced by only a chart.

## Limits and remaining optimisation opportunities

This validates the supported shared composition architecture and this measured
workload; it is not a claim about every possible grammar or a formal latency
bound. Approximately 2.6% of the joint-run positions still favour ordinary
dynamic masking. Some known-literal suffix states retain additional scoped
provider work. Those are future local optimisation opportunities, not evidence
that the former whole-median gap remains. Static linking still does real work,
and observed cold masking is not eliminated. Independent global TBM-tail and
build-time work remains a separate task.
