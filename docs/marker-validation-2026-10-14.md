# Passive streaming markers: validation and performance

Validated on 14 October 2026. This report accompanies the local feature commit; it is not a package release or a main-branch deployment.

## Result

All selected canonical correctness gates passed: **2,549 Rust tests**, with **56 pre-existing ignored tests** left under their normal defaults, and **232 Python tests**. No new marker test is ignored. The exact release binary was separately benchmarked on 24 synthetic cases. The previously observed dense-LR-marker and moderate-inactive-frontier bottlenecks were resolved.

The scope is passive, existential boundary reporting. Markers do not consume tokens, mutate the parsing frontier, run marker transitions, construct capture histories, or choose a derivation. Observations can belong to alternatives that later fail. The API and interpretation contract are in [Streaming markers](markers.md).

## Implemented contract

`@mark(id)` is accepted by GLRM and EBNF and by the explicit Lark LLM profile. IDs are application-chosen `u32` values unrelated to model-token IDs. `Constraint::marker_ids()` lists distinct IDs; `ConstraintState::marker_events()` borrows the initial or latest actual-commit batch of `MarkerEvent { id, byte_offset }`. Python returns integer IDs and `(id, byte_offset)` tuples. Existing commit return types are unchanged.

Initial events, internal model-token boundaries, UTF-8 byte offsets, repeated/sparse IDs, ambiguity, ignored input, nullable/recursive grammars, exact special events, cloning, failed commits and end-token policy have explicit tests. Event batches are reused and sorted by `(byte_offset, id)`. Marker reporting does not execute inside mask or forcing probes. The caller owns any retained history.

Prepared source bindings, native compiled-child linking, nested components, ordinary save/load, and external-vocabulary artifacts preserve typed marker metadata. The optional `GLRMARK1` body envelope validates lengths, ordering, terminal range and incompatible lexical/event/control roles. Unannotated bodies keep the existing wire format.

## Why the slow paths changed

**Exact LR admission was repeated for every marker.** FastBuild uses an ExactSimulation policy, so merely treating any action-row entry as admissible would have been incorrect. The implementation instead compiles read-only input domains for the selected marker terminals, once during compilation/load, in the existing LR stack coordinate system. Output/action programs used during construction are discarded; the ordinary LR recognizer remains the runtime parser. Top certificates handle simple cases, and suffix-sensitive predicates query the existing stack without changing it.

**Moderate lexer reset frontiers missed an existing shortcut.** The common bounded scanner formerly used its exact first-byte reset projection only after a scratch-capacity threshold was exceeded. Consequently 16/128 inactive families could be slower than 512. The scanner now uses a prepared projection before reconstructing a reset epsilon closure. The cache is built from only the reset and reached one-byte closures, including at packed fast-load publication, rather than materializing all-state closure tables or moving work into first commit. This is a grammar-independent optimization, not a marker/schema special case.

**The representation audit closed two edge gaps.** Empty concrete stack languages are now observed consistently in GSS, flat, virtual and interned-language adapters, including mixed empty/nonempty languages and a drained virtual prefix with a hidden shared floor. The bounded lexer preserves all distinct accepting end states at the same longest width, matching the generic lexer oracle rather than collapsing by terminal ID.

A separate generic vocabulary-partition coverage bug exposed by randomized testing was also fixed: a sole partial family no longer drops tokens outside that family. The normal exact refinement supplies its common dead class. The standalone grammar package also no longer gates a helper that an unconditional public API requires.

## Correctness evidence

| Gate | Passed | Existing ignored |
|---|---:|---:|
| Root library + integration | 1,617 | 51 |
| Lexer | 269 | 3 |
| Grammar standalone | 120 | 0 |
| GLR | 268 | 2 |
| Terminal automata | 275 | 0 |
| Python complete test directory | 232 | 0 |

The root total uses the final result for each actual Cargo test target, not every embedded subprocess result printed into the log. Repeated diagnostic runs are not added to this total.

Randomized marker tests completed **10,672 comparisons** of reporting-enabled and reporting-disabled commits on the same compiled parser. They compare exact frontier, masks and acceptance. They additionally compare each optimized marker predicate with the original parser admission operation at represented fresh-lexer configurations. Distinct tests compare annotated and erased grammar masks and acceptance.

Reset projection tests compare all 256 first bytes and longer/empty inputs across moderate and large frontiers, all three explicit modes, and fresh/loaded artifacts. A low-level test verifies that reset preparation does not populate the unrelated all-state closure cache. The multiple-endpoint fixture uses the public same-terminal-domain tokenizer union and compares full `(terminal, width, end_state)` triples.

Commands used for the canonical gates (task-local development profile with debug assertions retained):

```sh
cargo +1.95 test -p glrmask --features internal-api --tests -- --test-threads=2
cargo +1.95 test -p glrmask-lexer --features internal-api --lib -- --test-threads=2
cargo +1.95 test -p glrmask-grammar --lib
cargo +1.95 test -p glrmask-glr --features internal-api --lib
cargo +1.95 test -p glrmask-terminal-dwa --features internal-api --lib
python -m pytest python/tests -q
python scripts/sync-guide.py --check
```

## Release benchmark methodology

The baseline is revision `de1a09b54700448db8ebc6691ab77a33b9ec9d5c`. Baseline and candidate use the identical public-API harness in `examples/marker_bench.rs`, Rust 1.95, release optimization, incremental compilation disabled, two build jobs, and identical task-local stripping settings on an Apple M1 Pro / 16 GB Mac running macOS 27. Baseline and candidate were built in separate target directories.

Baseline binary SHA-256: `2d4dd0df403ada37eb8ac496a698e435f6fa09bae029dfcf2fb854bb544cdc1e`.

Final candidate binary SHA-256: `98d3275c2001a04ef5f0570348a09a00b0b7098c6c9ddbcfa6d1fce75996060c`.

The vocabulary has 2,065 spellings. Eight workloads run in FastBuild, Balanced and FastRuntime. Three counterbalanced blocks each run the old unannotated baseline, new unannotated candidate and annotated candidate; each run contains 51 samples of 16 sequence repetitions. Compilation and `start()` are outside commit timing. Mask timings follow actual commits, with legitimate cross-commit cache reuse retained in all builds. Runs were serialized under the machine isolation gate; no compilation ran concurrently.

Every table entry below is the **median of three run medians of sequence-average nanoseconds per model token**, converted to microseconds. These are not individual-token p99/p100 measurements. Readings below a few tens of nanoseconds of difference can reflect code layout, caches and timing noise; a slightly negative incremental marker cost is not evidence that reporting is free or faster.

### Final commit timings

| Mode | Workload | Old baseline, no markers (µs) | Final, no markers (µs) | Final, markers (µs) |
|---|---|---:|---:|---:|
| Balanced | alternating single tokens | 1.067 | 1.062 | 1.144 |
| Balanced | dense 16 markers | 0.858 | 0.867 | 1.101 |
| Balanced | fused eight lexemes | 6.181 | 6.331 | 6.637 |
| Balanced | inactive 128 markers | 0.880 | 0.862 | 0.865 |
| Balanced | inactive 16 markers | 1.606 | 0.842 | 0.848 |
| Balanced | inactive 512 markers | 0.883 | 0.874 | 0.880 |
| Balanced | long fixed lexeme in chunks | 0.384 | 0.372 | 0.376 |
| Balanced | nested parentheses | 0.945 | 0.933 | 1.140 |
| FastBuild | alternating single tokens | 0.656 | 0.654 | 0.725 |
| FastBuild | dense 16 markers | 0.497 | 0.481 | 0.768 |
| FastBuild | fused eight lexemes | 2.709 | 2.698 | 3.117 |
| FastBuild | inactive 128 markers | 0.502 | 0.492 | 0.490 |
| FastBuild | inactive 16 markers | 1.254 | 0.473 | 0.487 |
| FastBuild | inactive 512 markers | 0.509 | 0.492 | 0.504 |
| FastBuild | long fixed lexeme in chunks | 0.446 | 0.428 | 0.441 |
| FastBuild | nested parentheses | 0.689 | 0.691 | 0.865 |
| FastRuntime | alternating single tokens | 0.864 | 0.868 | 0.883 |
| FastRuntime | dense 16 markers | 0.856 | 0.853 | 1.116 |
| FastRuntime | fused eight lexemes | 4.830 | 4.831 | 4.920 |
| FastRuntime | inactive 128 markers | 0.871 | 0.865 | 0.878 |
| FastRuntime | inactive 16 markers | 0.859 | 0.851 | 0.870 |
| FastRuntime | inactive 512 markers | 0.870 | 0.878 | 0.901 |
| FastRuntime | long fixed lexeme in chunks | 0.380 | 0.376 | 0.383 |
| FastRuntime | nested parentheses | 0.769 | 0.764 | 0.843 |

### Previously failing performance cases

| Case | Initial marker implementation (µs) | Final implementation (µs) |
|---|---:|---:|
| FastBuild: dense 16 markers | 14.046 | 0.768 |
| FastBuild: nested parentheses | 4.539 | 0.865 |
| Balanced: inactive 128 markers | 4.025 | 0.865 |

For unannotated grammars, final/old commit ratios range from **0.378× to 1.024×** in this matrix. The largest increase is about **2.4%**; the broad earlier unmarked slowdown was not reproduced in the final matrix. The moderate-frontier generic optimization improves several unannotated workloads substantially. This is bounded empirical evidence, not a universal zero-overhead guarantee.

Unannotated mask-after-commit ratios range from **0.965× to 1.044×** against the old baseline. Annotated/unannotated mask ratios within the final build range from **0.959× to 1.029×**. Marker observation is absent from mask traversal; annotations can still affect grammar representation and cache behavior, so identical source code paths are not a claim of mathematically identical wall-clock latency.

## Build/load trade-off and limits

Selected LR marker predicates are additional immutable preparation paid during compile/load. That work is not free and is not deferred into `start()`, the first mask, or first marker commit. The retained-LR backend is not replaced by another parser. The predicate inventory and candidate index have explicit resource limits and reject unsupported/excessive construction rather than silently omitting observations.

A separate same-interpreter, unoptimized lifecycle diagnostic compared 12 cases with nine repetitions against a frozen pre-domain module. For example, the dense-16 FastBuild fixture measured about 35.4→51.4 ms compile and 12.0→13.4 ms load. Other native controls varied by several milliseconds as well. These diagnostic values are not production TTFM numbers, and not all variation can be attributed to LR-domain preparation. First-commit work was not used to hide the new preparation cost.

This validation does not establish individual-token tail bounds, all possible grammar behavior, GPU serving throughput, or a complete parse/provenance interpretation of markers. Ignored input, nullable children and rejected alternatives deliberately retain their documented possible-boundary semantics.

## Reproduction and retained evidence

The structured results and gate counts are in [markers-2026-10-14.json](benchmarks/markers-2026-10-14.json). The benchmark executable can be rebuilt with `cargo +1.95 build --release --example marker_bench`; run its `plain 51 16` and `marked 51 16` modes on the appropriate revisions. Use distinct target directories and verify binary identity before comparing different worktrees.

On the task Mac, all nine final raw run summaries/logs and the counterbalanced runner are retained under `Projects2/glrmask-marker-artifacts-469935/performance-final/` and `run_marker_comparison_final.py`. Canonical test logs, source fingerprints and final binary hashes are under `Projects2/glrmask-marker-work-469935/logs/`. The earlier invalid cross-worktree cached candidate was caught by equal binary hashes and failure to parse `@mark`, renamed `INVALID-marker-candidate-baseline-cache-reuse`, and excluded from every reported candidate result.

The final release source was fingerprinted before and after compilation. A subsequent correction to the test-only overlapping-lexer fixture changed only the section under `#[cfg(test)]`; its production prefix and every other Rust/build file were verified unchanged. The accepted tests and final benchmark therefore cover the same production implementation.

Both Mac Obsidian compaction notes retain setup, decisions, commands, failures, fixes, benchmark provenance, disk-cleanup audits and the final commit handoff. Source, benchmark evidence and installed Python environments were preserved during authorized cleanup. No package release or default-branch modification is part of this change.
