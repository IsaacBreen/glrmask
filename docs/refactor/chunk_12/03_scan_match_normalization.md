# Scanner matches and normalization

## Problem

A tokenizer execution may report duplicate `(width, terminal)` pairs through different internal paths. Commit must not advance the parser frontier twice for the same semantic observation because that would duplicate parser branches and distort profiling.

## New source boundary

`acceptance.rs` owns `collect_unique_actionable_matches` and `NormalizedMatch` is defined in `types.rs`.

The normalization key is:

```text
(width, terminal_id)
```

not just `terminal_id`. The width matters because a terminal match at width 1 and a terminal match at width 3 induce different byte offsets and therefore different future scanner executions.

## Linear versus hash path

Small match lists use a linear scan because allocation and hashing dominate the common case. Larger match lists use a reusable `FxHashSet` where available. This is a performance choice only. Both branches must compute the same set of normalized observations.
