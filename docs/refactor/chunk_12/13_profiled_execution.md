# Profiled execution

`profiled.rs` contains profiled versions of the Commit transition and per-advance diagnostics. Keeping profiling separate from `general.rs` has two benefits:

1. The reference transition can be read without instrumentation noise.
2. Profiling can accumulate extra summaries, stack snapshots, and timing fields without redefining Commit.

The profile path is allowed to do additional observation work. It is not allowed to change state-update semantics. The correct mental model is:

```text
profiled_commit(S, b) = (commit(S, b), observations)
```

not a different transition relation.
