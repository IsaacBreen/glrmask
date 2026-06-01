# Diagnostics and per-advance records

Per-advance diagnostics are lower-level than normal Commit profiling. They snapshot parser stacks before and after individual terminal advances and store the byte slice that induced the match.

This is useful for validating template-DFA stack effects and debugging parser-frontier explosions. It is not part of the core runtime API and should remain opt-in.

`profiled.rs` now owns diagnostic construction, while `profile.rs` owns the public record structures.
