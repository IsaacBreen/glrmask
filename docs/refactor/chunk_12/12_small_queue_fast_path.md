# Small queue fast path

The small-queue fast path specializes short byte fragments with small live-state counts. It uses `SmallVec` entries rather than hash maps at each offset.

This is a representation optimization, not a semantic optimization. The queue keys and transition rules are identical to `general.rs`; only the concrete collection type changes.

The reason this deserves a named file section is that small byte fragments are the common LLM token case. Publication code should make clear why the optimized representation is safe and where it falls back.
