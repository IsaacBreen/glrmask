# Chunk 10 review checklist

Review this chunk by answering the following questions.

## Artifact shape

- Does `src/runtime/artifact/mod.rs` give a clear reading order?
- Is `Constraint` now located in `artifact/compiled.rs`?
- Does `CompiledArtifactParts` contain only compile-produced semantic fields?
- Does compile finalization avoid spelling out every cache field?

## Cache separation

- Are derived cache types named in `cache_types.rs`?
- Is cache rebuilding artifact-local?
- Are cache fields still skipped by serde?
- Does load rebuild caches on both envelope and legacy paths?

## Token-space separation

- Are token-space quotient functions artifact-local?
- Are original/internal token ids documented?
- Are original/internal tokenizer-state ids documented?

## Serialization

- Does `save` use a versioned envelope?
- Does `load` reject unknown magic/version values?
- Does `load` keep a legacy fallback?
- Are feature flags present even if not yet used for migration?

## Runtime root hygiene

- Are top-level `runtime/serde.rs`, `runtime/finalize.rs`, and
  `runtime/token_space.rs` gone?
- Is `runtime/bitmask_ops.rs` present?
- Does `runtime/mod.rs` export the same public `Constraint` type?

## Non-goals respected

- Did this chunk avoid changing Mask algorithm semantics?
- Did this chunk avoid changing Commit algorithm semantics?
- Did this chunk avoid compile/test/rustfmt work?
