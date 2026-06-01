# Chunk 10 static checks

No compile, test, benchmark, or rustfmt run was performed. These are static shape checks only.

- [x] expected file exists: src/runtime/artifact/mod.rs
- [x] expected file exists: src/runtime/artifact/compiled.rs
- [x] expected file exists: src/runtime/artifact/cache_types.rs
- [x] expected file exists: src/runtime/artifact/caches.rs
- [x] expected file exists: src/runtime/artifact/token_space.rs
- [x] expected file exists: src/runtime/artifact/serialization.rs
- [x] expected file exists: src/runtime/artifact/finalize.rs
- [x] expected file exists: src/runtime/artifact/templates.rs
- [x] expected file exists: src/runtime/artifact/dense.rs
- [x] expected file exists: src/runtime/artifact/accessors.rs
- [x] expected file exists: src/runtime/bitmask_ops.rs
- [x] old top-level file absent: src/runtime/artifact.rs
- [x] old top-level file absent: src/runtime/serde.rs
- [x] old top-level file absent: src/runtime/token_space.rs
- [x] old top-level file absent: src/runtime/finalize.rs
- [x] runtime/mod.rs has no top-level serde module
- [x] runtime/mod.rs has no top-level finalize module
- [x] runtime/mod.rs has no top-level token_space module
- [x] compile finalizer uses CompiledArtifactParts
- [x] compile finalizer calls from_compiled_parts
- [x] Constraint struct lives in artifact/compiled.rs
- [x] CompiledArtifactParts defined
- [x] serialization marker present: SERIALIZATION_FORMAT_VERSION
- [x] serialization marker present: SERIALIZATION_MAGIC
- [x] serialization marker present: SerializedArtifactEnvelope
- [x] serialization marker present: SerializedArtifactFeatures
- [x] serialization marker present: legacy
- [x] only one pub struct Constraint definition — src/runtime/artifact/compiled.rs
- [x] chunk 10 docs directory exists
- [x] chunk 10 docs include at least 30 files — 37

## Runtime artifact LOC
- `src/runtime/artifact/accessors.rs`: 109 lines
- `src/runtime/artifact/cache_types.rs`: 125 lines
- `src/runtime/artifact/caches.rs`: 1296 lines
- `src/runtime/artifact/compiled.rs`: 298 lines
- `src/runtime/artifact/dense.rs`: 16 lines
- `src/runtime/artifact/finalize.rs`: 20 lines
- `src/runtime/artifact/mod.rs`: 30 lines
- `src/runtime/artifact/serialization.rs`: 88 lines
- `src/runtime/artifact/templates.rs`: 24 lines
- `src/runtime/artifact/token_space.rs`: 156 lines
