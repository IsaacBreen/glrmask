# Static shape expectations

Expected after this chunk:

```text
src/runtime/artifact/mod.rs exists
src/runtime/artifact/compiled.rs exists
src/runtime/artifact/cache_types.rs exists
src/runtime/artifact/caches.rs exists
src/runtime/artifact/token_space.rs exists
src/runtime/artifact/serialization.rs exists
src/runtime/artifact/finalize.rs exists
src/runtime/artifact/templates.rs exists
src/runtime/artifact/dense.rs exists
src/runtime/artifact/accessors.rs exists
src/runtime/bitmask_ops.rs exists
```

Expected absent:

```text
src/runtime/artifact.rs
src/runtime/serde.rs
src/runtime/token_space.rs
src/runtime/finalize.rs
```

Expected compile finalization marker:

```text
Constraint::from_compiled_parts(CompiledArtifactParts { ... })
```

Expected serialization markers:

```text
SERIALIZATION_FORMAT_VERSION
SERIALIZATION_MAGIC
SerializedArtifactEnvelope
SerializedArtifactFeatures
legacy fallback to bincode::deserialize::<Constraint>
```
