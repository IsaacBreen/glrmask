# Manual audit commands for Chunk 10

Run these after applying the patch.

```bash
find src/runtime/artifact -maxdepth 1 -type f | sort
```

Should list all artifact submodules.

```bash
rg "mod serde|mod finalize|mod token_space" src/runtime/mod.rs
```

Should return nothing.

```bash
rg "CompiledArtifactParts|from_compiled_parts" src/compile/pipeline/finalize.rs src/runtime/artifact
```

Should show the compile-to-runtime handoff.

```bash
rg "SERIALIZATION_FORMAT_VERSION|SerializedArtifactEnvelope|SerializedArtifactFeatures" src/runtime/artifact/serialization.rs
```

Should show the versioned serialization boundary.

```bash
rg "pub struct Constraint" src/runtime
```

Should show exactly `src/runtime/artifact/compiled.rs`.

```bash
rg "pub\(crate\) type OriginalTokenId|InternalTokenId|OriginalTokenizerStateId|InternalTokenizerStateId" src/runtime/artifact/token_space.rs
```

Should show coordinate aliases.

```bash
rg "or_dense_buf|andnot_dense_buf|copy_dense_buf" src/runtime/bitmask_ops.rs
```

Should show low-level bitmask operations in one place.
