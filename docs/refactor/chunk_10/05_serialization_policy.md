# Serialization policy

## Previous format

Before this chunk, `save` serialized `Constraint` directly with bincode.  That
had two problems:

1. there was no explicit version marker;
2. old/new formats could only be distinguished by trying to deserialize them.

## New format

`save` now serializes:

```text
SerializedArtifactEnvelope {
    magic: "glrmask.constraint",
    format_version: 1,
    features: SerializedArtifactFeatures,
    constraint: Constraint,
}
```

## Feature flags

The envelope records whether the artifact includes:

```text
final_internal_token_map
internal_token_bytes
terminal_display_names
```

These flags do not currently alter load behavior.  They are deliberately present
so future loaders can reject artifacts missing required fields or perform
compatibility migration.

## Legacy fallback

`load` first tries the versioned envelope.  If that fails, it tries the old
direct-`Constraint` bincode format.  This means:

```text
new loader reads new artifacts
new loader reads old artifacts
old loader does not necessarily read new artifacts
```

That is the correct compatibility direction for a publication cleanup.

## Cache rebuilding

Both the new envelope path and the legacy fallback path call:

```text
constraint.rebuild_runtime_caches()
```

before returning.

## Explicit non-goal

This chunk does not attempt a stable cross-language or cross-major-version
binary format.  It merely introduces a versioned boundary and documents the
current compatibility policy.

## Future versioning rules

A future incompatible change should:

1. increment `SERIALIZATION_FORMAT_VERSION`;
2. add a migration function if old artifacts can be mapped forward;
3. add a feature flag if the new field is optional or derivable;
4. add tests with a small fixture from the previous version;
5. document whether serialized artifacts are intended for long-term storage or
   only for same-version cache reuse.
