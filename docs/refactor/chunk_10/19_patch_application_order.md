# Patch application order

Apply this chunk after Chunk 09.

1. Replace top-level runtime artifact files with the new artifact directory.
2. Add `runtime/bitmask_ops.rs`.
3. Move cache rebuild methods out of `constraint.rs`.
4. Move artifact accessors into `artifact/accessors.rs`.
5. Update `runtime/mod.rs`.
6. Update compile finalization to use `CompiledArtifactParts`.
7. Add docs and static-check artifacts.

If a conflict occurs, preserve the newer Chunk 09 GLR paths (`crate::parser::glr`)
and reapply only the runtime-artifact edits.
