# Chunk 10 risk register

## R1: Visibility errors after moving impl methods into `artifact` submodules

**Mitigation:** Compile repair should first adjust `pub(crate)`/`pub(super)` on moved helper methods, not move logic back into `constraint.rs`.

**Review status:** open until the compile-repair pass validates this point.

## R2: Serialization envelope breaks older loaders

**Mitigation:** This is accepted: compatibility is forward-loading, not old-loader compatibility. Keep legacy fallback for new loader reading old artifacts.

**Review status:** open until the compile-repair pass validates this point.

## R3: Cache fields accidentally treated as semantic fields

**Mitigation:** Use `constraint_field_inventory.csv` and verify all `#[serde(skip)]` fields are rebuildable.

**Review status:** open until the compile-repair pass validates this point.

## R4: Compile finalizer still knows runtime layout

**Mitigation:** Review `src/compile/pipeline/finalize.rs`; it should construct `CompiledArtifactParts` only.

**Review status:** open until the compile-repair pass validates this point.

## R5: Large `artifact/caches.rs` remains intimidating

**Mitigation:** Accept as transitional for this chunk; split by cache phase after Mask/Commit disentanglement.

**Review status:** open until the compile-repair pass validates this point.

## R6: Unsafe bitmask ops become harder to audit after move

**Mitigation:** The move improves auditability by collecting ops in one file; the unsafe audit is deferred explicitly.

**Review status:** open until the compile-repair pass validates this point.

## R7: Token-space names regress to ambiguous token/state terminology

**Mitigation:** Keep `token_space.rs` aliases and docs; future code should use original/internal names in signatures.

**Review status:** open until the compile-repair pass validates this point.

## R8: Template DFAs remain in artifact rather than parser subsystem

**Mitigation:** The template DFA subsystem chunk will move or rename these more deeply; this chunk only localizes runtime storage.

**Review status:** open until the compile-repair pass validates this point.

## R9: Python binding behavior changes through serialization envelope

**Mitigation:** Python calls Rust `save/load`, so versioning applies automatically. Add Python serialization tests later.

**Review status:** open until the compile-repair pass validates this point.

## R10: No rustfmt means formatting may drift

**Mitigation:** This is intentional under the user instruction; run rustfmt only in validation chunk.

**Review status:** open until the compile-repair pass validates this point.

