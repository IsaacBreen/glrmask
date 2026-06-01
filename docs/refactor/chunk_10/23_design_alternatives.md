# Design alternatives considered for Chunk 10

## Alternative A: rename `Constraint` to `CompiledArtifact` immediately

Rejected for this chunk.  It would force every public API binding and impl block
to move at once.  The mathematically cleaner endpoint is real, but the current
chunk is only the artifact/finalization boundary.  Keeping `Constraint` as the
public storage type preserves external shape while still moving construction
behind `CompiledArtifactParts`.

## Alternative B: introduce `Constraint { artifact: Arc<CompiledArtifact> }`

Deferred.  This is the likely final architecture, but it changes borrow patterns
inside `ConstraintState<'a>`, Python self-cell bindings, and many runtime method
receivers.  It should happen only after Mask and Commit are disentangled.

## Alternative C: serialize only semantic fields in a custom struct

Deferred.  The envelope currently contains `Constraint` because the existing
serde annotations already skip cache fields.  A custom semantic-only struct would
be cleaner but would duplicate every field and risk accidental divergence before
compile repair.

## Alternative D: keep old direct bincode format

Rejected.  A publication-facing artifact needs explicit version metadata, even
if the first version is simple.

## Alternative E: make `RuntimeCaches` the actual nested field now

Deferred.  That would require rewriting every direct field access in Mask,
Commit, and cache builders.  This chunk instead introduces `RuntimeCaches` as a
named aggregate for construction and documentation.  The nested-field migration
belongs after runtime algorithms are split.
