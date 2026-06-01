# Definition of done for Chunk 11

Chunk 11 is done when the following are true:

- The runtime tree visibly separates artifact, state, Mask, and Commit.
- `ConstraintState` struct definition is short enough to understand on screen.
- cache and scratch types are no longer mixed with semantic state methods.
- Mask accumulator and bitset helpers are no longer buried in `mask/mod.rs`.
- Commit env/options, parser advance dispatch, token lookup, and mask assertion
  are no longer buried in `commit/mod.rs`.
- Documentation explains the denotation of every new file.
- The package includes a patch, source zip, checks, and this documentation.

Chunk 11 is not done merely because files were moved.  It is done because each
moved file has a mathematical identity and a review checklist.
