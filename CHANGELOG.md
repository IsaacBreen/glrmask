# Changelog

All notable user-facing changes should be recorded here.

This project follows a pre-1.0 compatibility policy until the public API and serialized artifact format are finalized.

## Unreleased

### Repository cleanup

- Added root crate metadata and workspace configuration.
- Moved Python bindings from `python/` to `bindings/python/`.
- Added publication documentation skeletons.
- Added repository ignore rules for build outputs, local caches, benchmark outputs, and macOS metadata.
- Removed uploaded archive artifacts such as `__MACOSX`, `.DS_Store`, and local vocab caches from the working tree.

### Deferred

- No implementation modules were refactored in this chunk.
- No compile, test, or benchmark validation has been run for this chunk.
