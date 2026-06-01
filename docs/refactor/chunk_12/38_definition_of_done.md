# Definition of done for Chunk 12

Chunk 12 is done when:

- `runtime/commit/mod.rs` is no longer the implementation blob;
- public commit methods live in `api.rs`;
- reference transition lives in `general.rs`;
- fast paths live outside the reference transition;
- profiling lives outside the unprofiled transition;
- pruning, queueing, acceptance, and single-top shortcuts each have named homes;
- documentation explains the mathematical relation and the source split;
- remaining large files are identified explicitly as deferred work.

This chunk satisfies that shape definition. Compile repair and algorithmic improvement are intentionally later tasks.
