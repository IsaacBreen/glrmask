# Fast-path contracts

## Fast paths are not semantics

Fast paths are alternate implementations of the same transition relation. The reference shape remains the queue-based transition in `general.rs`. A fast path may return `None` when its preconditions are not met; it must never produce a result that differs from the general transition for the same state and bytes.

## Fast path files

`fast_path.rs` contains all unprofiled fast paths after this chunk. The file is still large because there are several distinct optimizations, but it is no longer mixed with the public API, pruning definitions, profile record construction, or the reference implementation.

## Required precondition style

Every fast path should be readable as:

1. check structural preconditions,
2. compute the same semantic observations as the reference path,
3. perform the specialized advance/update,
4. return `None` rather than guessing when the shape no longer matches.
