# Serialization

This document will describe serialized `Constraint` artifacts and their compatibility policy.

## Baseline policy

The current implementation serializes compiled constraints for reuse. Before publication, the crate should state whether serialized artifacts are:

- stable across patch releases,
- stable only within the same crate version,
- tied to the exact vocabulary and options used at compile time,
- architecture-independent,
- or explicitly experimental.

## To document

- Format and version tag
- What is canonical data versus cache data
- How runtime caches are rebuilt on load
- Compatibility guarantees
- Failure modes for incompatible artifacts
