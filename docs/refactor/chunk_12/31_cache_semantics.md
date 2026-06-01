# Commit cache and scratch semantics

Commit receives scratch buffers through `CommitBuffers`. This chunk does not redesign those buffers, but it clarifies where they should be used:

- scanner execution caching belongs near scanner execution,
- terminal advance caching belongs in `terminal_advance.rs`,
- queue-local maps belong to `general.rs` or fast-path code,
- public API methods should not know about cache internals except to pass buffers through.

The long-term target is for Commit to have a small transition context object that borrows buffers explicitly and documents cache lifetime.
