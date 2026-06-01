# Configuration and diagnostics

This document will list every public option, profile hook, and legacy environment variable.

## Policy

- Public behavior should be controlled by typed options, not scattered environment variable reads.
- Legacy `GLRMASK_*` environment variables may remain temporarily for benchmark compatibility, but they should be parsed in one location and documented here.
- Normal library calls should not print diagnostics. Profiling information should be returned to callers or emitted through an explicit diagnostics/tracing configuration.

## To fill in during later chunks

- Compile options
- JSON Schema frontend options
- Runtime/mask options
- Commit options
- Profiling options
- Legacy environment-variable compatibility table
