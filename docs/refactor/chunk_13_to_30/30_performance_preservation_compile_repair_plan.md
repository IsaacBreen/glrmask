# Chunk 30 compile repair plan: performance_preservation

## Expected first errors

Because this pass intentionally prioritized architecture over compilation, expect these errors first:

1. unresolved imports caused by moved canonical modules;
2. private module visibility after moving files into deeper directories;
3. stale `super::` references inside files moved from one parent to another;
4. duplicate compatibility aliases;
5. rustfmt changes in generated examples or include files.

## Repair order

1. Fix module declarations.
2. Fix imports in pure modules.
3. Fix imports in compile/runtime orchestrators.
4. Fix visibility only as narrowly as needed.
5. Run rustfmt only after path errors are resolved.
6. Run unit tests before examples.
7. Run integration tests before benches.

## What not to do

- Do not delete shims before the first green compile.
- Do not rename public API symbols during compile repair.
- Do not tune performance while import errors remain.
- Do not silence warnings by adding broad `allow` attributes unless there is a documented publication reason.

## Final acceptance

This chunk is accepted when canonical imports compile and every old path either disappears or is confined to a documented compatibility shim.
