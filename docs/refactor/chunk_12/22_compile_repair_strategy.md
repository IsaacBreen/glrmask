# Compile repair strategy

Because this chunk is a no-compile architectural split, the first later compile pass should be mechanical and disciplined.

## Expected first errors

- missing imports after moving helpers into submodules,
- visibility too narrow or too broad,
- unused imports due to the shared routing-layer import style,
- private field access where a local record was moved to `types.rs`.

## Repair order

1. Repair syntax and module path errors without changing semantics.
2. Replace broad `use super::*` imports with explicit imports once the module graph stabilizes.
3. Reduce visibility from `pub(super)` to private when a helper is used by only one file.
4. Only after all mechanical errors are gone, run semantic tests and compare fast-path/reference behavior.

## Non-goal

Do not rewrite the transition algorithm while repairing imports. Algorithmic changes should be separate chunks with tests.
