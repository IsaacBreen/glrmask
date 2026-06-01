# Chunk 18 implementation manual: configuration_profiling_diagnostics_logging

## Scope

src/config + src/diagnostics

## Exact files to open first

1. `src/config/env.rs`
2. `src/config/mod.rs`
3. `src/diagnostics/cache.rs`
4. `src/diagnostics/frontend.rs`
5. `src/diagnostics/logging.rs`
6. `src/diagnostics/mod.rs`

## Mechanical procedure

1. Open the canonical module boundary file before editing children.
2. Read the directory README and confirm the denotation it claims.
3. For every import that uses an old path, choose one of two actions: update to the canonical path, or leave only inside a compatibility shim.
4. For every public or crate-visible symbol, classify it as constructor, transformer, evaluator, policy, reporter, or compatibility.
5. Move constructors and evaluators into semantic modules; keep reporters in diagnostics/profiling modules.
6. Preserve old names only as `#[doc(hidden)]` shims.
7. Do not change algorithmic logic unless a move forces a path update.
8. Do not add environment-variable reads to pure files.
9. Add or update README text whenever a directory boundary changes.
10. Record every deliberate non-split large file as future mechanical extraction, not as forgotten work.

## Beginner-level edit recipe

- If you see a file whose name says only `mod.rs` and it is longer than 250 lines, look for obvious groups separated by comments.
- If a group contains option parsing, move it to `options.rs`.
- If a group contains print or profile formatting, move it to `profile.rs` or diagnostics.
- If a group contains helper structs used only by one algorithm, keep it near that algorithm.
- If a group defines a mathematical carrier type used by many algorithms, move it upward into a named domain module.
- After each move, search for the old path across `src`, `bindings`, `examples`, `tests`, and `benches`.

## Definition of complete for this chunk

- The target directory exists.
- The compatibility directory, if any, contains only shims.
- Canonical source files import canonical paths.
- Documentation names the denotation, forbidden dependencies, and validation checks.
- The changeset explains why the new grouping is mathematically better.
