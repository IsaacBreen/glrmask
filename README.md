# glrmask

glrmask is a Rust crate for grammar-constrained token masking during large language model decoding. It compiles a grammar and vocabulary once, then answers repeated masking and commit queries during generation.

The implementation is organized around the paper vocabulary:

- **Terminal DWA**: maps completed terminal sequences to lexer-state/token pairs.
- **Parser DWA**: maps parser stack prefixes to the lexer-state/token pairs that remain valid.
- **Mask**: walks active parser stacks through the Parser DWA and combines the encountered weights into a vocabulary mask.
- **Commit**: scans generated token bytes, completes terminals, and advances the GLR parser state.
- **Template DFA**: accelerates commit-time handling of repeated grammar templates where applicable.

This repository is being prepared for publication. The public API is intentionally small: users construct a `Vocab`, compile a `Constraint`, start a `ConstraintState`, then alternate between mask generation and committing accepted model output.

## Status

This snapshot is in cleanup. The first publication baseline establishes the repository shell, package metadata, documentation skeleton, Python binding location, and ignore rules. Deeper module restructuring is intentionally deferred to later chunks.

## Supported grammar sources

The current codebase contains frontends for:

- JSON Schema
- Lark grammars
- EBNF grammars
- the internal GLRM grammar format

The exact supported subset of each frontend should be documented before release, especially JSON Schema behavior around objects, strings, numeric constraints, and regular expressions.

## Rust quickstart

```rust
use glrmask::{Constraint, Vocab};

# fn main() -> glrmask::Result<()> {
let vocab = Vocab::from_tokens(vec!["{".into(), "}".into()]);
let schema = r#"{"type":"object"}"#;
let constraint = Constraint::from_json_schema(schema, &vocab)?;
let mut state = constraint.start();

let mask = state.mask();
# Ok(())
# }
```

The quickstart above is documentation-only at this stage. It should be verified and replaced with a compiling example once the source tree has been refactored and the test suite is restored.

## Python bindings

Python bindings live under `bindings/python` and build through maturin. The package name remains `glrmask` for continuity while the Rust crate is cleaned up.

```bash
cd bindings/python
maturin develop
```

## Documentation map

- `docs/architecture.md`: module map and how it corresponds to the paper.
- `docs/api_boundary.md`: public facade, diagnostics boundary, and token-space compatibility aliases.
- `docs/paper_mapping.md`: paper term to code module/type/function mapping.
- `docs/configuration.md`: compile/runtime options, profiling flags, and legacy environment variables.
- `docs/json_schema_support.md`: supported JSON Schema subset and known limitations.
- `docs/serialization.md`: serialized artifact format and compatibility policy.
- `docs/performance.md`: benchmark methodology and reproducibility notes.

## Development policy

Normal library API calls should not print to stdout/stderr. Diagnostics should be returned through profile structs or explicit options. Generated caches, local vocab dumps, macOS metadata, and benchmark output must not be committed.

## License

This repository is prepared with dual MIT/Apache-2.0 licensing. Confirm that this is the intended publication license before release.
