# Static checks explained

The package includes static checks rather than compile results. They verify:

- old `mod.rs` line count has collapsed,
- expected Commit submodules exist,
- no `mask_game` terminology reappears in Commit,
- brace balance is zero for Commit Rust files,
- key function names are present in exactly one owning file,
- `runtime/commit/README.md` describes the new layout.

These checks are not a substitute for compilation. They are publication-shape checks for this no-compile phase.
