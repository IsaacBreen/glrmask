# Paper mapping

This file tracks the intended one-to-one mapping between exposition terms and code names.

| Paper/exposition term | Current code area | Cleanup target |
| --- | --- | --- |
| Sep1 | crate-level algorithm identity | Mention in README/docs; do not rename the crate in this baseline chunk. |
| Terminal DWA | `src/compile/terminal_dwa/` | Promote to a named `terminal_dwa` subsystem. |
| Parser DWA | `src/compile/parser_dwa/` | Keep prominent and document with the same notation as the paper. |
| Scan / `CanMatch` | `src/compile/scan_relation/` and runtime tokenizer scanning | Split compile-time scan relation from runtime token scanning. |
| Mask | `runtime/mask` and `ConstraintState::mask/fill_mask` | Explain as stack walk plus encountered-weight combination. |
| Commit | `runtime/commit` | Split scanner, parser advance, validation, and fast paths. |
| Weights/masks over lexer-state/token pairs | `ds/weight.rs` and runtime dense/sparse mask code | Move representation and operations to an explicit weights/masks module. |
| Template DFA | `src/compiler/stages/templates/`, `runtime/commit/template_advance.rs` | Treat as a first-class cross-cutting artifact. |

Every later code move should update this mapping so readers can verify that implementation and paper agree.
