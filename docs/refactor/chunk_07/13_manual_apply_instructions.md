# Manual apply instructions

These instructions describe how a human could reproduce Chunk 07 from the previous tree.

## Step 1: Create the new namespace

Create:

```text
src/grammar_ir/
  mod.rs
  ast.rs
  flat.rs
  expr_nfa.rs
  lower/
  transforms/
  render/
  glrm/
```

Add `pub(crate) mod grammar_ir;` to `src/lib.rs` immediately after the old `grammar` module declaration.

## Step 2: Move syntax definitions

Move the following from `src/grammar/ast.rs` into `src/grammar_ir/ast.rs`:

- `GrammarExpr`
- `CommaSepShape`
- `NamedRule`
- `NamedGrammar`
- `NamedGrammar::terminal_names_set`
- `NamedGrammar::prune_unreachable`

Do not move `Lowerer` into `ast.rs`.

Replace `NamedGrammar::to_lark` with a delegate call to `crate::grammar_ir::render::lark::to_lark(self)`.

## Step 3: Move renderer code

Move Lark-like printing helpers into `src/grammar_ir/render/lark.rs`:

- `to_lark`
- `grammar_expr_to_lark`
- `grammar_expr_to_lark_with_indent`
- `u8set_to_class_def`
- `escape_byte`
- `regex_escape_byte`

Make byte/regex escape helpers crate-visible if lowering still uses them.

## Step 4: Move lowering code

Create `src/grammar_ir/lower/mod.rs` and move:

- `Lowerer`
- `lower`
- general expression emission methods
- terminal registry methods
- shared `char_class_pattern`

Then split specialized methods into children:

- repeat methods -> `repeat.rs`
- separated sequence methods -> `separated_sequence.rs`
- ExprNFA methods -> `expr_nfa_lower.rs`
- terminal expression conversion/nullability -> `terminal_expr.rs`
- local exact subtraction -> `exact_subtraction.rs`

Use `impl super::Lowerer` in child modules.

## Step 5: Move named grammar transforms

Create `src/grammar_ir/transforms/`.

Move:

- `factoring.rs` -> `transforms/factor.rs`
- `named_simplify.rs` -> `transforms/simplify.rs`
- `terminal_choice_promotion.rs` -> `transforms/terminal_choice.rs`
- `exact_subtraction_lowering.rs` -> `transforms/exact_subtraction/mod.rs`

Move exact-subtraction tests into `transforms/exact_subtraction/tests.rs`.

## Step 6: Split GLRM

Move GLRM rendering functions to `render/glrm.rs`.

Move parser/lexer functions to `glrm/mod.rs`.

Move GLRM tests to `glrm/tests.rs`.

Have `glrm/mod.rs` re-export `to_glrm` for compatibility.

## Step 7: Replace old grammar files with shims

Each old `src/grammar/*.rs` file should contain only a compatibility re-export.

Example:

```rust
//! Compatibility shim for `crate::grammar::flat`.

pub use crate::grammar_ir::flat::*;
```

## Step 8: Add documentation

Add:

- `src/grammar_ir/README.md`
- `src/grammar_ir/lower/README.md`
- `src/grammar_ir/transforms/README.md`
- `src/grammar_ir/render/README.md`
- `docs/refactor/chunk_07/*`

## Step 9: Do not compile yet

Stop here. The next phase can compile and fix import/visibility errors without rethinking the structure.
