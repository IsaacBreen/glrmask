# Chunk 07 definition of done

This chunk is done when all of the following are true.

## Structural done

- `src/grammar_ir` exists.
- `src/grammar_ir/ast.rs` owns named grammar syntax.
- `src/grammar_ir/lower` owns conversion to `GrammarDef`.
- `src/grammar_ir/transforms` owns named grammar transforms.
- `src/grammar_ir/render` owns Lark and GLRM rendering.
- `src/grammar_ir/glrm` owns GLRM parsing.
- Old `src/grammar` files are shims.

## Mathematical done

- Syntax, transforms, lowering, and rendering are different directories.
- The lowerer is not in `ast.rs`.
- Renderers are not in `ast.rs`.
- Whole-grammar exact subtraction and local lowering-time exact subtraction are in different files.
- Separated-sequence lowering has a dedicated module.
- Repeat lowering has a dedicated module.

## Documentation done

- `src/grammar_ir/README.md` explains the namespace.
- `src/grammar_ir/lower/README.md` explains lowerer files.
- `docs/refactor/chunk_07` contains self-contained review docs.
- `glrmask_chunk_07_CHANGESET.md` exists.
- `glrmask_chunk_07_CHECKS.md` exists.

## Explicitly not required yet

- successful compile;
- successful tests;
- rustfmt;
- global import migration;
- env-var option cleanup;
- removal of compatibility shims.
