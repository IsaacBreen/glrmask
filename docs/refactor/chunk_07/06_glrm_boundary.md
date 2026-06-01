# GLRM boundary

GLRM used to be a single file that did both parsing and rendering. Chunk 07 splits this into two conceptual sides.

## Parser side: `grammar_ir::glrm`

`grammar_ir::glrm::from_glrm` parses text into `NamedGrammar`. It is a frontend. It owns:

- tokenization;
- GLRM grammar parsing;
- escape decoding;
- ExprNFA reconstruction;
- parser errors.

It does not own rendering.

## Renderer side: `grammar_ir::render::glrm`

`render::glrm::to_glrm` formats `NamedGrammar` as GLRM text. It owns:

- rule emission;
- expression dumping;
- ExprNFA dumping;
- byte and regex escaping.

It does not parse.

## Re-export

For compatibility, `grammar_ir::glrm` re-exports `to_glrm`, and old `grammar::glrm` re-exports the new module. This preserves old calls such as:

```rust
crate::grammar::glrm::to_glrm(&grammar)
crate::grammar::glrm::from_glrm(text)
```

New code should prefer:

```rust
crate::grammar_ir::render::glrm::to_glrm(&grammar)
crate::grammar_ir::glrm::from_glrm(text)
```

## Later cleanup

In a later frontend restructuring chunk, `from_glrm` may move to `frontend::glrm`, while `grammar_ir::glrm` remains as the exchange-format parser implementation or disappears behind the frontend facade.
