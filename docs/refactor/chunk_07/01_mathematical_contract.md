# Mathematical contract

This chunk is not primarily a filesystem cleanup. It is a correction of the mathematical object model.

## Objects

### `GrammarExpr`

`GrammarExpr` is syntax. It is not yet a parser production and it is not yet a lexer regular expression. Some variants have regular-language meanings (`Literal`, `CharClass`, `RawRegex`, `LexerDfa`, `AnyByte`), some have context-free meanings (`Sequence`, `Choice`, `Ref`, `RepeatRange`), and some are higher-level conveniences (`SeparatedSequence`, `ExprNFA`).

A `GrammarExpr` is allowed to contain source-level conveniences that the compiler will never see directly.

### `NamedRule`

A `NamedRule` binds a name to a `GrammarExpr` and records whether that name is terminal, nonterminal, or internal-only terminal. The flags matter because a `Ref` node does not by itself say whether it is a parser nonterminal edge or a terminal-body reference. That interpretation is defined by the surrounding `NamedGrammar`.

### `NamedGrammar`

`NamedGrammar` is an ordered set of named rules plus a start symbol and optional ignore terminal. It is the boundary object returned by frontends.

A `NamedGrammar` may still contain source-level conveniences. It is not the same thing as the flat compiler grammar.

### `GrammarDef`

`GrammarDef` is the flat compiler grammar. It has numerical terminal ids, numerical nonterminal ids, explicit productions, display names, and an optional ignore terminal. It is the representation consumed by GLR analysis and all downstream automata construction.

## Morphisms

### Transform: `NamedGrammar -> NamedGrammar`

A transform preserves the denoted language while changing grammar shape.

Examples:

- factoring common choices;
- simplifying singleton sequences/choices;
- exact-subtraction lowering;
- terminal choice promotion.

A transform must not allocate `TerminalID` or `NonterminalID` for `GrammarDef`. If it needs helper rules, it creates named helper rules in `NamedGrammar`.

### Lowering: `NamedGrammar -> GrammarDef`

Lowering changes representation. It is the first time the system allocates flat IDs and emits parser productions.

Lowering is therefore the first place where a `GrammarExpr::RepeatRange` or `GrammarExpr::SeparatedSequence` becomes concrete productions.

### Rendering: `NamedGrammar -> String`

Rendering is observational. Rendering must not rewrite grammar syntax, allocate compiler ids, or change parser semantics. It may expose non-Lark constructs as comments or GLRM-specific syntax, but it must remain a view.

### Parsing GLRM: `String -> NamedGrammar`

GLRM parsing is a source-language frontend. It belongs near grammar IR because it reconstructs `NamedGrammar`, not `GrammarDef`.

## Invariants introduced by the split

1. `ast.rs` has no lowering code.
2. `render/` has no lowerer state and no `GrammarDef` construction.
3. `lower/` is the only directory that may allocate flat ids from named grammar syntax.
4. `transforms/` never call `lower`.
5. `flat.rs` describes the compiler input after frontend lowering; it is not a source AST.
6. Old `grammar::*` imports remain shims only.
