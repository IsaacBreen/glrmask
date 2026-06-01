# Proof obligations by module

This document states what each module must prove or preserve. It is written as a review guide, not as a formal proof script.

## `grammar_ir::ast`

### Obligation A1: references are syntax, not resolution

`GrammarExpr::Ref(String)` does not by itself decide whether it refers to a terminal or a nonterminal. The interpretation happens in a `NamedGrammar` context. Reviewers should reject any future code in `ast.rs` that assumes a naming convention is enough to determine terminalness.

### Obligation A2: reachability follows only syntactic references

`NamedGrammar::prune_unreachable` traverses `GrammarExpr::Ref` edges. It must recursively traverse every expression variant that can contain a child expression. If a new variant is added, the reachability traversal must be updated in the same commit.

### Obligation A3: `to_lark` is a delegate

The `to_lark` convenience method may remain on `NamedGrammar`, but implementation must remain in `render::lark`. This preserves user ergonomics without re-entangling AST and renderer code.

## `grammar_ir::lower`

### Obligation L1: lowering is representation-changing

`lower` is the only function in grammar IR that creates `GrammarDef`. It may allocate ids and helper rules. Transform modules may not.

### Obligation L2: terminal rules lower through lexer expressions

A terminal rule body must become a lexer-level `Expr` or literal/pattern terminal. A nonterminal rule body must become parser productions. Internal-only terminals may be used by terminal bodies but must not independently produce parser productions.

### Obligation L3: helper names are internal

Generated helper names must begin with `__`. `nonterminal_names` should filter out those helpers so user-facing diagnostics show source names where possible.

### Obligation L4: duplicate emitted productions are deduplicated after emission

The old behavior deduplicated rules preserving first occurrence. This chunk preserves that step. If deduplication moves later, it must remain semantics-preserving.

## `grammar_ir::lower::repeat`

### Obligation R1: exact-repeat language

For symbol `S`, exact repeat of count `n` must denote `S^n`. Count zero denotes epsilon. Count one denotes `S`.

### Obligation R2: range-repeat language

For symbol `S`, range repeat `[m,n]` must denote the union of `S^k` for every `k` in `m..=n`.

### Obligation R3: tree shape is not language shape

`Left`, `Right`, `Balanced`, `Countdown`, and `LeftBalanced` may change the grammar topology, recursion direction, and number/depth of helper productions. They must not change the accepted language.

### Obligation R4: cache keys include symbol identity

It is invalid to cache only by numeric bound. `S{2}` and `T{2}` need different helper nonterminals.

## `grammar_ir::lower::separated_sequence`

### Obligation S1: optional absence differs from nullable presence

This is the central law. A required nullable item may produce no bytes/tokens, but it is still structurally present. It can still require separators around it depending on surrounding items. Optional absence removes the item from the sequence.

### Obligation S2: no local epsilon for optional subtree emptiness

The recursive lowering returns `can_be_empty`. It must not blindly emit `nt -> ε` at every optional subtree because that can create dangling separators in enclosing rules.

### Obligation S3: separator threading through repetition

If an item is itself `RepeatOne(item)` inside a separated sequence, it must lower to `item (sep item)*`, not bare `item+`.

## `grammar_ir::lower::terminal_expr`

### Obligation T1: terminal-body references resolve only through terminal bodies

A `Ref` inside a terminal body must resolve to a terminal rule. If no terminal body exists, lowering must error.

### Obligation T2: terminal cycles are reported

Terminal-body reference cycles must be detected through the `visiting` set.

### Obligation T3: nullability accounts for lexer-level terminals

`CharClass`, `RawRegex`, and `LexerDfa` nullability cannot be decided by context-free syntax alone; they must use lexer expression semantics.

## `grammar_ir::transforms`

### Obligation X1: transforms preserve named grammar type

No transform in this directory should return `GrammarDef`.

### Obligation X2: transforms do not read parser/DWA runtime state

Transforms happen before parser table construction. They must not depend on GLR state ids, Terminal DWA states, Parser DWA states, or runtime masks.

## `grammar_ir::render`

### Obligation V1: renderers are pure views

Renderers must not modify the grammar or allocate compiler ids.

### Obligation V2: unsupported target syntax is represented honestly

Lark rendering cannot faithfully express every internal variant. It may use comments for constructs with no direct Lark equivalent, but it must not silently drop them.
