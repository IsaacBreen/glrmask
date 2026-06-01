# Lowering deep dive

Lowering is the most mathematically delicate part of the grammar layer because it is where source syntax turns into flat productions. This chunk does not try to redesign the lowering algorithm completely; it makes the existing algorithm legible enough that such redesign is possible later.

## Lowerer state

The shared `Lowerer` state still contains:

- emitted rules;
- terminal pattern map;
- terminal list;
- nonterminal id map;
- generated helper counter;
- display-name maps;
- internal terminal names;
- named rule expression map;
- terminal body map;
- terminal expression cache;
- repeat-helper caches.

That shared state is acceptable for this chunk because splitting it prematurely would require compile/test work. The important improvement is that method groups are now separated by denotation.

## Repeat lowering

A repeat expression denotes a regular language. The lowering implementation translates that regular cardinality constraint to context-free productions. The tree shape matters for grammar size and parser behavior but must not change the accepted language.

Core obligations:

1. `RepeatRange { min, max }` with `min == max` must match exactly `min` copies.
2. `RepeatRange { min, max }` with `min < max` must match the union of exact counts `min..=max`.
3. `Repeat` must include epsilon.
4. `RepeatOne` must exclude epsilon unless the repeated expression itself is nullable.
5. Caches must be keyed by both the repeated symbol and the bound.

The repeat module keeps the policy enum private to lowering, while exposing just enough to the parent module to choose and pass tree shapes.

## Separated-sequence lowering

A separated sequence is not equivalent to `item (sep item)*` when optional items and nullable items are involved. The old code had a warning comment for this. Chunk 07 turns that warning into a file boundary.

Key law:

```text
optional absence != nullable derivation
```

If an item is required but nullable, it may be structurally present while deriving empty. It still participates in arity and separator placement. Treating it as absent can admit strings with dangling separators or reject strings that should be accepted.

The separated-sequence module therefore returns both:

- a symbol for the subtree;
- a `can_be_empty` flag.

The flag is not an instruction to emit epsilon locally. It is a signal to the parent split that a subtree may be absent when forming separator alternatives.

## ExprNFA lowering

`ExprNFA` is an automaton whose transition labels are grammar expressions. It is already structured as a state graph, so the natural lowering is:

```text
automaton state -> helper nonterminal
transition      -> production
accept state    -> production into caller lhs
```

Nullable and nonnullable lowering variants remain separate because nonnullable construction must exclude paths that consume no symbol.

## Terminal expression conversion

Terminal expressions are lowered to lexer-level `Expr` objects. This is not parser lowering. It is conversion from grammar-syntax fragments to scanner recognizers.

The terminal conversion module is also the natural location for nullability because nullability requires knowing when a terminal recognizer can accept epsilon.

## Local exact alternative subtraction

The local exact-subtraction helper handles source expressions like `A - B` when `A` denotes a named nonterminal with explicit alternatives. It is separated from the whole-grammar exact-subtraction transform because it has a different scope and different preconditions.

The whole-grammar transform rewrites `NamedGrammar` before lowering. The local helper only fires opportunistically during emission.
