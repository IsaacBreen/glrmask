# Test design after the split

The split suggests more precise tests than the old monolithic grammar tests.

## AST tests

Test `prune_unreachable` on every recursive variant:

- `Grouped`
- `Sequence`
- `Choice`
- `Exclude`
- `Intersect`
- `Optional`
- `Repeat`
- `RepeatOne`
- `RepeatRange`
- `SeparatedSequence`
- `ExprNFA`

The test should assert that reachable terminal and nonterminal rule names are retained and unreachable ones are removed.

## Renderer tests

### Lark renderer

Test that renderer output includes:

- start comment;
- ignore directive;
- terminal/nonterminal sections;
- internal marker comments;
- comments for non-Lark-native constructs.

### GLRM renderer

Test round trips:

```text
NamedGrammar -> GLRM -> NamedGrammar
```

for:

- terminal literals;
- regex terminals;
- exact subtraction;
- ExprNFA definitions;
- internal terminals.

## Lowering tests

### Repeat tests

For small counts, compare generated language against expected strings over a one-symbol alphabet. This can be done by enumerating derivations up to a small depth after lowering.

### Separated sequence tests

Construct cases where:

- all items required;
- some items optional;
- required nullable item appears between required nonnullable items;
- optional nullable item appears at ends;
- repeated item appears in a separated sequence.

The key regression is that `required nullable` must not be treated as absent.

### Terminal expression tests

Test terminal-body reference cycles, unresolved terminal references, and `Expr::Shared` reuse.

### ExprNFA tests

Existing tests should remain near `expr_nfa.rs` or `lower/expr_nfa_lower.rs`, depending on whether they test automaton construction or lowering.

## Transform tests

Each transform should get tests framed as:

```text
input NamedGrammar -> transformed NamedGrammar -> lower -> language sanity
```

Do not test transforms only by looking for exact helper names unless helper-name determinism is the actual property.

## Compile-repair tests

After the first compile pass, add a smoke test that imports both old and new paths:

```rust
use crate::grammar::ast::GrammarExpr as OldGrammarExpr;
use crate::grammar_ir::ast::GrammarExpr as NewGrammarExpr;
```

This verifies shim compatibility while migrations proceed.
