# Exact subtraction has two levels

This chunk makes a distinction that the old file layout obscured.

## Level 1: named-grammar transform

Location:

```text
src/grammar_ir/transforms/exact_subtraction/
```

Type:

```text
NamedGrammar -> NamedGrammar
```

Purpose: rewrite exact subtraction sites before flat lowering, potentially generating helper named rules.

This pass can inspect the named grammar globally. It is part of frontend normalization.

## Level 2: local lowering helper

Location:

```text
src/grammar_ir/lower/exact_subtraction.rs
```

Type:

```text
Lowerer local method while emitting GrammarDef
```

Purpose: handle cases where lowering sees an exact alternative subtraction expression and can reduce it directly because the left-hand side names a nonterminal with explicit alternatives.

This helper is not a whole-grammar rewrite and should not generate a family of named helper rules. It emits through the flat lowerer context.

## Why the distinction matters

Putting both in one file makes it look like exact subtraction is one algorithm. It is not. The transform and the lowering helper have different inputs, outputs, scopes, and failure modes.

## Review questions

1. Does an exact-subtraction bug occur before or during lowering?
2. Does the code need access to all named rules as mutable syntax, or only to lowerer maps?
3. Does it generate named grammar helpers or flat productions?
4. Does the error belong to frontend normalization or flat lowering?

Use those answers to choose the right file.
