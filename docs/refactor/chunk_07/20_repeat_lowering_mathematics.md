# Repeat lowering mathematics

Repeat lowering is where regular cardinality constraints become context-free productions.

## Exact repeat

For a symbol `S`, exact repeat `Exact(S, n)` denotes:

```text
ε            if n = 0
S            if n = 1
S^n          otherwise
```

The implementation may construct a balanced tree, a left-linear tree, or a right-linear tree. Those are implementation choices.

## Bounded repeat

For `Range(S, min, max)`, the denotation is:

```text
Union_{k=min..max} S^k
```

The implementation may build this union by recursive range splitting or by composing exact/min/max helpers.

## Why balanced helpers matter

Long right-linear or left-linear expansions can create deep parse paths and many intermediate states. Balanced trees reduce depth. However, they must not change the accepted language.

## Cache correctness

A helper nonterminal is reusable only if all semantic parameters match. For repeat helpers, that means at least:

- repeated symbol;
- lower bound;
- upper bound;
- shape where shape changes emitted helper topology enough to matter.

The existing caches key by symbol and bounds. Shape is effectively controlled within a lowering run. If shape becomes a compile option, review whether it needs to enter cache keys.

## Test sketch

For a one-terminal symbol `a`, generate grammar for:

```text
a{0,0}
a{1,1}
a{0,2}
a{2,4}
a*
a+
```

Enumerate accepted strings up to length 5 and compare against expected cardinality sets.
