# Blue-sky object lowerer rewrite

A complete rewrite of object lowering would not start from textual JSON.  It
would start from finite map semantics.

## Semantic core

Represent an object schema as:

```text
Fixed: Map<KeyLiteral, SchemaNode>
Required: Set<KeyLiteral>
Patterns: Vec<(Regex, SchemaNode)>
Additional: AllowAny | Deny | SchemaNode
Bounds: min/max cardinality
```

Then derive a symbolic member classifier:

```text
class(k) =
  Fixed(i)                         if k is fixed key i
  Pattern(P)                       if k matches pattern set P and is not fixed
  Additional                       if k is not fixed and matches no pattern
```

For every key class, derive a value schema:

```text
value_schema(k) = intersection of all applicable fixed/pattern/additional schemas
```

The grammar problem becomes generating an ordered list of distinct keys whose
classes satisfy required/bounds constraints.  This can be represented as a DFA
over key classes plus value grammars, rather than fully enumerating permutations.

## Ideal algorithm

1. Build fixed-key bitset state: which required/optional fixed keys have
   appeared.
2. Track cardinality up to `maxProperties` or a saturation bound.
3. Track whether each `anyOf`/variant obligation is satisfied, if lowering a
   factored object variant.
4. For each next key class, emit key grammar and corresponding value grammar.
5. Reject duplicate fixed keys by transition absence.
6. Reject close until required bits and minProperties are satisfied.
7. Allow close if maxProperties is not exceeded and variant obligations hold.

This would make object lowering look like a small automaton construction instead
of a recursive expression-generation problem.  It would also align better with
the rest of the crate's automata-first paper narrative.

## Why not implement immediately

The current object lowerer has many benchmark-specific optimizations and domain
special cases.  A rewrite should be done only after a semantic oracle exists,
because object schemas are where silent overacceptance is most dangerous.
