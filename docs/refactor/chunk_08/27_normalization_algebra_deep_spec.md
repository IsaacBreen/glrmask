# Normalization algebra deep specification

Normalization is the algebraic middle layer of JSON Schema import.  It operates
on value-level schema denotations, not on grammar expressions.

## 1. Denotational baseline

For a schema node `S`, define `[[S]]` as the set of JSON values accepted by the
supported JSON Schema semantics.  A normalization function `N` is acceptable if
it satisfies one of these contracts:

```text
exact:       [[N(S)]] = [[S]]
broadening:  [[S]] subseteq [[N(S)]]
rejecting:   N(S) is undefined with a diagnostic
```

Narrowing is not allowed unless the schema itself is contradictory and the
result is the empty language.  In code, that means no function should replace a
schema with a strictly smaller schema merely because it is easier to lower.

## 2. `allOf`

`allOf` is intersection.  Exact merge rules are allowed when both inputs are in
a family with a simple meet operation.

Object examples:

```text
required(A) meet required(B) = required(A union B)
properties(P) meet properties(Q) = pointwise meet on common properties plus union
additionalProperties(false) meet anything = additionalProperties(false)
```

Array examples:

```text
minItems(a) meet minItems(b) = minItems(max(a,b))
maxItems(a) meet maxItems(b) = maxItems(min(a,b))
```

Number examples should eventually move to rational arithmetic:

```text
minimum(a) meet minimum(b) = minimum(max(a,b))
maximum(a) meet maximum(b) = maximum(min(a,b))
```

Current inherited code uses `f64` in the schema model.  Chunk 08 deliberately
keeps that behavior but documents it as a publication target.

## 3. `anyOf`

`anyOf` is union.  Factoring is allowed when it preserves union denotation or
when a broadening is named and tested.

The important object-specific optimization is factoring variants that share a
serialized object shape.  For example, variants with disjoint required marker
properties can become one object-language automaton rather than a grammar choice
of many full object languages.  This is semantic union factoring, not parser
optimization.

The normalizer should state which of these is happening:

1. exact branch factoring,
2. branch subsumption removal,
3. broad collapse to `json_object`,
4. fallback to grammar choice.

## 4. `oneOf`

`oneOf` is exclusive union.  The current importer treats many `oneOf` shapes as
choice because exact mutual-exclusion checking is hard.  Publication-quality code
must mark each `oneOf` path as one of:

1. exact because branches are known disjoint,
2. broad because exclusive semantics are ignored,
3. rejected because broadening is unsafe for the target use case.

The code should never leave future readers wondering whether a `oneOf` choice is
exact.  Put the reason beside the lowering branch.

## 5. `not`

General negation is not a grammar-friendly operation over JSON values.  The only
acceptable `not` support is narrow, named, and proven by a local argument, such
as mutually exclusive object property presence patterns.  General `not` should
remain rejected until a finite complement construction is introduced.

## 6. Proof obligation template

Every new normalization helper should have a comment matching this form:

```text
Input shape:
  <schema shape>
Rewrite:
  <old schema algebra> -> <new schema algebra>
Contract:
  exact | broadening | rejecting
Reason:
  <one or two paragraphs>
Tests:
  <test names>
```
