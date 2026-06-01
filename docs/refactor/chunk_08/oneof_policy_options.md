# `oneOf` policy options

JSON Schema `oneOf` means exactly one branch validates.  A grammar choice means
at least one branch has a derivation.  These coincide only when branches are
pairwise disjoint over JSON values.

## Option A — exact exclusive-one

Construct for branches `S_1..S_n`:

```text
⋃_i ( S_i ∩ ⋂_{j≠i} complement(S_j) )
```

This requires complement support for the relevant regular/value fragment.  It is
hard for arbitrary object schemas but possible for restricted literals, primitive
types, and closed object variants.

## Option B — disjointness proof plus choice

Before lowering `oneOf` as choice, prove branches are pairwise disjoint.  Useful
sufficient conditions:

- disjoint primitive types;
- closed object variants with distinct required discriminators;
- enum/const literal sets with no overlap;
- numeric ranges with empty intersections;
- string formats/patterns only if regex disjointness is decidable in the chosen representation.

## Option C — reject non-disjoint unknown shapes

Reject `oneOf` when disjointness cannot be proven.  This is the safest
publication default if exact complement is not implemented.

## Option D — compatibility broadening mode

Provide an option named something like
`JsonSchemaOneOfPolicy::TreatAsAnyOfCompatibility`.  It must be opt-in and loudly
documented because it changes semantics by broadening.
