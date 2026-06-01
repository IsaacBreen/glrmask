# Proof obligations by phase
## Schema phase

Prove that every loaded schema node stores enough information to reconstruct the
supported validation assertions at that location.  No grammar-specific invariant
is allowed here.

## Load phase

Prove that each accepted raw JSON Schema keyword is either validation-relevant
and represented in `schema::*`, annotation-only and intentionally ignored, or
unsupported and rejected.

## Normalize phase

For each rewrite `S -> S'`, write either `⟦S⟧ = ⟦S'⟧` or `⟦S⟧ ⊆ ⟦S'⟧`.
The proof must mention object key order, required sets, and pattern-property
intersections when applicable.

## Lower phase

For each schema family, prove that the grammar expression emits exactly or
broadly the encoded JSON texts for the accepted value denotation.  The proof must
include string escaping and whitespace policy.
