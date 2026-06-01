# Risk register for JSON Schema importer cleanup

## R1 — oneOf semantics

Current code contains places where `oneOf` lowers as ordinary choice.  That is
not the same as exclusive-one JSON Schema semantics when branches overlap.  The
publication decision must be explicit:

1. implement exact exclusive-one for the supported regular fragment;
2. reject overlapping/unknown `oneOf` branches;
3. provide a named compatibility mode that treats `oneOf` as `anyOf`.

## R2 — numeric precision

The schema layer currently stores numeric bounds and `multipleOf` using `f64`.
This can misrepresent decimal JSON Schema constraints.  The target representation
is exact decimal rational arithmetic.

## R3 — broadening fallbacks

Several existing fallbacks intentionally broaden parser-shaped intersections to
choices.  The code must identify these sites and state the inclusion relation.

## R4 — conditionals and annotation-like ignoring

`if`/`then`/`else` should not be silently ignored unless the accepted fragment
proves the ignore is semantics-preserving.  Prefer explicit rejection or an
implemented normalizer.

## R5 — benchmark-shaped heuristics

Object lowering contains schema-specific names and thresholds.  Publication code
should move these into generically named strategies or mark them as benchmark
compatibility hacks outside the core semantic path.

## R6 — regex engine limits

String pattern lowering may broaden on regex compile limits.  Each such case
needs a test and a diagnostic/profiling hook so users know precision was lost.
