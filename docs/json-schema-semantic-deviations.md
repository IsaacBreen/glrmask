# JSON Schema acceptance policy and semantic deviations

GLRMask compiles a documented JSON Schema subset into generation grammars. It
is not a full JSON Schema validator. Within that subset, compilation preserves
supported assertions; assertions that cannot be enforced produce an error.
The object serialization restriction below remains an intentional limitation.

## Dialects, annotations, and references

An absent `$schema` selects Draft 2020-12. Explicit Draft 4, 6, 7, 2019-09, and
2020-12 declarations select their keyword semantics. For example, older drafts
use tuple-form `items`, while Draft 2020-12 requires `prefixItems`; Draft 4 uses
boolean exclusive bounds. Draft 4/6/7 `$ref` objects ignore assertion siblings;
2019-09/2020-12 apply them. This does not imply support for every assertion in
each dialect.

Descriptive annotations and unknown keyword payloads are ignored. `const` and
`enum` values remain instance data; their contents are never scanned as schemas.
`format` is annotation-only in every supported standard dialect, including
unknown format names. Explicit required format-assertion vocabularies and
unknown required vocabularies produce unsupported-feature errors; optional
unknown vocabularies may be ignored.

Mixed dialect resources, named anchors, dynamic/recursive references, nested URI
resource scopes, and references into ignored annotation payloads are currently
unsupported. Supported local JSON-pointer references and legacy fragment aliases
retain their existing resolution behavior.

## Exact restricted assertions

`oneOf` retains exclusive semantics. The compiler accepts cases it can lower
exactly, including provably disjoint branches, and rejects unsupported overlap.
It no longer silently replaces `oneOf` with `anyOf`. No approximation mode is
provided: setting `GLRMASK_JSON_SCHEMA_COERCE_ONE_OF_TO_ANY_OF` to true, or setting
the internal `coerce_one_of_to_any_of` option, now produces an error.

Restricted exact cases remain useful:

- `not` supports boolean schemas and complements of supported primitive type
  sets; general negation and integer-only complements are unsupported.
- Conditionals support constant conditions and vacuous branches. Unsupported
  assertions are checked inside nested branches before simplification.
- `uniqueItems: false` has no effect. `uniqueItems: true` is supported when arrays
  are excluded or provably contain at most one item; general uniqueness is
  unsupported.
- Property counts, property-name constraints, and fixed-property dependencies
  have exact supported cases. Dynamic map minima above one require distinct
  required names to prove the minimum; counting repeated unknown key spellings
  cannot establish distinct data-model properties. Unsupported combinations produce errors instead
  of dropping their constraints. General schema dependencies and evaluated-item
  or evaluated-property tracking remain unsupported; vacuous cases may compile.

Malformed schemas, unsupported features, compiler resource limits, and proven
empty languages have distinct diagnostic text. A false schema or a proven
contradiction can represent an empty language. An unsupported assertion is never
used as evidence of contradiction. Diagnostics include a schema pointer where
available.

## Patterned string lengths preserve their assertions

Sibling `pattern`, `minLength`, and `maxLength` constraints are retained. Exact
simplification can prove an envelope redundant, prove the language empty, or
clamp a supported repetition. Otherwise the compiler constructs the exact
intersection or refuses it when its structural resource budget is exceeded.
The former behavior of dropping an expensive upper length bound has been
removed. Successful build counts can therefore decrease without indicating a
recognition regression.

## Additional and pattern properties are tail-only in objects

When lowering object schemas, glrmask preserves fixed properties in their
schema-lowered positions and only permits `additionalProperties` and
`patternProperties` in the free-property tail after the remaining fixed
properties for that branch.

This is a deliberate grammar-size and build-time tradeoff. Fully interleaving
fixed, additional, and pattern properties creates much larger unordered object
state spaces. In benchmark and regression experiments on recursive, object-heavy schemas such
as `Github_hard---o13029`, broad interleaving attempts caused severe build-time
blowups and timeouts.

Compared with full JSON Schema semantics, this can reject objects where a
non-fixed additional or pattern-matched property appears
before a later fixed property that the same branch still expects. Discrepancies
should only be classified under this deliberate deviation when acceptance truly
depends on that non-fixed property appearing before a later fixed property. Do
not apply this deviation when the relevant keys are themselves fixed properties
in the same branch.

## Canonical serialization

The generated language uses a space after each comma and colon. It may also
require fixed object keys and object literals in their lowered order. Integer
and numeric literal grammars retain their existing canonical spelling limits:
for example, a constant `1` need not accept the spelling `1.0`, and integer
schemas need not accept decimal or exponent spellings. These restrictions can
reject valid JSON representations of the same value; they do not permit an
instance that violates the supported assertions. Differential reports retain
such rejections separately from unsupported builds and invalid acceptances.
