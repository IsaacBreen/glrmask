# Error and broadening policy

Publication-quality JSON Schema support must be honest about exactness.  The
importer should never look more complete than it is.

## 1. Categories

Every unsupported or partially supported construct must be placed in exactly one
category.

### Rejected

Use this when accepting the schema would likely produce surprising invalid JSON
values.  Examples:

- `uniqueItems`, unless exact finite-memory support is introduced,
- general `not`,
- `contains` with `minContains`/`maxContains`,
- nonlocal references.

### Annotation-only

Use this when JSON Schema itself treats the keyword as annotation or the importer
chooses a harmless annotation policy.  Examples may include `title`,
`description`, `default`, and unknown `format` names.

### Exact

Use this when the emitted grammar denotes exactly the accepted encoded JSON
texts under the importer's whitespace policy.

### Broadening

Use this only when the generated grammar is a superset and the broadening is
explicitly documented.  A broadening must have a test showing at least one value
that is accepted by the broad grammar but not by the exact schema if such a value
is easy to construct.

## 2. Error location format

All loader and lowerer diagnostics should use schema locations:

```text
#
#/properties/name
#/$defs/node
#/anyOf/3/properties/id
```

Avoid compiler-stage terms in these messages.  Users should not see Terminal DWA
or grammar automata names when their schema is unsupported.

## 3. Current policy decisions to preserve

- Exact subtraction lowering is enabled by default for open-object key languages.
- Unknown formats are annotations unless recognized by string lowering.
- General nonlocal references are unsupported.
- General `not` remains unsupported except for named mutually-exclusive object
  patterns.
- Some object `anyOf` collapses intentionally broaden to `json_object`; those
  paths must remain named and tested.

## 4. Checklist for a new diagnostic

Before adding a new error string, answer:

1. Is the error raised in loading, normalization, or lowering?
2. Does the message name the JSON Schema keyword?
3. Does the message include the schema location?
4. Does the message avoid internal compiler jargon?
5. Is there a regression test that checks the keyword and location?
