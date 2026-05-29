# JSON Schema semantic deviations

This document records intentional places where the JSON Schema importer does not
fully enforce the source schema semantics. Each entry must include the reason and
the expected impact.

## Patterned strings ignore length bounds

When a string schema has `pattern`, the importer compiles the pattern constraint
and intentionally ignores sibling `minLength` and `maxLength` constraints.

This is a deliberate performance tradeoff. Combining a broad JSON-string length
counter with a complex string pattern can create very large terminal DFAs. One
representative schema, `Github_hard---o62060`, contains:

- `serviceBenefits.items`: `maxLength: 120`, `pattern: ^(?:\S+\s+){0,9}\S+$`
- `serviceFeatures.items`: `maxLength: 120`, same pattern
- `serviceSummary`: `maxLength: 500`, `pattern: ^(?:\S+\s+){0,49}\S+$`

Compiling the `maxLength` counter intersected with the `{0,49}` word-count
pattern produced a tokenizer/terminal DFA with about 253k states and caused
large build-time regressions. Ignoring the sibling length bound keeps the
pattern enforcement, avoids that product construction, and may accept strings
that match the pattern but violate the ignored length bound.

## Additional and pattern properties are tail-only in objects

When lowering object schemas, glrmask preserves fixed properties in their
schema-lowered positions and only permits `additionalProperties` and
`patternProperties` in the free-property tail after the remaining fixed
properties for that branch.

This is a deliberate grammar-size and build-time tradeoff. Fully interleaving
fixed, additional, and pattern properties creates much larger unordered object
state spaces. In production experiments on recursive, object-heavy schemas such
as `Github_hard---o13029`, broad interleaving attempts caused severe build-time
blowups and timeouts.

Compared with full JSON Schema semantics and tools such as llguidance, this can
reject objects where a non-fixed additional or pattern-matched property appears
before a later fixed property that the same branch still expects. Discrepancies
should only be classified under this deliberate deviation when acceptance truly
depends on that non-fixed property appearing before a later fixed property. Do
not apply this deviation when the relevant keys are themselves fixed properties
in the same branch.
