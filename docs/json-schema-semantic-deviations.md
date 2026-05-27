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
