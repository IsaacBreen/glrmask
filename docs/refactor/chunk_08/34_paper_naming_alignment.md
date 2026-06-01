# Paper naming alignment for JSON Schema frontend

The paper-facing code should use names that match the mathematical story.  JSON
Schema import is upstream of Terminal DWA, scan relation, Parser DWA, and runtime
masking.  It should therefore avoid names that imply token masking internals.

## 1. Preferred paper-language terms

Use these terms in JSON Schema docs and comments:

```text
schema denotation
value language
encoded JSON text language
grammar IR
normalization
lowering
reference graph
finite object language
residual additional-key language
exact subtraction
broadening
```

## 2. Terms to avoid in importer comments

Avoid these unless the code genuinely touches that concept:

```text
mask
parser DWA
Terminal DWA
scan relation
token IDs
internal token space
GSS
runtime state
```

The importer should produce grammar.  The compiler and runtime decide how grammar
becomes masks.

## 3. Renaming recommendations

Current inherited names that should eventually change:

```text
Lowerer                         -> SchemaToGrammarLowerer
lower_json_literal              -> lower_canonical_json_literal
JSON_STRING                     -> JSON_TEXT_STRING or JSON_ENCODED_STRING
string_pattern_as_body_regex    -> decoded_pattern_to_json_body_regex
collect_shared_ap_exclusion_plan -> plan_shared_additional_key_exclusions
```

Do not do all renames in one compile-repair pass.  Rename at module boundaries
first, then internal helpers after tests are green.

## 4. Comment style

Every module-level comment should answer:

1. What is the input mathematical object?
2. What is the output mathematical object?
3. Is the transformation exact?
4. Where are broadenings documented?
5. Which later stage consumes the output?
