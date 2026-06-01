# JSON importer test plan

## Loader tests

- boolean schemas true/false;
- object schemas with each supported keyword;
- wrong keyword type for every supported keyword;
- unsupported keyword at root and nested locations;
- nested `$defs` and legacy `definitions`;
- local `$ref`, self `$ref`, nested local pointer refs;
- local `$id` alias at root and nested schemas;
- `properties` and `patternProperties` pointer escaping with `~` and `/`.

## Lowering golden tests

- primitive type schemas;
- const and enum values across all JSON value families;
- bounded arrays with prefixItems and items;
- closed objects with required and optional properties;
- open objects with additionalProperties false/true/schema;
- patternProperties overlap examples;
- allOf merged object examples;
- anyOf object variants;
- oneOf overlapping and disjoint examples;
- string patterns and formats;
- numeric bounds.

## Semantic oracle tests

For a supported finite schema subset, generate JSON values and compare:

```text
value_satisfies_schema(schema, value)
```

against grammar acceptance of serialized candidate values.  This is the most
important test class for publication because it catches denotational errors that
compilation alone cannot see.

## Negative tests

- rejected unsupported keyword names;
- unresolved refs;
- remote refs;
- invalid regex patterns;
- numeric constraints with invalid types;
- schema expansion over configured limits once limits exist.
