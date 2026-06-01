# Symbol visibility and prominence

This document ranks JSON Schema importer symbols by prominence.  The goal is to
make important concepts easy to find and temporary helpers visually subordinate.

## 1. Public within the crate

These symbols deserve prominent names and rustdoc because other importer stages
or tests rely on them:

```text
schema_to_named_grammar
schema::SchemaDocument
schema::Schema
schema::SchemaKind
schema::SchemaAssertions
schema::SchemaType
options::JsonSchemaConfig
load::load_document
lower::lower_document
lower::Lowerer
normalize::all_of_schema
normalize::try_merge_all_of_objects
```

## 2. Important subsystem entry points

These should stay near the top of their module:

```text
Lowerer::lower_schema
Lowerer::lower_ref
Lowerer::lower_assertions
Lowerer::lower_object
Lowerer::lower_array
Lowerer::lower_string
Lowerer::lower_number
Lowerer::lower_any_of
Lowerer::lower_all_of
Lowerer::lower_one_of
```

## 3. Helper symbols that should not dominate file openings

These should be moved downward or into helper modules when files are split:

```text
factored_small_string_enum_expr
large_string_enum_regex_literals
string_enum_regex
collect_shared_ap_exclusion_plan
is_regex_compile_limit_error
positive_integer_multiple_i64
decimal_fraction_regex
schema_slices_shape_equivalent
option_numbers_shape_equivalent
```

## 4. Naming policy

Prefer names that reveal denotation:

- `schema` means value-level schema node.
- `grammar` means grammar IR object.
- `json_text` or `encoded` means serialized bytes.
- `decoded` means decoded JSON string contents.
- `lower` means schema-to-grammar translation.
- `normalize` means schema-to-schema algebra.
- `resolve` means reference graph interpretation.

Avoid names that encode experiments or implementation accidents:

- `simple importer`,
- `mask game`,
- `pm`,
- `l1`,
- `hack`,
- `snowplow` outside targeted benchmark-specific comments.

The Snowplow names currently remain because they document observed schema shapes,
but publication code should eventually generalize them into structural names such
as `large_pattern_key_prefix_trie`.
