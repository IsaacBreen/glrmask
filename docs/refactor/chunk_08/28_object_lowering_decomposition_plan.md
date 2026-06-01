# Object lowering decomposition plan

`lower/object/mod.rs` remains the largest JSON Schema source file after Chunk
08.  The file is now isolated under the object-lowering namespace, but it should
be split in the next implementation pass.  This document defines the exact
future decomposition and the mathematical reason for each file.

## 1. Why object lowering is intrinsically large

A JSON object schema combines two different mathematical structures:

1. a set constraint over property names, and
2. a sequence constraint over the concrete serialized order of key-value pairs.

A schema such as

```json
{
  "type": "object",
  "properties": {"a": A, "b": B},
  "required": ["a"],
  "additionalProperties": C
}
```

denotes unordered JSON values, but generation sees ordered text:

```text
{"a": valueA, "b": valueB, "x": valueC}
{"x": valueC, "a": valueA}
{"b": valueB, "a": valueA, "x": valueC}
```

The lowerer therefore constructs finite languages over permutations, loops over
additional keys, and residual languages for keys not already consumed.

## 2. Target file split

### `lower/object/mod.rs`

Only module declarations and the public object entry methods:

```text
lower_object
lower_object_requiring_any_property
lower_object_with_exclusive_properties
try_lower_closed_object_any_of_variants
try_lower_open_object_any_of_variants
try_lower_ref_string_path_object_any_of
```

### `lower/object/types.rs`

State records and lightweight descriptors:

```text
ObjectItem
AnyOfFixedObjectItem
AnyOfFixedObjectVariant
AnyOfObjectVariant
AnyOfFixedObjectState
AnyOfObjectState
ShadowOwnerState
AnyOfObjectPhase
```

### `lower/object/fixed.rs`

Closed objects and fixed-property bodies:

```text
lower_fixed_object_body_exprnfa
lower_fixed_object_body_exprnfa_without_group
lower_large_closed_object_prefix_chain
lower_large_closed_object_fixed_pair_loop
```

### `lower/object/open.rs`

Open objects and additional-property languages:

```text
dynamic_pair_list_body
lower_required_prefix_open_object_pair_loop
lower_additional_key_colon paths through string lowerer
```

### `lower/object/patterns.rs`

Pattern-property overlap and pattern maps:

```text
try_lower_pattern_map_pair_list_object
lower_snowplow_large_pattern_object_key_trie
pattern_schema_for_property
property_matches_pattern
```

### `lower/object/any_of.rs`

Variant factoring:

```text
collect_closed_any_of_object_variant
collect_open_any_of_object_variant
lower_closed_any_of_object_variants_expr_nfa
lower_open_any_of_object_variants_expr_nfa
```

### `lower/object/shadow.rs`

Duplicate-key and residual-branch suppression:

```text
select_shadow_owner_for_variant
shadow_owner_suppresses_close
shadow_owner_can_take_additional
advance_shadow_owner_on_key
invalid_residual_value_for_owner
```

### `lower/object/properties.rs`

Property item lowering:

```text
lower_property_item
lower_object_property_value_schema
object_with_required_synthetic_properties
single_numeric_property_type
has_non_numeric_assertions
```

## 3. Visibility rule

When splitting, prefer `pub(super)` over `pub(crate)`.  If a method must be used
from `normalize`, it should stay on `Lowerer` with `pub(crate)`.  Pure helper
functions should remain private to the object subtree.

## 4. Exactness rule

Every object helper should state whether it builds:

1. exact closed object language,
2. exact open object language,
3. broad open object fallback,
4. exact factored union,
5. broad factored union.

The object file currently mixes these categories.  The split should make the
category visible from the file name and comments.
