# JSON Schema lowering phase

This directory owns the interpretation

```text
schema::SchemaDocument -> import::ast::NamedGrammar
```

The core `mod.rs` owns rule allocation, builtin JSON lexical terminals,
reference-rule allocation, literal emission, and generic type dispatch.  Domain
lowerers live beside it:

- `object.rs`: object languages, required-property permutations, open-object
  residual pairs, pattern-property factoring, and object-anyOf factoring.
- `array.rs`: homogeneous arrays, tuples, bounded arrays, and array terminal fast
  paths.
- `string.rs`: JSON string value constraints, regex lowering, formats, and UTF-8
  body encoding.
- `number.rs`: integer/number range and multipleOf lowering.

No code in this directory should parse raw `serde_json::Value` except when
emitting exact JSON literals from `const`/`enum` values that have already been
loaded.
