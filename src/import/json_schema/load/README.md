# JSON Schema loading phase

This directory owns the interpretation

```text
serde_json::Value -> schema::SchemaDocument
```

It should not import `crate::import::ast::GrammarExpr`, `grammar_ir`, `Lowerer`,
or any automata/runtime type.  When a future pass further splits `mod.rs`, use
these boundaries:

- `pointer.rs`: JSON Pointer escaping, local-id aliases, fragment normalization.
- `reference_scan.rs`: collection of `$defs`, `definitions`, local ids, and
  non-definition local targets.
- `keyword_reader.rs`: type-safe loading of individual JSON Schema keywords.
- `schema_reader.rs`: recursive construction of `schema::Schema`.

The present chunk keeps loader code in one Rust file to minimize needless
cross-file private visibility churn, but the mathematical boundary is now
explicit and documented.
