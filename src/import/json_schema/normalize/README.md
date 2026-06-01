# JSON Schema normalization phase

Normalization is the value-level schema algebra layer.  It is not a generic
optimizer: every rewrite must have one of the following documented contracts.

1. **Exact equality**: the accepted JSON values are unchanged.
2. **Safe over-approximation**: the output language is a superset, and the loss
   of precision is explicitly named.
3. **Rejected unsupported shape**: the importer refuses a schema rather than
   silently changing its denotation.

The current implementation preserves the large existing combinator code in
`combinators.rs` but gives it a dedicated namespace.  Further chunks should
split that file into `all_of.rs`, `any_of.rs`, `shape.rs`, `merge.rs`, and
`factor.rs` once compiler-repair passes begin.
