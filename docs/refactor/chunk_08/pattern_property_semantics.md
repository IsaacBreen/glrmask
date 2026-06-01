# Pattern-property semantics

For each object key `k`, JSON Schema applies every pattern-property schema whose
pattern matches `k`.  If a literal property schema also exists for `k`, the value
must satisfy both the literal property schema and every matching pattern schema.

Mathematically:

```text
value_schema(k) = intersection(
  literal_property_schema(k), if present,
  all pattern_property_schema(p) where regex p matches k,
  additionalProperties schema if k is not covered and additionalProperties is schema-valued
)
```

Lowering must be careful not to treat pattern properties as alternatives.  They
are cumulative constraints.  Any factoring of fixed keys must preserve that
intersection.
