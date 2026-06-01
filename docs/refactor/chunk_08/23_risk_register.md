# Risk register

| Risk | Severity | Why it matters | Mitigation |
|---|---:|---|---|
| `oneOf` accepted as `anyOf` when branches overlap | High | Violates JSON Schema semantics | add disjointness check or reject overlapping cases |
| Rust regex dialect differs from JSON Schema ECMA-262 | High | Pattern schemas may accept/reject wrong strings | document subset or translate patterns |
| f64 numeric constraints | High | JSON numbers are arbitrary decimal text | use exact decimal/rational representation |
| duplicate object keys | High | Byte grammar may admit ambiguous objects | choose parser semantics or reject duplicates |
| object permutation explosion | Medium | Publication examples may time out | add expansion diagnostics and symbolic object automaton |
| environment knobs change language | Medium | Results become irreproducible | classify every option as shape-only or semantic |
| local `$id` alias handling incomplete | Medium | refs may silently fail or map wrong target | introduce resolver graph |
| unsupported annotations ignored silently | Low/Medium | users may think `format`/annotations enforced | support matrix and warnings |
| tests remain monolithic | Medium | hard to find coverage gaps | split tests by stage |
| broad imports after refactor | Low | compile warnings under deny(warnings) | cleanup during compile-repair phase |
