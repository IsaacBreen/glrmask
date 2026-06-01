# Supported-subset matrix and publication claims

This matrix is deliberately conservative.  It describes how the current importer
should be presented until dedicated semantic tests prove a stronger claim.

| JSON Schema feature | Current posture | Exactness claim | Next action |
|---|---|---|---|
| boolean schema `true` | loaded as `Any` | exact over JSON values | keep |
| boolean schema `false` | loaded as `Never` | exact when lowered to empty grammar | add golden test |
| `type` string/array | typed load | exact for basic JSON families | document integer/number overlap |
| `const` | lowered as JSON literal | exact for JSON structural equality except textual number caveats | add duplicate-key note |
| `enum` | lowered as choice/literal or string regex optimization | exact for literals if serialization convention accepted | add large-enum snapshot |
| `$ref` local pointer | supported for local refs and aliases | intended exact for local graph subset | replace strings with NodeId |
| `$defs` / `definitions` | collected recursively | intended exact local target set | add nested ref tests |
| `properties` | typed object field constraints | exact if duplicate keys rejected or semantics specified | decide duplicate policy |
| `required` | object lowerer enforces presence | intended exact | add property-set oracle |
| `additionalProperties` bool/schema | lowered through complement/exclusion rules | exactness depends on key complement and pattern dialect | prove with string subtraction tests |
| `patternProperties` | pattern matching via Rust regex path | not exact ECMA-262 unless subset documented | document regex dialect |
| `minProperties` / `maxProperties` | object lowering attempts bounds | intended exact for no-duplicate object semantics | add cardinality tests |
| `items` schema | supported | exact for homogeneous tail | add unbounded-tail tests |
| tuple-form `items` | legacy support | draft-dependent | document draft posture |
| `prefixItems` | supported | intended exact | add draft-2020 tests |
| `additionalItems` | supported with prefix items | draft-dependent | document draft posture |
| `minItems` / `maxItems` | supported | intended exact | add bounded-array tests |
| `minLength` / `maxLength` | supported with policy shortcuts | not globally exact unless length metric documented | decide scalar-vs-byte semantics |
| `pattern` | Rust regex conversion | dialect-dependent | translate or document |
| `format` | recognized formats only | format semantics are annotation in JSON Schema by default | document as optional strengthening |
| `minimum` / `maximum` | stored as f64 | approximate for arbitrary JSON numbers | replace with decimal rational |
| `exclusiveMinimum` / `exclusiveMaximum` | bool or numeric supported | same numeric caveat | add exact tests |
| `multipleOf` | stored as f64 | approximate | redesign numeric model |
| `anyOf` | grammar choice plus factoring | exact for simple unions; some object factoring requires proof | label helper exactness |
| `allOf` | merge/factor/intersect when safe | exact only under side conditions | comment every path |
| `oneOf` | lowered as choice in current code | overapproximate when branches overlap | either check disjointness or document |
| `not` | mostly rejected/limited | no broad exactness claim | structured unsupported error |
| `propertyNames` | rejected | unsupported | keep diagnostic |
| `uniqueItems` | rejected | unsupported | future semantic array set reasoning |
| `contains` and min/maxContains | rejected | unsupported | future automaton with witness counting |
| dependencies/dependent* | rejected | unsupported | object dependency lowerer |
| unevaluated* | rejected | unsupported | requires annotation/evaluation tracking |

The publication text should not say "supports JSON Schema" without a qualifier.
It should say "supports a documented, grammar-lowerable subset of JSON Schema,"
then point to this matrix or its cleaned-up successor.
