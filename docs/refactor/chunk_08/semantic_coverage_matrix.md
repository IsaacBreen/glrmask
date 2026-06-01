# Semantic coverage matrix

This file classifies JSON Schema keyword families by current importer treatment.
It is intentionally conservative: if a feature is only partially supported, it
is not marked simply "supported".

| Family | Keywords | Current status | Contract | Notes |
|---|---|---:|---|---|
| Boolean schemas | `true`, `false` | Supported | Exact | Loaded as `Any` / `Never`. |
| Type | `type` | Supported | Exact for primitive families | Type unions lower as choices over supported primitive languages. |
| Const | `const` | Supported | Exact | Emits exact JSON literal. |
| Enum | `enum` | Supported | Exact for listed JSON values | Large string enums may use regex fast path when safe. |
| Object literal props | `properties`, `required` | Supported | Exact for many closed/open shapes | Key-order handling is complex and remains a hotspot. |
| Pattern props | `patternProperties` | Supported | Intended exact when regex lowering succeeds | Regex matching is value-level over property names. |
| Additional props | `additionalProperties` | Supported | Intended exact for allow/deny/schema shapes | Uses exact subtraction for fixed-key exclusions. |
| Property bounds | `minProperties`, `maxProperties` | Partially supported | Exact for many enumerated/factored shapes, otherwise documented broadening | Needs more explicit source comments at fallback sites. |
| Arrays | `items`, `prefixItems`, legacy tuple `items`, `additionalItems` | Supported | Intended exact for regular tuple/tail shapes | `additionalItems` only affects tuple contexts. |
| Array bounds | `minItems`, `maxItems` | Supported | Exact for regular shapes | Bounded fast paths are grammar-shape optimizations. |
| Strings | `minLength`, `maxLength`, `pattern`, `format` | Supported subset | Exact where regex conversion succeeds; unknown format annotation ignored | Format support is explicitly enumerated. |
| Numbers | `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf` | Supported subset | Exact-ish under current regex implementation, but f64 representation is publication risk | Needs exact decimal representation eventually. |
| Combinators | `anyOf`, `oneOf`, `allOf`, `not` | Partial | Mixture of exact/factored/broadened/rejected | `not` is shape-limited. `oneOf` is currently lowered like choice in many places, not full exclusive-one semantics. |
| References | `$ref`, `$defs`, `definitions`, local ids | Local refs supported | Exact for supported local graph | Remote refs are not supported. |
| Metadata | `$id`, `id`, `title`, `description`, `$schema`, etc. | Accepted/ignored where annotation-only | Exact because annotations do not affect validation | `$id` local alias affects ref resolution. |
| Conditionals | `if`, `then`, `else` | Broadly ignored / limited | Needs explicit policy | Publication should either reject or implement. |
| Unevaluated/dependent | `unevaluatedProperties`, `dependentSchemas`, etc. | Rejected | Rejection | Safer than silent broadening. |
| Unique/contains | `uniqueItems`, `contains`, `minContains`, `maxContains` | Rejected | Rejection | Not regular in general when uniqueness interacts with arbitrary values. |

## Required follow-up before publication

1. Audit every broadening fallback and add an in-code comment with the exact
   inclusion relation.
2. Decide whether `oneOf` should be mathematically exclusive or intentionally
   treated as `anyOf` with a clearly named option.
3. Replace numeric `f64` schema storage with exact decimals or loudly document
   the precision contract.
4. Decide whether conditionals are rejected or implemented; silent ignoring is
   publication-hostile unless proven annotation-like in the accepted subset.
