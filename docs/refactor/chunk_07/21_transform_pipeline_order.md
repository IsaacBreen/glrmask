# Transform pipeline order

The grammar transforms are not arbitrary. Their order affects both performance and sometimes which later transforms become applicable.

## Current conceptual order

The import pipeline roughly does:

```text
source parse
  -> NamedGrammar
  -> factoring
  -> simplification
  -> exact subtraction lowering
  -> terminal choice promotion
  -> flat lowering
```

JSON Schema may have additional frontend-specific simplifications before or between these steps.

## Why factoring before simplification can make sense

Factoring may introduce helper structure that simplification can clean. Conversely, simplification can expose common structure for factoring. This suggests that in the long term the pipeline may need a small fixed-point between safe simplification and factoring.

## Why exact subtraction should happen before flat lowering

Exact subtraction is easier to reason about at the named grammar level because alternatives still have source structure. Once lowered to flat productions, recovering exact source-level alternatives is harder.

## Why terminal choice promotion should be late

Promotion changes whether alternatives are represented as terminal-level recognizers or parser-level choices. It should happen after source-level exact-subtraction decisions, because promoting too early can hide alternatives from exact subtraction.

## Future typed pipeline

A later chunk can define:

```rust
struct GrammarTransformPipeline { ... }
```

with named phases, profiling, and explicit options. That should not be done in Chunk 07 because it would entangle this structural split with compile behavior changes.
