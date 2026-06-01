# Vocabulary partitioning theory and policy

## Why vocabulary partitioning exists

The Terminal DWA can be large because token bytes can complete terminal sequences in many different ways from many lexer states.  Building one monolithic automaton over all caller tokens is simple to state but expensive in practice.  Partitioning the vocabulary reduces intermediate products and permits parallel local construction.

The important point is that vocabulary partitioning is not a semantic restriction.  It is safe only because the final merge unions the local relations.

## Char-type partitioning

Char-type partitioning is the older stable strategy.  Tokens are grouped by coarse byte/character shape:

- non-alnum structural;
- mixed;
- ASCII alpha;
- digits;
- Unicode alpha;
- short auxiliary non-alnum;
- long auxiliary non-alnum.

This is encoded by `classify_vocab_char_type` and cached by `CharTypeSubVocabs`.

## Pair-partition cost partitioning

Pair-partition cost partitioning estimates which tokens are likely to drive expensive multi-step terminal construction.  It then splits tokens to minimize an objective.  The cost function and objective are read by `options.rs`, not by the builder.

The cost function answers: how expensive does this token look under the pair-partition construction?  The objective answers: how do we combine costs across partitions?

## Auto pair-partition selection

The auto strategy computes both char-type and pair-cost evidence and chooses pair-cost only if it passes safety guards.  The guards prevent overfitting to a cost estimate that produces suspiciously imbalanced partitions or too few/many estimated pair terminals.

The safety quantities are:

- second-largest partition size;
- maximum estimated pair-partition terminal count;
- minimum estimated pair-partition terminal count;
- minimum grammar-terminal count before the auto strategy is considered.

These are construction-quality safeguards.  They are not part of the final DWA semantics.

## Review rule

Any future partitioning strategy must satisfy this type-level interface:

```rust
fn choose(...) -> Arc<[Vocab]>
```

It should not return DWA states, id maps, parser data, runtime masks, or scan-relation entries.  Keeping the return type narrow protects the construction layers.
