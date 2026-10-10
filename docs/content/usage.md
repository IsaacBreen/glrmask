GLRMask has three core types:

- `Grammar`: the language you want to generate.
- `Constraint`: a grammar compiled against a specific vocabulary.
- `ConstraintState`: the mutable state for a single generated sequence.

To use it, compile a `Grammar` into a `Constraint`, then call `constraint.start()` to initiate a `ConstraintState`. Then in the decoding loop:

1. Compute the next-token mask.
2. Sample a token the mask allows.
3. Commit that token to the state.

```python pseudocode
state = constraint.start()
end_token_ids = configured_end_token_ids

while generating:
    in parallel:
        logits = llm.forward(...)
        mask = state.mask()

    logits = apply_mask(logits, mask)
    token_id = sample(logits)
    state.commit_token(token_id)
    if token_id in end_token_ids:
        break
```
