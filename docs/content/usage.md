GLRMask has four core types:

- `Grammar`: an immutable source or constructed grammar, with its bindings.
- `PreparedGrammar`: vocabulary-specific reusable machinery, which can still have unresolved bindings.
- `Constraint`: a complete, runnable grammar compiled against that vocabulary.
- `ConstraintState`: the mutable state for a single generated sequence.

For direct use, compile a `Grammar` into a `Constraint`. Use `prepare()` and `link()` when you need reusable components or late binding. Then call `constraint.start()` to initiate a `ConstraintState`. Then in the decoding loop:

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
