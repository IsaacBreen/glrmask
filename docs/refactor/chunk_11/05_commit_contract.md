# Commit contract after the split

Commit is the only runtime operation that mutates the frontier.  It has four
logical stages:

1. Map a token id to bytes, or receive raw bytes directly.
2. Execute tokenizer scanning from every active tokenizer state.
3. For each completed terminal boundary, advance the parser stacks.
4. Replace the frontier map and increment generation.

This chunk extracts helper concerns that were polluting `commit/mod.rs`:

- `token_lookup.rs` handles stage 1 for token commits.
- `parser_advance.rs` handles stage 3 dispatch between template DFA and GLR.
- `options.rs` owns environment flags used by Commit.
- `mask_assert.rs` owns the optional debug oracle relating Commit to Mask.

The Commit denotation should not mention environment variables.  It should not
mention debug assertions.  It should not know whether token bytes came from a
dense or sparse vocabulary lookup beyond the boundary helper.  This chunk does
not fully achieve that, but it moves the cleanest pieces out first.

The public contract is unchanged:

```rust
commit_token(token_id)
commit_token_timed_ns(token_id)
commit_token_profiled(token_id)
commit_token_per_advance(token_id)
commit_bytes(bytes)
commit_tokens(tokens)
```

Every successful or failed Commit increments `generation`, because after an
attempted transition any cached mask for the previous frontier is invalid.
