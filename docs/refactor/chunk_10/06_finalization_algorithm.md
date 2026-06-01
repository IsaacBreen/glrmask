# Runtime finalization algorithm

## Entry point

The entry point is:

```rust
Constraint::rebuild_runtime_caches()
```

which delegates to `rebuild_runtime_caches_impl` in `artifact/caches.rs`.

## Phase decomposition

The algorithm currently performs these phases:

1. rebuild parser-table guarded-shift indices;
2. compute internal-token sparse output masks;
3. compute tokenizer fast transitions;
4. compute dense masks for weight token sets;
5. compute Parser-DWA fast transitions;
6. compute sparse block masks for 64-token, 4-token, and 8-token groups;
7. compute prefix masks and sparse-entry prefixes;
8. compute sliding dense group masks for 2, 4, 8, 16, and 32 word groups;
9. compute the universal output mask;
10. compute JSON escape prefix mask;
11. compute heavy-token dense masks;
12. flatten sparse token masks into one contiguous entry table;
13. compute per-token and per-group operation costs;
14. compute dense/sparse weight-to-output caches;
15. install fast transition tables;
16. build seed dense masks.

## Parallelism policy

The existing policy is preserved:

```text
if rayon::current_num_threads() == 1:
    run sequentially
else:
    use rayon::join / parallel iterators for independent cache groups
```

This chunk does not change performance behavior; it merely relocates the code.

## Correctness proof sketch

Every phase computes a deterministic function of semantic fields and previously
computed cache fields.

For example:

```text
internal_token_buf_masks = f(internal_token_to_tokens, original_token_to_internal)
word_group_sparse_masks = g(internal_token_buf_masks)
all_tokens_buf_mask     = h(word_group_sparse_masks)
```

No phase is allowed to mutate:

```text
parser_dwa
table
tokenizer
can_match
original_token_to_internal
internal_token_to_tokens
state_to_internal_tsid
internal_tsid_to_states
```

except `table.rebuild_guarded_shift_index`, which rebuilds a table-local cache.

## Profiling behavior

The profiling environment variables remain where they were semantically:

```text
GLRMASK_PROFILE_COMPILE
GLRMASK_PROFILE_COMPILE_SUMMARY
```

The output labels still say `runtime_finalize` and
`runtime_finalize_derived`.  This chunk does not rename them because it is not
the diagnostics/configuration chunk.

## Future split target

`artifact/caches.rs` is now the only large runtime artifact file.  It should
later be split into:

```text
cache_build/orchestrate.rs
cache_build/token_materialization.rs
cache_build/weight_masks.rs
cache_build/seed_masks.rs
cache_build/fast_transitions.rs
cache_build/cost_model.rs
```

That split is safe after Mask and Commit stop reaching into cache helpers.
