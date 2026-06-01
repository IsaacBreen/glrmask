# Future chunks suggested by this split

## Commit fast-path phase split

`runtime/commit/mod.rs` should be split into:

```text
commit/reference.rs
commit/initial_fast_path.rs
commit/full_width_fast_path.rs
commit/small_queue_fast_path.rs
commit/direct_linear.rs
commit/profiled.rs
commit/public_api.rs
```

Each file should state the relation it computes and the preconditions required
for the fast path.

## FinalMaskMapping split

`runtime/mask_mapping.rs` should become:

```text
runtime/mask_materialize/
  mod.rs
  types.rs
  build.rs
  dense.rs
  sparse.rs
  complement.rs
  groups.rs
  profile.rs
```

This is mathematically the quotient from internal token ids to original token
ids.  It deserves a name closer to the paper and a clearer separation from Mask
traversal.

## Profile naming migration

If publication uses `can` for exact predicates, diagnostic fields named
`may_advance_ns` should either be renamed with compatibility aliases or
explicitly documented as historical names.

## Runtime option object

The existing `RuntimeOptions` placeholder should eventually own runtime feature
switches instead of reading environment variables directly.

## Mask traversal proof tests

After compile repair, add tests showing that direct single-path Mask, queue Mask,
and profiled Mask produce identical masks on small grammars.
