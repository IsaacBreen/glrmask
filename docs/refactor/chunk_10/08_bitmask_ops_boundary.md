# `runtime/bitmask_ops.rs` boundary

## Purpose

`bitmask_ops.rs` contains low-level Boolean operations over output-vocabulary
masks.  It is intentionally not a parser module, lexer module, grammar module,
or artifact module.

## Operations

The module currently owns:

```text
or_dense_buf
andnot_dense_buf
copy_dense_buf
or_sparse_buf_entries
andnot_sparse_buf_entries
count_complement_subgroups
```

## Mathematical role

These functions implement Boolean algebra over packed bitsets:

```text
OR:      A := A ∪ B
AND-NOT: A := A \ B
COPY:    A := B
```

where the universe is original vocabulary token ids grouped into `u32` words.

## Why extract it?

Before this chunk, cache rebuilding and dense-to-buffer materialization depended
on helper functions hidden inside `constraint.rs`.  That caused a false
ownership relation: `Constraint` appeared to own bitset algebra.

The new location makes the dependency correct:

```text
cache builder          -> bitmask_ops
mask materializer      -> bitmask_ops
bitmask_ops            -> no runtime artifact knowledge
```

## Safety note

The functions use unchecked indexing or unaligned reads for speed.  This chunk
moves them but does not audit their unsafe blocks.  Unsafe justification belongs
in the future error/invariant/safety chunk.
