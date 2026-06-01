# Runtime cache taxonomy

## Cache classes

The runtime cache fields fall into several classes.

### 1. Token-to-output materialization caches

These turn final internal token ids into original-vocabulary bitmasks.

Fields include:

```text
internal_token_buf_masks
internal_token_buf_flat
internal_token_buf_offsets
all_tokens_buf_mask
word_group_sparse_masks
word_group_prefix_buf_masks
quad_group_sparse_masks
byte_group_sparse_masks
```

Mathematical role: implement the map

```text
S_internal -> ⋃_{i ∈ S_internal} originals(i)
```

as a packed `u32` mask over original token ids.

### 2. Dense internal-token masks

These are masks over final internal token ids.

Fields include:

```text
internal_token_dense_words
weight_token_dense_masks
seed_terminal_dense
seed_state_dense
seed_universe_dense
```

Mathematical role: keep Parser-DWA and CanMatch computations in the compact
internal coordinate system as long as possible.

### 3. Dense output-mask caches

These are output-vocabulary masks over original token ids.

Fields include:

```text
weight_token_buf_masks
seed_state_buf_masks
heavy_token_dense_masks
pair_word_group_buf_masks
quad_word_group_buf_masks
super_word_group_buf_masks
mega_word_group_buf_masks
giga_word_group_buf_masks
```

Mathematical role: accelerate materialization of common internal-token sets.

### 4. Sparse output-mask caches

These are sparse lists of `(word_index, mask)` entries.

Fields include:

```text
weight_token_sparse_buf_masks
word_group_sparse_masks
quad_group_sparse_masks
byte_group_sparse_masks
```

Mathematical role: avoid scanning full output masks when the selected original
token set is small.

### 5. Transition lookup caches

These convert map-based automata representation into runtime-friendly arrays or
hash maps.

Fields include:

```text
dwa_fast_transitions
tokenizer_fast_transitions
```

Mathematical role: no semantic change.  They implement the same transition
function with a different representation.

### 6. Cost-model caches

These help choose between dense, sparse, group, and complement materialization.

Fields include:

```text
total_internal_buf_cost
heavy_total_cost
light_avg_cost_x256
internal_token_buf_op_costs
word_group_buf_op_costs
word_group_sparse_total_entries
word_group_sparse_max_entries
```

Mathematical role: runtime strategy selection only.

### 7. Special lexical escape cache

```text
json_escape_prefix_buf_mask
json_u_prefix_token_id
```

These support JSON string escape behavior.  `json_u_prefix_token_id` is semantic
metadata derived from token bytes.  `json_escape_prefix_buf_mask` is a cache over
original token ids.

## Serialization rule

Only fields required to reconstruct the language should be serialized.  Cache
fields must be skipped and rebuilt.

This rule avoids locking the serialized format to a particular mask
materialization strategy.
