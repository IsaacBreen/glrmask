# Function-level ledger

This ledger describes the major functions after Chunk 06.  It is not a full API
reference; it is a map for future cleanup.

## Ordered vocabulary functions

### `build_internal_token_bytes_from_groups`

Given a final or intermediate grouping of original token ids, choose one byte
string representative for each internal token id.  This is used after CanMatch
vocabulary quotienting so the next scan-relation pass can run over a compact
byte vocabulary.

Important: the representative byte string must be one of the grouped original
bytes.  It is safe only because the group was formed by a relation that makes all
members equivalent for the subsequent computation.

### `build_ordered_vocab`

Sorts token bytes lexicographically and groups duplicate byte strings.

Important: do not use original token ids as ordered token ids.  The ordered id is
just position in the byte-sorted list.

### `build_ordered_vocab_prefix_tree`

Builds a `VocabPrefixTree` from ordered byte strings.

Important: the tree assumes input is already sorted/presorted.  Do not feed it an
unsorted list unless the builder explicitly supports that.

### `get_ordered_vocab_trie_artifacts`

Returns ordered vocab and trie artifacts for a token-byte map, using a small
process-global cache when enabled.

Important: cache hits are verified against the source map, not accepted by hash
alone.

### `get_ordered_vocab_trie_artifacts_for_vocab`

Uses the `Vocab`-attached derived artifact cache where possible.  This is the
fast path for repeated compiles with the same vocabulary.

## Vocabulary equivalence functions

### `scan_relation_vocab_equiv_enabled`

Reads the historical env var controlling pre-quotienting.  This should later move
into typed compile options.

### `compute_scan_relation_vocab_equivalence_map`

Naive or direct CanMatch quotient builder over all tokenizer states and trie
paths.  Useful as a reference path.

### `compute_scan_relation_vocab_equivalence_map_fast`

Fast quotient builder using the existing vocab-equivalence machinery but with a
CanMatch-specific tokenizer view.  It deliberately leaves future-group metadata
empty because this quotient is not Terminal-DWA equivalence.

## Materialization functions

### `build_sweep_events`

Turns grouped interval maps into add/remove events over ordered token positions.

### `apply_sweep_events`

Mutates the active group set at one sweep position.

### `build_signature_from_active_group_ids`

Computes the sorted `(state_class, terminal)` label signature induced by active
groups.  This signature is the token class key.

### `build_scan_relation_vocab_and_weights_from_interval_maps`

Main materializer.  It builds the vocabulary quotient and runtime weights.

## Legacy functions

### `build_legacy_scan_relation_vocab_and_weights_from_interval_maps`

Expanded baseline implementation.  It is slower and more verbose but easier to
validate against.  It should not be the code path explained in the paper.

### `validate_group_scan_relation_vocab_outputs`

Compares grouped materialization against the legacy materializer.  Keep this as
an oracle until tests are strong enough to remove it.

## Root collection functions

### `root_terminal_union_count`

Computes how many terminal ids matter at root states.  Used to decide whether
sparse root collection is viable.

### `collect_sparse_root_can_match`

Builds state classes using the sparse `CanMatchComputer`.  This is a compile-time
optimization path only.

## Compute functions

### `compute_scan_relation`

Entry point for explicit token-byte maps.

### `compute_scan_relation_for_vocab`

Main entry point for a `Vocab`.  Handles optional CanMatch pre-quotienting.

### `prepare_vocab_for_scan_relation`

Warms derived vocab/trie artifacts.

## Runtime scan function

### `scan::execution::execute_tokenizer_from_state`

Scans one chosen byte slice from one tokenizer state.  This is runtime execution,
not global relation construction.
