# Mask runtime

Mask evaluates the current `ConstraintState` frontier against the Parser DWA.
It is not responsible for modifying the state.

Files:

- `mod.rs`: Mask phase graph and public `ConstraintState::mask`/`fill_mask` methods.
- `dense_acc.rs`: dense internal-token accumulators carried along DWA walks.
- `bitset.rs`: packed original-vocabulary bit operations.
- `constants.rs`: numerical thresholds for fast paths.
- `queue.rs`: traversal queue policy.
- `profile.rs`: Mask diagnostic output and profile structures.

The core denotation is independent of all fast paths: every path computes a set
of internal token ids and then materializes those internal ids to original token
ids.
