# Mask/Commit equivalence oracle

The optional assertion in `runtime/commit/mask_assert.rs` encodes a crucial
sanity check:

```text
token t is in Mask(S)  <=>  Commit(S, bytes(t)) succeeds
```

This equivalence is subtle because Mask works over vocabulary tokens while
Commit works over bytes and terminal boundaries.  It is also affected by longest
match exclusions and partial tokenization states.

The oracle snapshots Mask membership before mutation, runs Commit, and asserts
that the boolean result agrees.  This is a debugging check, not the definition
of Commit.

Useful future tests:

1. Enable the assertion for all small grammar tests.
2. Generate every token in a small byte vocabulary and compare Mask membership
   to Commit success on cloned states.
3. Repeat from states ending in partial lexer matches.
4. Repeat from ambiguous parser frontiers.
5. Repeat with EOS enabled and disabled.

If the oracle fails, do not immediately assume Commit is wrong.  Possible causes
include incorrect final materialization, stale caches, bad token-space mapping,
or a mismatch in how EOS is handled.
