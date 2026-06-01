# Interaction with Mask

Mask computes which next vocabulary tokens are valid from a state. Commit consumes a chosen token or byte string and mutates the state. They must be consistent but separate.

The optional mask assertion at the Commit API boundary checks this relationship for token commits. It deliberately sits at the boundary:

```text
API token id -> mask membership snapshot -> commit -> assertion
```

The internal Commit transition should not call Mask while computing next state. Doing so would entangle two denotations and hide bugs.
