# Runtime token-space contract

## Coordinate systems

The artifact has four coordinate systems:

1. original vocabulary token id;
2. final runtime-internal token id;
3. original tokenizer state id;
4. final runtime-internal tokenizer-state id.

The old name `mask_game` was already removed in earlier chunks.  This chunk
makes the coordinate boundary live next to the artifact itself.

## Original token id

An original token id is the id used by the model and by the user's tokenizer. It
is the coordinate system of the public mask returned by `ConstraintState::mask`
and `fill_mask`.

## Internal token id

An internal token id is a quotient class after the compile pipeline has
reconciled Parser-DWA and CanMatch token classes.  It is not public model state.

## Token quotient maps

The two token quotient maps are:

```text
original_token_to_internal : OriginalTokenId -> InternalTokenId or u32::MAX
internal_token_to_tokens   : InternalTokenId -> Vec<OriginalTokenId>
```

They must be inverses up to excluded original tokens.

## State quotient maps

The two tokenizer-state quotient maps are:

```text
state_to_internal_tsid : OriginalTokenizerStateId -> InternalTokenizerStateId
internal_tsid_to_states : InternalTokenizerStateId -> Vec<OriginalTokenizerStateId>
```

They allow weights to use a final shared tokenizer-state coordinate system.

## CanMatch query

`can_match_for_state` expands internal token sets back to original token ids for
diagnostics.  `can_match_for_state_internal` preserves the final internal token
coordinate system for mask and commit internals.

## Rule for new code

New code must state which coordinate system it receives and which coordinate
system it returns.  A variable named simply `token` or `state` is acceptable only
in a local context where the coordinate system is obvious from the surrounding
function name.
