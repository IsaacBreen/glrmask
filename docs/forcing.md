# Strict token forcing

`ConstraintState.forced_tokens()` is a read-only query for a prefix of
**individually inevitable original model-token IDs**. Rust returns
`Result<Vec<u32>>`; Python returns `list[int]` and raises an exception if a
recognition proof fails for reasons other than grammatical rejection.

The query never selects a canonical tokenization. There is no public
`forced()`, `forced_token_ids()`, or `forced_bytes()` method.

## What is forced?

At each prospective token boundary the query checks two things:

1. Can the root already finish? If so, stop rather than remove that choice.
2. Is exactly one original token ID admissible? If so, advance a private
   recognition cursor with that complete ID and continue. Otherwise, stop.

For a grammar requiring `ab`, a vocabulary containing tokens for `a`, `b`, and
`ab` gives an empty initial forcing result: both `a` and `ab` can start the
output. Choosing the longest token would be a tokenization policy, not forcing.
After committing `a`, the query can return the ID for `b` if it is the only
admissible next ID. Two different IDs with identical byte spellings remain
two choices.

The source state does not advance. The caller commits the returned IDs when
using the result:

```python
ids = state.forced_tokens()
for token_id in ids:
    state.commit_token(token_id)
```

An inference integration must also feed those IDs through its model to update
the model's decoding state. Skipping token sampling does not remove that work.
An empty list does not mean that the grammar is complete or rejected; inspect
acceptance and use ordinary masking as appropriate.

## Exact events and ambiguous successors

An exact-token event is one original ID, not its printed spelling. A
spelling-less event can be forced. Empty bytes are not such an event, and
byte-identical IDs are not interchangeable.

One allowed ID can have several valid parser/lexer successors, including
both its byte interpretation and an exact-event interpretation. All remain
live after the private token commit. Dynamic grammar alternatives are queried
together: a repeated ID is counted once across their union, not once per owner.
Child acceptance is a call/return possibility, not necessarily root completion.
Root end-token controls retain the same policy as ordinary masking.

The query proves **local next-ID admission**, not the existence of a complete
future tokenization. A sparse model vocabulary can have a locally admissible
prefix without a vocabulary-realizable finish. Forcing does not introduce a
global completion search or choose another tokenization to work around that.

## Bounded proof without a full token mask

The implementation uses a private cardinality query rather than constructing
every vocabulary bit and counting afterward. It can stop as soon as two
distinct, finally admissible original IDs are proved. It may read an existing
exact mask or already-prepared vocabulary data; it does not generate a full
mask merely to establish uniqueness.

Candidate lexer matches alone are insufficient. Maximal munch, parser
continuations, nullable boundaries, exact events and root policy must all be
accounted for before an ID is a positive witness. When two alternatives admit
the same ID, that is still one witness. A class of grammar-equivalent tokens
can contain multiple original IDs and must not be counted as one token.

Proving that only one ID exists can require more work than finding two. The
query is not guaranteed to be faster than ordinary masking for every state.
Normal `mask()` and `fill_mask()` do not run the bounded-cardinality logic.

## Limits and errors

One call returns at most 10,000 IDs. Reaching that output limit returns the
proved prefix, which the caller can commit before asking again. This is not a
grammar length budget or a total wall-clock limit.

Proof/resource errors are errors, not fabricated empty masks, ambiguity, or
grammar completion. The original state remains unchanged. A real rejection can
remove an alternative; a resource failure cannot be used to prove that a
competing alternative is impossible.

## Raw-byte recognition is a separate operation

`commit_bytes(data)` remains available for incrementally recognizing raw text
or testing grammar-covered input. It does not deliver exact-ID events and may
reach a state that cannot be reached using the model vocabulary. When the
application knows the actual model token IDs, it should use `commit_token` to
preserve their semantics. No corresponding public byte-forcing operation is
required for raw-byte input to be useful.

## Validation contract

Bounded-query tests compare cold and warm results against complete ordinary
masks at multiple thresholds and across optimization modes. Forcing is checked
against the full-mask singleton reference and replayed with ordinary token
commits. Coverage includes duplicate spellings, event-only IDs, empty event
sets, root controls, partial UTF-8 sequences, child scopes, dynamic unions and
saved/loaded artifacts. The earlier canonical algorithm is retained only as an
internal test/reference facility, not a public serving policy.

## Measured qualification

See the [15 October 2026 core-batch report](validation/core-batch-2026-10-15.md)
for matched measurements, exact correctness gates and remaining trade-offs.
