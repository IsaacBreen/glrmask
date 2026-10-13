# Forcing and deterministic text

GLRMask has two read-only continuation queries. Both return original model-token
IDs, leave the source state unchanged, and stop at an accepting prefix. An
accepting grammar may still allow a continuation: stopping is a competing choice.

## Canonical byte fast-forwarding: `forced()`

`state.forced()` proves a common ordinary-byte prefix and selects a stable,
grammar-admitted encoding of that prefix. It chooses the longest admissible token
spelling, breaking identical-spelling ties by the lowest original token ID.
An admissible longer token that extends beyond the proved prefix blocks an
unstable shorter choice. A longer spelling that the grammar rejects does not.

For a grammar accepting only `ab` and vocabulary entries `a`, `b`, and `ab`,
this query can return the ID for `ab`, even though `[a, b]` is another legal
encoding. This is a **library canonicalization policy**, not the model tokenizer's
BPE algorithm and not a proof that the chosen token IDs are inevitable. Selecting
one encoding rather than another can change the model's later logits and its
distribution over token sequences. The serving integration must opt into that
trade-off when it consumes the result.

## Inevitable original IDs: `forced_token_ids()`

`state.forced_token_ids()` emits a token only when the authoritative next-token
mask contains exactly one original ID. It commits that ID to a private successor
and repeats. It does not choose between byte aliases or alternative segmentations.
For the `ab` example above it returns an empty list. With only `a` and `b` in the
vocabulary, it can return their two IDs.

This is a local admission guarantee, not a global completion search. The query
does not eliminate a locally admitted branch merely because a sparse vocabulary
might make eventual completion impossible. The result is a safe prefix and need
not be maximal when the library's finite query limits are reached.

## Exact-token events and grammar boundaries

An exact-token match is an atomic ID event, not a decoded byte string. A spelling-
less event is not epsilon. Equal decoded bytes do not make different exact IDs
interchangeable. Both queries retain an ID's ordinary byte interpretation and its
exact-event interpretation whenever both are valid.

The canonical byte proof stops when an exact event competes with text or another
event. A unique admitted ID may cross that boundary, but only its **full ID
commit** determines what can follow. No byte-only cursor is used to discard an
exact-event branch. Reserved root end IDs are not ordinary spelling candidates.
Every returned token is revalidated and committed through the ordinary public
recognition semantics before any later returned token is proved.

Dynamic union alternatives are analyzed together, with each grammar owner's
lexer/parser state kept correlated. Independently computed forced vectors need
not be equal for the union to have a common prefix. Incompatible ordinary spelling
maps or root policies cause canonical forcing to decline rather than invent an
alias policy.

## Limits and failures

Forcing is optional and conservative. Acceptance, ambiguity, an unstable encoding,
an unavailable proof, or a library resource limit can produce an empty or shorter
result. There is no implicit retry or background continuation. Calling the query
again on the unchanged state does not advance it.

The implementation limits output IDs, output bytes for canonical forcing, and
checked byte/event probes. These counters are **not** a total CPU, wall-clock, or
memory budget; a parser closure or ordinary mask operation can itself be expensive.
A proven grammatical rejection can remove a branch. A metadata/proof/resource
failure cannot be used as evidence that a competing branch is impossible.

Normal mask and commit behavior does not depend on whether either forcing query
is used. Runtime state still belongs to the caller: commit the returned IDs only
when the serving policy has chosen to use that kind of forcing.


## Regression contract

The test suite checks both queries against an independent finite-language model
whose symbols are bytes or exact model-token events. An ID may consume its full
byte spelling or a matching atomic event; both successors survive when valid.
Empty and missing spellings are not epsilon byte transitions. Reference masks
and acceptance are computed by this model, not by another GLRMask backend.

The same cases run across the four optimization choices, source-bound and
prepared-bound components, and saved/loaded constraints. Additional cases cover
UTF-8 scalars split across model tokens, distinct aliases, root end controls,
child ignore competing with parent continuations, and unused high token IDs.
A conservative query result may be shorter than the reference result, but may
not contain a different unproved token. Every returned token is replayed through
the ordinary mask and commit interface, and the original state remains unchanged.
