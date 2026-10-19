# Canonical forced-token lookahead

`ConstraintState.forced_tokens()` is a read-only query for a safe prefix of
**library-canonical forced output**. Rust returns `Result<Vec<u32>>`; Python
returns `list[int]` and raises an exception for a typed recognition/resource
failure.

There is no public `forced()`, `forced_token_ids()`, or `forced_bytes()` method.
The historical canonical implementation remains an implementation detail behind
the single public `forced_tokens()` surface.

## What is forced?

The query first proves ordinary bytes that every live parser/lexer alternative
must consume. It encodes that text greedily with the longest currently admitted,
nonempty model-token spelling; the lowest original ID breaks equal-spelling
ties. This is GLRMask's stable canonicalization policy, not the model tokenizer's
BPE algorithm and not preservation of the model's original distribution.

For a grammar requiring `ab`, a vocabulary containing `a`, `b`, and `ab`
therefore returns the ID for `ab`. Duplicate admitted `ab` IDs are resolved to
the lowest original ID. A vocabulary spelling that is not actually admitted at
the current state does not participate in the choice.

The source state does not advance. The caller commits the returned IDs when
using the result:

```python
ids = state.forced_tokens()
for token_id in ids:
    state.commit_token(token_id)
```

An inference integration must also feed those IDs through its model to update
the model's decoding state. An empty list does not mean that the grammar is
complete or rejected; inspect acceptance and use ordinary masking as needed.

## Acceptance and exact-event boundaries

The query stops before a boundary where the root can already accept. Exact-token
terminals and special LLM tokens are original-ID events, not their decoded byte
spellings. A spelling-less event can be returned, and an empty byte spelling is
not silently assumed to be such an event. The compiled terminal-to-ID mapping is
authoritative.

An event opportunity also prevents canonical text from crossing its boundary.
For example, consider these two grammar paths with vocabulary tokens `A`, `AB`,
`B`, and exact event `MARK`:

```text
AB | A MARK B
```

Both paths project to ordinary bytes `AB`, but returning token `AB` would cross
the valid event boundary after `A`. Returning `A` would abandon the greedy
canonical tokenization of the proved text. The safe canonical result is therefore
empty. Without the event path, the result is token `AB`.

At an exact-event frontier, the query advances only a unique admitted original
ID. If the same ID has both byte and event interpretations, its complete commit
keeps all valid successors correlated. Multiple event IDs remain ambiguous.

## This is one canonical sequence, not an exhaustive token tree

`forced_tokens()` returns at most one canonical prefix. It does not enumerate all
original-ID histories that realize the same bytes. In particular, a result using
`AB` does not prove that the alternate history `A, B` is impossible. Applications
that need exhaustive guaranteed-token trees across aliases, alternate segmentations,
end IDs, special events, and boundary-crossing tokens need a different API and
must not treat this canonical prefix as tree completion.

## Full-mask reference implementation and bounds

The current correctness-first implementation may construct ordinary full token
masks. It proves bytes with bounded private recognition cursors, then commits each
selected original ID through the complete token/event successor before continuing.
Mask-generation performance is not part of this initial reference contract.

One call returns at most 10,000 IDs and proves at most 10,000 output bytes using
at most 100,000 byte/event probes. Reaching a query bound returns the proved
prefix. A typed library resource failure is returned as an error rather than
fabricated ambiguity, completion, or rejection. The original state remains
unchanged.

## Raw-byte recognition is separate

`commit_bytes(data)` incrementally recognizes raw text. It does not deliver
exact-ID events and may reach a state that no model-token sequence can reach.
When actual model token IDs are known, use `commit_token` to preserve their
semantics.

## Validation contract

Tests compare the public query with an independent finite byte/event trace model,
replay every returned ID through authoritative masks and commits, and verify
source-state preservation. Coverage includes duplicate IDs, greedy prefix
choices, root acceptance, UTF-8 fragments, exact events before and after text,
empty and spelling-less event representations, multiple event IDs, composed
children, all optimization modes, and saved/loaded artifacts.
