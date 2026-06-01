# Algorithm walkthrough

This walkthrough follows one compile through the scan-relation subsystem.

## Step 1: choose the source vocabulary

The public `Vocab` maps original token ids to bytes.  The scan-relation
subsystem first builds an ordered byte vocabulary.  Ordering by bytes lets the
prefix tree group common byte prefixes and makes interval representations
possible.

File: `ordered_vocab.rs`.

Output:

```text
OrderedVocab {
  original_slot_count,
  ordered_to_originals,
  ordered_token_bytes,
}
```

The important invariant is that an ordered token id is not an original token id.
Every conversion must go through `ordered_to_originals` or the final
`ScanRelationVocabMap`.

## Step 2: optionally pre-quotient by CanMatch signatures

For large vocabularies, many tokens behave identically for CanMatch.  The
subsystem may compute a CanMatch-specific equivalence map first.

File: `vocab_equivalence.rs`.

This is not Terminal-DWA equivalence.  It is a separate quotient over all lexer
states and future completion terminals.

Output:

```text
ManyToOneIdMap {
  original_to_internal,
  internal_to_originals,
  representative_original_ids,
}
```

Then the compile builds a compact token-byte map from representatives and runs
scan-relation construction on that smaller vocabulary.

## Step 3: build a trie over token bytes

The ordered vocabulary becomes a prefix tree.  Each trie edge is a byte segment.
Each leaf corresponds to one ordered token, which may represent many original
ids if duplicate byte strings exist.

File: `ordered_vocab.rs`.

The trie is the domain over which both sparse terminal-sequence collection and
grouped interval collection walk.

## Step 4: collect interval CanMatch maps

For every lexer state, the collector determines which terminal completions are
reachable for which ranges of ordered token ids.  The grouped representation is:

```text
TerminalRangeGroup {
  terminals: [t₁, t₂, ...],
  ranges: [(lo₁, hi₁), (lo₂, hi₂), ...],
}
```

A group denotes a Cartesian product:

```text
terminals × token_ranges
```

File: `collector.rs`.

This representation is compact when many terminals share the same token ranges.

## Step 5: choose dense or sparse root path

For tiny state/terminal sets, a sparse root collection path can be faster and
simpler.

File: `root_collect.rs`.

The decision is controlled by thresholds and remains compile-time only.

## Step 6: materialize scan-relation vocabulary classes

The interval maps are not yet runtime weights.  They tell us that certain
terminal groups are active over ranges of ordered token ids for certain tokenizer
state classes.

The materializer converts interval starts/ends into sweep events.  As the sweep
passes each ordered token position, it maintains an active group set.  That set
implies the token's signature.

File: `vocab_materialize.rs`.

Core signature:

```text
signature(position) = sorted set of (state_class, terminal) labels active at position
```

Tokens with identical signatures share the same scan-relation internal token id.

## Step 7: build runtime weights

After signatures are assigned, the materializer inverts the relation into the
runtime-friendly layout:

```text
terminal -> weight over tokenizer-state class and token-set
```

This becomes `RuntimeCanMatchByTerminal`.

## Step 8: map into shared ID space

The compile pipeline later reconciles the scan-relation artifact with Terminal
DWA and Parser DWA ID spaces.  Chunk 06 does not change that.  It makes the
local scan-relation ID map explicit before reconciliation.

File: `compute.rs` returns `MappedArtifact<RuntimeCanMatchByTerminal>`.

## Step 9: runtime use

Runtime masking reads the final `can_match` weights from the compiled
`Constraint`.  Runtime commit scans chosen token bytes through
`src/scan/execution.rs`; it does not rebuild CanMatch tables.
