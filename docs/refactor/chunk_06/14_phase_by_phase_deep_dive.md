# Phase-by-phase deep dive

This document describes each scan-relation phase as a relation transformation.
It is intentionally redundant with other Chunk 06 documents because it is meant
for someone trying to recover the mathematics from the code after the refactor.

## Phase A: original vocabulary to ordered vocabulary

Input:

```text
V_original = { original_token_id ↦ byte_string }
```

Output:

```text
V_ordered = [ byte_string classes sorted lexicographically ]
ordered_to_originals : ordered_token_id ↦ non-empty set of original ids
```

Reason:

The trie and sweep algorithms need token ids ordered by byte string/prefix.  The
user-facing tokenizer ids can be sparse, large, and unrelated to bytes.  The
ordered vocabulary creates a local coordinate system in which a subtree of the
prefix trie usually corresponds to a contiguous token interval.

Correctness condition:

For every original token id `o`, if `o` appears in `ordered_to_originals[k]`, then
`V_original[o] == V_ordered[k]`.  Conversely, every original token id with bytes
in the input appears in exactly one `ordered_to_originals[k]`.

Failure mode:

If duplicate byte strings are collapsed without recording all original ids, final
mask materialization will drop valid tokenizer ids.  This is especially easy to
miss because many unit tests use tiny vocabularies without duplicate byte
strings.

## Phase B: ordered vocabulary to prefix trie

Input:

```text
V_ordered
```

Output:

```text
Trie whose leaves are ordered token ids
```

Reason:

CanMatch collection is a prefix problem.  If scanning a byte segment fails from a
lexer state, the entire subtree under that segment can be pruned.  If scanning a
segment reaches a state where certain terminals are matched, every token under
that child subtree inherits those terminal possibilities for that range.

Correctness condition:

Every ordered token id appears at exactly one token node.  Every root-to-token
path spells exactly that ordered token's byte string.

Failure mode:

If a trie node's reachable token ids are not represented as inclusive ranges in
ordered-token space, the grouped interval collector will produce wrong sweep
events.

## Phase C: optional CanMatch vocabulary quotient

Input:

```text
Tokenizer DFA × ordered vocabulary
```

Output:

```text
ManyToOneIdMap from original token ids to CanMatch-equivalence classes
```

Reason:

Large vocabularies contain many byte strings with identical future-completion
behavior.  Quotienting them before building the full scan relation can shrink the
rest of the work.  The quotient must be CanMatch-specific.

Correctness condition:

Tokens are merged only when the terminal masks reached while scanning their bytes
from every lexer state are identical.

Failure mode:

Using Terminal-DWA equivalence here can merge tokens that agree on completed
terminal sequences but disagree on partial future completions.

## Phase D: grouped interval collection

Input:

```text
Tokenizer DFA × prefix trie × lexer states
```

Output:

```text
state_classes : lexer_state -> state_class
class_maps    : state_class -> [TerminalRangeGroup]
```

Each `TerminalRangeGroup` means:

```text
for every terminal t in terminals
and every token id k in any inclusive range
CanMatch includes pair (t, k) for this state class
```

Reason:

The naive representation is a map from each terminal to a potentially huge set
of token ids for every lexer state.  Grouping terminals and ranges avoids
materializing millions of repeated ranges.

Correctness condition:

Expanding every group into individual `(terminal, token_id)` facts produces the
same relation as a direct scan of every state/token pair.

Failure mode:

Incorrect interval merging can either lose tokens or leak tokens into terminals
that do not actually complete.

## Phase E: sparse root shortcut

Input:

Same as Phase D.

Output:

Same as Phase D.

Reason:

When the number of states and terminals at the root is small, the old sparse
walker is simpler and can be cheaper than building grouped interval machinery.

Correctness condition:

Sparse root output must expand to the same relation as grouped interval
collection.

Failure mode:

Thresholds chosen too aggressively could send large cases down a slow path.  The
path should remain controlled by explicit limits and eventually typed options.

## Phase F: sweep-line materialization

Input:

```text
state_classes × class_maps × ordered_vocab
```

Output:

```text
ScanRelationVocabMap
RuntimeCanMatchByTerminal
```

Reason:

The runtime needs token classes and weights, not grouped interval maps.  The
sweep-line converts interval start/end events into active signatures.  Each
unique signature is one scan-relation internal token class.

Correctness condition:

Two ordered tokens share a scan-relation internal id iff they have identical
sets of active `(state_class, terminal)` labels.

Failure mode:

If active groups are compared by hash only, collisions can merge distinct token
classes.  The implementation must verify exact keys after hash bucketing.

## Phase G: runtime artifact mapping

Input:

```text
local CanMatch weights × local scan-relation id map
```

Output:

```text
MappedArtifact<RuntimeCanMatchByTerminal>
```

Reason:

The scan-relation subsystem produces its own local state and token classes.  The
compile pipeline later reconciles these with Parser-DWA and Terminal-DWA token
spaces.

Correctness condition:

The mapped artifact correctly describes how local classes map back to original
lexer states and vocabulary tokens.

Failure mode:

If local ids are treated as final shared ids before reconciliation, masks will be
written into the wrong token slots.
