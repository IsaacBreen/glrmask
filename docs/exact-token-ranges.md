# Token-set bindings

An `extern token` slot matches **one model token** from its bound `TokenSet`. It
does not match a sequence of IDs, nor the byte spelling shared by those IDs. Two
vocabulary IDs with identical bytes remain distinct; special IDs with no byte
spelling and empty-byte tokens are supported as well.

## Python

`TokenSet(ids)` takes individual token IDs. `TokenSet.from_ranges` takes Python
`range` objects with step 1; Python range stops are **exclusive**:

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({i: b"" for i in range(131_072)})
grammar = glrmask.Grammar.from_glrm(
    "extern token T start root nt root = T"
)

all_tokens = glrmask.TokenSet(range(131_072))
# All IDs except 65,535, represented by two intervals rather than 131,071 IDs.
selection = glrmask.TokenSet.from_ranges([range(0, 65_535), range(65_536, 131_072)])
grammar.bind("T", selection)
constraint = grammar.compile(vocab)
state = constraint.start()
state.commit_token(100_000)
assert state.is_accepting()

print(selection)  # TokenSet[0..=65534, 65536..=131071]
```

A token set can also be attached to an already compiled open component:

```python
module = glrmask.Grammar.from_glrm(
    "extern token T start root nt root = T"
).prepare(vocab)
bound = module.clone()
bound.bind("T", selection)
constraint = bound.link()
# module is still open: its independent clone was specialized.
```

`TokenSet(range(n))` reads the range's IDs when constructing. Overlapping,
adjacent and unordered intervals are normalized, and duplicate IDs are ignored.
An empty or reversed range contributes nothing.

## Rust

`TokenSet::new(ids)` takes individual token IDs. `TokenSet::from_ranges` accepts
any `RangeBounds<u32>`, including inclusive `0..=9` and half-open `0..10`:

```rust
use glrmask::{Grammar, TokenSet, Vocab};

let vocab = Vocab::new(Vec::new()).with_special_tokens(0..131_072);
let selection = TokenSet::from_ranges([0..=65_534, 65_536..=131_071]);

let mut grammar = Grammar::from_glrm(
    "extern token T start root nt root = T"
)?;
grammar.bind("T", selection)?;
let constraint = grammar.compile(&vocab)?;
```

A single ID can be bound directly: `grammar.bind("T", 9)?;`. `TokenSet` is
input-only: it exposes no ID or range accessor. Its `Debug` output prints the
normalized ranges.

## Node and C/C++

Node: `new TokenSet(ids)` and `TokenSet.fromRanges([{start, end}])` with
half-open ranges. C: `glrmask_token_set_new(ids, n, &set)` and
`glrmask_token_set_from_ranges(ranges, n, &set)` with inclusive `{first, last}`
entries. C++: `TokenSet(ids)` and `TokenSet::from_ranges({{first, last}, ...})`.
Empty or reversed ranges contribute nothing in every binding.

## Validation and performance model

Every selected ID must exist in the compile vocabulary. A range crossing an
undefined vocabulary hole is rejected rather than silently clipped. Empty sets
are valid and denote the empty language, not epsilon. Reversed ranges and IDs
outside the model's domain are errors. Duplicate IDs in an explicit ID list are
ignored because a token set is a set; overlaps between ranges are a valid union.

For **R normalized intervals**, parser-visible token events and associated
weighted transitions scale with R, not the number of selected model IDs. The
compiler partitions token equivalence classes at interval boundaries, including
when intervals overlap. Commit checks interval membership; mask generation
checks each interval's own parser transition, then fills the allowed mask in
word-sized chunks. Sharing an ID with another, currently inactive interval does
not make the rest of that interval valid.

This does not make vocabulary preparation or dense output free. The vocabulary
builds its reusable validity index once, original-ID maps still have a dense
vocabulary domain, and writing a full token mask necessarily touches its output
words. `TokenSet::new(explicit_ids)` must also read those IDs. Range input is
the efficient construction path for all or nearly all tokens.

Source bindings, compiled open-slot bindings, all public optimization modes,
repeated token events and save/load retain the same ID-set semantics. Prepared components preserve the backend-independent source contract; the selected
final optimization mode chooses a supported execution representation. Source,
prepared and loaded paths must preserve token-event identity. The changed serialized terminal representation
uses new artifact versions; older pre-release artifacts must be recompiled.
