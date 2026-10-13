# Exact-token range bindings

An `extern token` slot matches **one model token** from its bound set. It does
not match a sequence of IDs, nor the byte spelling shared by those IDs. Two
vocabulary IDs with identical bytes remain distinct; exact-only and empty-byte
tokens are supported as well.

## Python

Use a normal Python `range` for one contiguous interval, or `token_ranges` for a
union of intervals. Python range stops are **exclusive**:

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({i: b"" for i in range(131_072)})
grammar = glrmask.Grammar.from_glrm(
    "extern token T start root nt root = T"
)

all_tokens = vocab.tokens(range(131_072))
# All IDs except 65,535, represented by two intervals rather than 131,071 IDs.
selection = vocab.token_ranges([range(0, 65_535), range(65_536, 131_072)])
constraint = grammar.bind("T", selection).compile(vocab)
state = constraint.start()
state.commit_token(100_000)
assert state.is_accepting()

assert selection.ranges == [range(0, 65_535), range(65_536, 131_072)]
assert len(selection) == 131_071
```

A token set can also be attached to an already compiled open component:

```python
module = grammar.prepare(vocab)
bound = module.bind("T", selection)
constraint = bound.link()
# module is still open: binding returns a new immutable description.
```

`tokens([id, ...])` and other iterables remain supported. Contiguous ranges with
step `1` or `-1` take the compact path; strided ranges passed to `tokens` are
consumed as individual IDs and may require many intervals. `token_ranges`
requires nonempty contiguous Python ranges and accepts an iterable or generator
of them. An empty iterable denotes an empty event set; an individual empty or
non-contiguous range is still invalid. Overlapping, adjacent and unordered
intervals are normalized.

`selection.ranges` and `len(selection)` inspect the set without expanding it.
`selection.ids` deliberately materializes the individual IDs and can be large.

## Rust

Rust `token_ranges` accepts **inclusive** `RangeInclusive<u32>` values:

```rust
use glrmask::{Grammar, Vocab};

let vocab = Vocab::new_with_exact_token_ids(
    Vec::<(u32, Vec<u8>)>::new(),
    0..131_072,
);
let selection = vocab.token_ranges([0..=65_534, 65_536..=131_071])?;
assert_eq!(selection.len(), 131_071);
assert_eq!(selection.ranges().count(), 2);

let grammar = Grammar::from_glrm(
    "extern token T start root nt root = T"
).bind("T", selection)?;
let constraint = grammar.compile(&vocab)?;
```

The existing `Token::new(id)` and `vocab.tokens(ids)` APIs remain available.
`ExactTokens::ids()` explicitly materializes a sorted slice on first use;
`ExactTokens::ranges()` does not. Compilation and linking use compact intervals,
not that materialized slice.

## Validation and performance model

Every selected ID must exist in the binding's vocabulary. A range crossing an
undefined vocabulary hole is rejected rather than silently clipped. Empty unions
are valid and denote the empty language, not epsilon. Reversed Rust ranges and
IDs outside the model's domain are errors. Repeated IDs
in an explicit `tokens(ids)` list retain their existing error behavior; overlaps
between ranges are a valid union. Binding and loading retain exact vocabulary
identity checks.

For **R normalized intervals**, parser-visible exact-token events and associated
weighted transitions scale with R, not the number of selected model IDs. The
compiler partitions token equivalence classes at interval boundaries, including
when intervals overlap. Commit checks interval membership; mask generation
checks each interval's own parser transition, then fills the allowed mask in
word-sized chunks. Sharing an ID with another, currently inactive interval does
not make the rest of that interval valid.

This does not make vocabulary preparation or dense output free. The vocabulary
builds its reusable validity index once, original-ID maps still have a dense
vocabulary domain, and writing a full token mask necessarily touches its output
words. `tokens(explicit_ids)` must also read those IDs. Contiguous range input is
the efficient construction path for all or nearly all tokens.

Source bindings, compiled open-slot bindings, all public optimization modes,
repeated token events and save/load retain the same ID-set semantics. Prepared components preserve the backend-independent source contract; the selected
final optimization mode chooses a supported execution representation. Source,
prepared and loaded paths must preserve exact event identity. The changed serialized terminal representation
uses new artifact versions; older pre-release artifacts must be recompiled.
