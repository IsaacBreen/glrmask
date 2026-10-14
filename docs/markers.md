# Streaming markers

Markers report possible grammar boundaries while input is committed. They are
passive observations: they do not consume a model token, execute a parser
transition, change the live parsing frontier, or select a parse.

Use `@mark(id)` in a parser expression. The ID is an application-chosen unsigned
32-bit integer, unrelated to model-token IDs. The same ID may appear in several
places or repeat during matching.

```glrm
glrm 1;
start root;
nt root = @mark(10) "hello" @mark(20) "world" @mark(30);
```

GLRM and EBNF accept this syntax. Lark accepts it only with the explicit `llm`
profile; it is not a standard Lark extension enabled by the recognition profile.
Markers belong in parser expressions, not inside regexes or byte-terminal
definitions. No vocabulary entry or `bind()` call is needed for a marker.

## Rust

```rust
use glrmask::{Grammar, MarkerEvent, Result, Vocab};

# fn main() -> Result<()> {
let vocab = Vocab::new(vec![(0, b"helloworld".to_vec())]);
let grammar = Grammar::from_glrm(
    r#"glrm 1; start root; nt root = @mark(10) "hello" @mark(20) "world" @mark(30);"#,
);
let constraint = grammar.compile(&vocab)?;
assert_eq!(constraint.marker_ids(), &[10, 20, 30]);

let mut state = constraint.start();
assert_eq!(state.marker_events(), &[MarkerEvent { id: 10, byte_offset: 0 }]);

state.commit_token(0)?;
assert_eq!(state.marker_events(), &[
    MarkerEvent { id: 20, byte_offset: 5 },
    MarkerEvent { id: 30, byte_offset: 10 },
]);
assert!(state.is_accepting());
# Ok(())
# }
```

`MarkerId` is a `u32` alias. A `MarkerEvent` is a small `Copy` value containing
only `id` and `byte_offset`. `marker_ids()` returns the sorted, distinct IDs in
the compiled constraint.

`marker_events()` borrows the batch produced by `start()` or the most recent
commit. Read initial events before the first commit. Each subsequent commit
replaces the batch, even when it produces no events. To retain history, copy or
append each batch to application-owned storage. GLRMask reuses its event buffer
instead of retaining the whole sequence's event history.

`commit_token()` and `commit_bytes()` keep their usual return types. A failed
commit leaves no events from that attempt. Clone-based speculation copies the
marker state as well as the parser state, so branches remain independent.
Calling `mask()`, `fill_mask()`, `forced()`, or `forced_token_ids()` does not
change the last reported batch or publish events from internal probes.

## Python

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({0: b"helloworld"})
constraint = glrmask.Grammar.from_glrm(
    'glrm 1; start root; nt root = @mark(10) "hello" @mark(20) "world" @mark(30);'
).compile(vocab)

state = constraint.start()
assert constraint.marker_ids() == [10, 20, 30]
assert state.marker_events() == [(10, 0)]
state.commit_token(0)
assert state.marker_events() == [(20, 5), (30, 10)]
```

Python returns a list of `(marker_id, byte_offset)` tuples. This is a value copy;
modifying it does not change the constraint state. Names, capture objects, and
parse-tree nodes are not part of the event API.

## Positions and ordering

Positions are absolute byte offsets in the input supplied to the state, not
Unicode-character indices or offsets in an independently detokenized display.
A marker can occur inside one model token or between the bytes of different
model tokens. A partially consumed grammar terminal does not by itself create a
fresh boundary at which later markers are reported.

For `commit_bytes()`, offsets count the supplied bytes. For `commit_token()`,
offsets count that token's vocabulary byte spelling, including the spelling of
an exact-token event. An empty-spelled exact token advances no bytes. A final
root end token terminates generation without adding its spelling to the offset.
Thus different zero-byte events can share a byte position; markers do not encode
their order or multiplicity.

Each `(id, byte_offset)` pair is reported at most once on a state lineage.
Within a batch, events are ordered by `(byte_offset, id)`, not by grammar traversal
or nesting. Multiple parser alternatives reaching the same marker at the same
position do not create duplicate events. Two different IDs at the same position
remain separate observations. The same ID at different positions remains useful
for repeated boundaries.

## Ambiguity is not an error

```glrm
glrm 1;
start root;
nt root = "a" @mark(1) "b" | "a" @mark(2) "c";
```

After committing `a`, both `(1, 1)` and `(2, 1)` may be reported. Committing `b`
afterward does not retract the observation from the other alternative. The
marker stream records possibilities observed during recognition; it is not an
extraction from the eventual successful derivation.

GLRMask does not pair start/end IDs, infer nesting, resolve alternative captures,
or recover histories its recognizer has discarded. Applications that interpret
markers as field boundaries must arrange their grammar so those observations
are meaningful for that application.

Ignored input deserves particular attention. With an ignored one-space terminal
and `"a" @mark(1) "b"`, input `a  b` can report marker 1 at offsets 1, 2, and 3:
each ignored-terminal boundary leaves the marker admissible. This is intentional,
not an instruction to choose the first or last boundary. Scope ignores or place
markers beside explicit delimiters when a single position is needed.

Lexical ambiguity and longest-match decisions can also affect when a possible
boundary is observed. Do not use the event stream as a canonical segmentation
independent of how ambiguous input was incrementally supplied. Fixed, explicit
grammar-terminal boundaries are the simplest basis for application annotations.

A nullable child has a similar consequence. In `"(" @mark(1) child @mark(2) ")"`,
where `child` accepts either the empty string or `a`, committing `(` can report
both IDs at offset 1. Committing `a` can then report marker 2 again at offset 2.
The earlier observation is not retracted. This deliberately exposes possible
boundaries instead of guessing which pair the application intended.

## Compilation, composition, and persistence

Marker IDs survive source bindings, prepared grammar reuse, compiled-child
linking, and constraint save/load, including external-vocabulary artifacts.
Component linking preserves application IDs rather than assigning private
namespaces. Reusing an ID across components intentionally combines observations;
choose different IDs when the application needs to distinguish components.

The compiler retains observation-only terminal predicates separately from the
input-token inventory. During commit, marker reporting queries those predicates
against existing parser configurations; it never executes the marker's formal
transition. Candidate lookup uses parser-top information before deeper stack
queries are needed. No parse forest or second parser is maintained.

Unannotated constraints do not allocate marker state. Their inner commit loops
use a no-op observer specialization, and mask traversal does not perform marker
queries. Annotated constraints add predicate and event-buffer work to commits;
this is not a universal bound proportional only to the number of emitted events.
Marker compilation remains subject to explicit representation/resource limits.

Measured costs, test coverage, and reproduction details are recorded in the
[14 October validation report](marker-validation-2026-10-14.md).
