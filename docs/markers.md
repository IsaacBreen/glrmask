# Markers

A marker labels a point in a grammar. As text is committed, the state reports each marker the text passes, with its byte
offset. Use markers to find where parts of the output begin and end, such as the arguments of a tool call, without
parsing the output again.

```glrm
start call
nt call = "<weather>" @mark(1) ARGS @mark(2) "</weather>"
t ARGS = /[^<]+/
```

`@mark(id)` goes anywhere in a rule body. The ID is any unsigned 32-bit number you choose; it has nothing to do with
token IDs. The same ID can appear in several places. Markers work in GLRM, Lark and EBNF, but not inside a terminal or
a regular expression.

Markers never change what the grammar allows, and a grammar with no markers pays nothing for them.

## Reading markers

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({0: b"hello", 1: b"world"})
grammar = glrmask.Grammar.from_glrm('start root nt root = @mark(10) "hello" @mark(20) "world" @mark(30)')
constraint = grammar.compile(vocab)
assert constraint.marker_ids() == [10, 20, 30]

state = constraint.start()
assert state.marker_events() == [(10, 0)]
state.commit_token(0)
assert state.marker_events() == [(20, 5)]
state.commit_token(1)
assert state.marker_events() == [(30, 10)]
```

`marker_events()` lists the markers passed by the most recent commit, or by `start()` before any commit. Each commit
replaces the list, so read it after every commit if you need them all. In Python each event is a
`(marker_id, byte_offset)` pair; in Rust, Node.js, C and C++ it has `id` and `byte_offset` fields.
`constraint.marker_ids()` lists the grammar's marker IDs in ascending order.

Asking for a mask or forced tokens doesn't change the list. A failed commit leaves it empty. A copy of the state keeps
its own markers, so a copy and the original never see each other's events.

## Offsets

An offset counts bytes from the start of the sequence: the bytes of every committed token, or the bytes passed to
`commit_bytes`. It is not a character index. A marker can fall in the middle of a token. A special token has no bytes,
so a marker on either side of it has the same offset.

Within one list, events are sorted by offset, then by ID. The same marker is never reported twice at the same offset.

## Several parses

A marker means some parse reached that point. That parse may still fail later. In

```glrm
start root
nt root = "a" @mark(1) "b" | "a" @mark(2) "c"
```

committing `a` reports both markers 1 and 2 at offset 1. Committing `b` next doesn't take marker 2 back.

The same applies to ignored text and to rules that can match nothing. With spaces ignored, the grammar
`"a" @mark(1) "b"` reports marker 1 at offsets 1, 2 and 3 for `a  b`, because each space could be where `b` begins. Where
you need exactly one position for a marker, put it next to required text, such as a delimiter, that the grammar doesn't
ignore.

## Composition and saving

Markers keep their IDs through binding, precompiling, saving and loading. Parts of a composition share one set of IDs,
so give each part different IDs if you need to tell them apart.
