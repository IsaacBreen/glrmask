### Isolated components and typed bindings

GLRM distinguishes two external contracts. `extern token NAME` consumes one exact
model-token event from a bound ID set. `extern grammar NAME` invokes a separately
scoped grammar, which can contain bytes, exact events, or epsilon. Binding a value
of the wrong kind is an error; an exact ID is never replaced by its decoded text.

```rust
use glrmask::{Grammar, Module, Result, Vocab};

# fn demo(vocab: &Vocab, end_id: u32) -> Result<()> {
let grammar = Grammar::from_glrm(r#"
    extern grammar PAYLOAD
    extern token END
    start response
    nt response = PAYLOAD END
"#);
let mut parent = Module::new(&grammar);
let child = Grammar::from_json_schema(r#"{"type":"null"}"#);
parent.bind("PAYLOAD", &child)?;
parent.bind("END", glrmask::TokenSet::from(end_id))?;
let constraint = parent.compile(vocab)?;
# let _ = constraint;
# Ok(())
# }
```

Use `TokenSet::new(ids)` or `TokenSet::from_ranges(ranges)` for token sets. Rust
ranges are inclusive (`0..=9`) or half-open (`0..10`); Python `range` stops are
exclusive and `TokenSet.from_ranges` takes a list of them. The set consumes
**one** ID, not a sequence. Every ID must exist in the compile vocabulary.
IDs with the same spelling remain distinct, and spelling-less special IDs are
not epsilon. See [token sets](docs/exact-token-ranges.md).

A grammar-valued extern also accepts a `TokenSet` directly: this widens a token
event into a general fragment without changing its meaning. The conversion does
not use decoded token spelling.
An arbitrary grammar cannot bind a token-valued extern.

### Configure, precompile, attach and execute

`Module` is a reusable grammar snapshot. Constructing one from a `Grammar`
needs no vocabulary and does no grammar compilation. Strategy settings belong to
the compile call, never to the module or the source `Grammar`. The parent may
have unresolved externs while precompiling.
`bind(name, value)` mutates the receiving handle and returns `Result<()>` in Rust
or `None` in Python. In Python, `bind` also supports atomic keyword arguments
(`m.bind(**bindings)`).
Each successful bind publishes a new shared snapshot; an error leaves the previous
configuration intact. Attaching a child captures its current configuration, not
a live reference to later edits. Every child must have all of its externs bound
before it is attached. In Rust, use `clone()` to specialize an independent
snapshot; in Python, use `copy()`.

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({
    0: b"x", 1: b"|", 2: b"x", 4: b"", 5: b"",
})
leaf1 = glrmask.Module(glrmask.Grammar.from_glrm('''
    extern token MARK
    start body
    nt body = "x" MARK
'''))
leaf1.bind("MARK", glrmask.TokenSet([4]))
leaf1.precompile(vocab)

leaf2 = glrmask.Module(glrmask.Grammar.from_glrm('''
    extern token MARK
    start body
    nt body = "x" MARK
'''))
leaf2.bind("MARK", glrmask.TokenSet([5]))
leaf2.precompile(vocab)

parent = glrmask.Module(glrmask.Grammar.from_glrm('''
    extern grammar LEFT
    extern grammar RIGHT
    start pair
    nt pair = LEFT "|" RIGHT
'''))
parent.precompile(vocab)

package = parent.copy()
assert list(package.externs.unbound) == ["LEFT", "RIGHT"]
package.bind(LEFT=leaf1, RIGHT=leaf2)
assert len(package.externs.unbound) == 0

constraint = package.compile(vocab)
state = constraint.start()
for token_id in [0, 4, 1, 2, 5]:
    assert state.mask()[token_id]
    state.commit_token(token_id)
assert state.is_accepting()
```

Rust uses `Module::new(&grammar)`, `precompile(&vocab)?`,
`bind(name, &child)?`, and `compile(&vocab)?` or `compile_with(&vocab, options)?`.
Strategies are configured on the compile call via `CompileOptions`
(`mask_strategy` and `boundary_mask_strategy`; in Python: keyword arguments to
`compile` and `precompile`). One value applies to the whole compile, including
every composed child, and a part that already has a static preparation is never
downgraded to the dynamic `TokenTrie` path. Every child must be closed (have no
unbound externs) before attachment. Each extern is bound once on the module that declared it; qualified
`::` paths are rejected.

Inline grammars (`g name = { ... }`) cannot refer to anything outside
themselves, including outer rules, terminals, or externs. An outer reference
is an undefined-name error. The keywords `use`, `export`, `import`, and `super`
are reserved.

Rust `m.externs().unbound()` (Python: `m.externs.unbound`) reports unresolved
externs. Rust `m.externs()` (Python: `m.externs`) reports all declared externs
in declaration order. Introspection uses the `Extern` type (`name()`, `kind()`,
`span()`). Precompilation is allowed while externs remain open; final
`compile()` requires every declared extern to be bound.

### Persistence and root policy

Modules can be saved while incomplete, including mixed raw and precompiled
children. Retained work and source identities survive the round-trip:

```python
saved = package.save()
restored = glrmask.Module.load(saved, vocab=vocab)
assert len(restored.externs.unbound) == 0
```

Runnable constraints separately support `save()` and `Constraint.load(...)`.
Artifacts preserve original token coordinates and vocabulary identity, not merely
a set of decoded spellings. Unsupported older pre-release formats are rejected.
Retained compiled work is vocabulary-specific, not portable across arbitrary models.
The vocabulary is supplied once to each root `precompile` or `compile` operation,
not once per child. Final compilation handles outstanding work, so attaching a
cold child never requires a separate manual precompilation step. Incompatible
retained representations or targets produce errors, not implicit loss of work.

Configure `end_tokens` on the final compile operation. Acceptance means a
complete grammar match exists; it does not require generation to stop when longer
matches remain possible. A configured end ID is admitted only at acceptance. Commit
the sampled end ID before ending the generation loop. Child components do not
supply the enclosing generation's root termination policy.

### Lightweight expressions

`Expr.choice` is unordered context-free alternation, and `Expr.sequence` is
language concatenation. In Python, both accept positional parts:
`Expr.choice(*parts)` and `Expr.sequence(*parts)`. `Expr.literal` matches exact
bytes without whitespace or ignore inserted between parts. Leaves can be
`Grammar`, `Module`, `Expr`, or `TokenSet`. In Python, passing a
plain string raises `TypeError` (use `Expr.literal`). Parts with open externs
raise an error immediately. An expression can be bound directly into a
module extern (`m.bind("name", expr)`). Whole grammar and module leaves keep
their own lexical scopes and bindings; the compile options apply to them all.
Intermediate expression
operators add neither scope nor public path segments.

### Determined output

`state.forced_tokens()` returns a read-only library-canonical prefix for genuinely
forced ordinary text. It greedily chooses the longest admitted nonempty spelling,
then the lowest original-ID alias. Exact-token events retain their IDs, and the
query stops before root acceptance or an intermediate event opportunity that a
greedy text token would cross. Rust returns `Result<Vec<u32>>`; Python returns `list[int]`
and raises errors for typed proof/resource failures. The result is one canonical
sequence, not an exhaustive tree of equivalent token histories. See
[forcing contracts](docs/forcing.md).

### Streaming markers

Place `@mark(id)` in a parser expression to observe possible input boundaries.
`constraint.marker_ids()` lists the numeric IDs. `state.marker_events()` returns
the events from `start()` or the most recent commit, including markers inside a
model token. Copy a batch before the next commit when retaining history.

Markers only read the parsing frontier; they never consume a model token or
select a derivation. An event can come from an alternative that later fails.
There is no automatic pairing, capture nesting, or ambiguity resolution. The
[marker reference](docs/markers.md) defines ignored-input behavior, byte offsets,
composition, persistence, and the Rust/Python APIs.

### Build/runtime trade-off

{{dynamic}}

`MaskStrategy` trades off compilation time against per-token masking latency:

- `ParserDwa` (alias `FAST_RUNTIME`): static parser-DWA masks (slowest build, fastest runtime masks).
- `TokenTrie` (alias `FAST_BUILD`): dynamic token-trie walk (fastest build).
- `TokenClassTrie`: token-trie walk over token equivalence classes.
- `Auto`: library chooses the strategy (default).

`BoundaryMaskStrategy` selects the boundary parser for crossings between composed
grammars:

- `ParserDwa` (alias `FAST_RUNTIME`): static boundary parser; faster crossings. An over-budget build returns `Error::ResourceLimit` with a `boundary_parser_dwa:` message.
- `TokenTrie` (alias `FAST_BUILD`): segmented dynamic boundary token walk; faster build.
- `Auto` (default): `ParserDwa`, falling back to `TokenTrie` for each boundary whose static parser declines on budget.

All modes preserve accepted-language semantics and produce the same public `Constraint` type.

## Grammar formats

**JSON Schema acceptance policy.**
Compilation enforces supported assertions or reports an unsupported-feature error.
An absent `$schema` selects Draft 2020-12; explicit Draft 4, 6, 7, 2019-09, and
2020-12 declarations select their dialect semantics. `format` is annotation-only;
required format-assertion and unknown required vocabularies are unsupported.
`oneOf` remains exclusive, with no approximation mode. Existing canonical
serialization limits remain documented in the [acceptance policy](https://github.com/IsaacBreen/glrmask/pull/13).

GLRM is the native recognition language. A version header and statement semicolons
are not required; canonical source uses explicit declarations and insignificant
line breaks:

```glrm
start value
internal t DIGIT = /[0-9]/
t INTEGER = /-?(0|[1-9][0-9]*)/
nt value = INTEGER | "null"
```

It retains `internal t`, unordered context-free alternatives, scoped grammars,
regular intersection/subtraction, structural nonterminal-alternative subtraction,
separated sequences, explicit finite automata and Unicode/byte expressions.
Use regular-language subtraction for prefix-minimal matching. See [the native format](docs/glrm-format.md).

Lark is a recognition-focused frontend with selected GLRMask extensions. It adds
exact-token syntax, named token externs, `@name` grammar externs and isolated inline
`%json` schemas. Those literal families are **not native GLRM syntax**; native
callers provide equivalent components through typed host bindings. The regular-body
`[lazy]` annotation uses the same shortest terminal operation. See
[the Lark dialect](docs/lark-dialect.md).

Regex-internal reluctant quantifiers, priorities, captures, substring generation,
sampling annotations and grammar token/character budgets are not supported.
Unsupported semantic features produce errors rather than being silently discarded.
The Lark frontend does not claim Python regex endpoint ordering or compatibility
with every Python Lark lexer mode.
