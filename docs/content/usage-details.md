### Composition and reusable grammars

Use `extern token NAME` for a one-event exact-token contract and
`extern grammar NAME` for a general byte/event grammar fragment. Bind source
children to `Grammar`, or independently prepared children to `PreparedGrammar`.
Neither child needs to be complete before attachment:

```rust
use glrmask::{Grammar, Result, Vocab};

# fn demo(vocab: &Vocab, marker: u32) -> Result<()> {
let child = Grammar::from_glrm(
    r#"extern token MARK start body nt body = "item" MARK"#,
).prepare(vocab)?;
let mut parent = Grammar::from_glrm(
    "extern grammar ITEM start root nt root = ITEM",
).prepare(vocab)?;
parent.bind("ITEM", &child)?;
parent.bind("ITEM::MARK", glrmask::Token::new(marker))?;
let constraint = parent.link()?;
# let _ = constraint;
# Ok(())
# }
```

Binding mutates the receiving handle atomically. `clone()` makes an independently
editable handle sharing immutable internals. Children are captured as snapshots;
editing the original child later does not change existing parents. Repeated
references to one inline grammar share bindings, while separately attached
occurrences can be specialized independently.
`requirements()` (Python: `.requirements`) exposes outstanding typed paths such as
`ITEM::MARK`. Final compile/link requires every declared parameter to be resolved.
Open prepared graphs support save/load with the same vocabulary identity.

Exact token sets and compact ranges consume one original ID, never its decoded
spelling. A JSON Schema is an ordinary grammar-valued child. Exact event values
can also bind directly to grammar slots; `Grammar::exact` constructs a standalone
event-set grammar when needed. See the [full guide](GUIDE.md),
[native syntax](docs/glrm-format.md), and [exact-token ranges](docs/exact-token-ranges.md).

### Choosing an optimization mode

`FastBuild` favors short cold starts using ordinary Dynamic, `Balanced` uses the
native middle-ground configuration, and `FastRuntime` favors mask-generation speed.
`Auto` is the default. Python exposes the corresponding uppercase names. These are
implementation trade-offs, not different language semantics.

```rust
# use glrmask::{BuildOptions, Grammar, Optimization, Result, Vocab};
# fn demo(grammar: &Grammar<'_>, vocab: &Vocab) -> Result<()> {
let constraint = grammar.compile_with(
    vocab,
    BuildOptions::default().optimization(Optimization::FastBuild),
)?;
# let _ = constraint;
# Ok(())
# }
```

### Completion and forcing

Set `end_tokens` on the final compile or link call. `is_accepting()` means the
current output is a complete match, possibly with valid continuations. Commit the
sampled end token before stopping. An initially accepting grammar can instead stop
immediately if that is the application's policy.

The read-only `forced_tokens()` query returns only individually inevitable
original IDs; it never chooses between alternative tokenizations or constructs a
full token mask merely to prove uniqueness. It stops at root acceptance or a
non-singleton allowed set, and reports proof/resource failures as errors. Raw-byte
`commit_bytes()` remains a separate recognition operation, not a substitute for
committing exact model IDs. See [the forcing contract](docs/forcing.md).

### Source formats

Native GLRM has optional semicolons, `grammar NAME { ... }`, explicit `t`/`nt`
declarations, typed externs, scoped ignore, Unicode/byte expressions and regular
`shortest { ... }`. Native source contains no numeric-token, special-token-name or
inline JSON literals. Lark accepts those selected compatibility forms only in its
explicit `llm` profile. Unsupported priorities, capture/sampling annotations,
substring generation and regex-internal lazy quantifiers are errors.
