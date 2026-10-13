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
let parent = Grammar::from_glrm(
    "extern grammar ITEM start root nt root = ITEM",
).prepare(vocab)?;
let complete = parent.bind("ITEM", &child)?
    .bind("ITEM::MARK", glrmask::Token::new(marker))?;
let constraint = complete.link()?;
# let _ = constraint;
# Ok(())
# }
```

Bindings are immutable. Repeated references to one inline grammar share bindings;
separate attachments of one prepared definition can be specialized independently.
`requirements()` (Python: `.requirements`) exposes outstanding typed paths such as
`ITEM::MARK`. Final compile/link requires every declared parameter to be resolved.
Open prepared graphs support save/load with the same vocabulary identity.

Exact token sets and compact ranges consume one original ID, never its decoded
spelling. A JSON Schema is an ordinary grammar-valued child. Use `Grammar::exact`
for an explicit event-set-to-grammar adapter. See the [full guide](GUIDE.md),
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

Optional `forced()` queries choose a canonical admitted encoding of determined
bytes. `forced_token_ids()` does not choose among alternative tokenizations. They
are read-only and stop at acceptance or an unproved alternative; canonical forcing
is not a claim that the model would have sampled the selected IDs. See
[the forcing contract](docs/forcing.md).

### Source formats

Native GLRM has optional semicolons, `grammar NAME { ... }`, explicit `t`/`nt`
declarations, typed externs, scoped ignore, Unicode/byte expressions and regular
`shortest(...)`. Native source contains no numeric-token, special-token-name or
inline JSON literals. Lark accepts those selected compatibility forms only in its
explicit `llm` profile. Unsupported priorities, capture/sampling annotations,
substring generation and regex-internal lazy quantifiers are errors.
