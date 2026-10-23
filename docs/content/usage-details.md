### Composition and reusable grammars

Use `extern token NAME` for a one-event exact-token contract and
`extern grammar NAME` for a general byte/event grammar fragment. Bind
children to a `Module`. A child must have every extern bound before it is
attached. Modules keep bindings and retained work; strategy values are supplied
to the compile call, and `Grammar` stores no settings:

```rust
use glrmask::{Grammar, Module, Result, Vocab};

# fn demo(vocab: &Vocab, marker: u32) -> Result<()> {
let mut child = Module::new(&Grammar::from_glrm(
    r#"extern token MARK start body nt body = "item" MARK"#,
));
child.bind("MARK", glrmask::Token::new(marker))?;
child.precompile(vocab)?;
let mut parent = Module::new(&Grammar::from_glrm(
    "extern grammar ITEM start root nt root = ITEM",
));
parent.bind("ITEM", &child)?;
parent.precompile(vocab)?;
let constraint = parent.compile(vocab)?;
# let _ = constraint;
# Ok(())
# }
```

Binding mutates the receiving handle atomically. In Rust, `clone()` produces an
independent snapshot sharing immutable internals; in Python, use `copy()`.
Children are captured as snapshots; editing the original child later does not
change existing parents. Repeated references to one inline grammar share
bindings, while separately attached occurrences can be specialized
independently.
`m.externs().unbound()` (Python: `m.externs.unbound`) reports outstanding
unresolved externs; `m.externs()` (Python: `m.externs`) also includes bound
declarations. Qualified `::` paths are not supported; each extern is bound once
on the module that declared it. Final compilation requires every declared
extern to be resolved. `Module.save()` and `Module.load()` preserve source,
bindings and retained work with the same vocabulary identity.
Vocabulary is supplied to root compilation operations, not every cold child
constructor.

Exact token sets and compact ranges consume one original ID, never its decoded
spelling. A JSON Schema is an ordinary grammar-valued child. Exact event values
can also bind directly to grammar externs. See the [full guide](GUIDE.md),
[native syntax](docs/glrm-format.md), and [exact-token ranges](docs/exact-token-ranges.md).

`Expr.choice` and `Expr.sequence` combine whole grammar, module, or token leaves
without adding scopes at intermediate grouping operators. `Expr.literal` matches
exact bytes. The pieces retain independent bindings and any retained work.
Expressions can be bound directly to module externs.

### Choosing a mask strategy

`MaskStrategy` trades off compilation time against per-token masking latency:

- `ParserDwa` (alias `FAST_RUNTIME`): static parser-DWA masks (slowest build, fastest runtime masks).
- `TokenTrie` (alias `FAST_BUILD`): dynamic token-trie walk (fastest build).
- `TokenClassTrie`: token-trie walk over token equivalence classes.
- `Auto`: library chooses the strategy (the default).

`BoundaryMaskStrategy` selects the boundary parser for crossings between composed
grammars: `ParserDwa` (alias `FAST_RUNTIME`) installs the static boundary parser
and has faster crossings; an over-budget build returns `Error::ResourceLimit`
with a `boundary_parser_dwa:` message. `TokenTrie` (alias `FAST_BUILD`) keeps the
segmented dynamic boundary and builds faster. `Auto` (the default) uses
`ParserDwa` and falls back to `TokenTrie` for a boundary whose static parser
declines on budget.

These strategies are compile settings, supplied on the compile call (in Python:
keyword arguments to `compile` and `precompile`). One value applies to the whole
compile, including every composed child. `CompileOptions::end_tokens` applies to
the root compile call only; `precompile_with` accepts the same options so that
precompilation decides which static parts are built. A part that already has a
static preparation is never downgraded to `TokenTrie`:

```rust
# use glrmask::{CompileOptions, Grammar, MaskStrategy, Result, Vocab};
# fn demo(grammar: &Grammar<'_>, vocab: &Vocab) -> Result<()> {
let constraint = grammar.compile_with(
    vocab,
    CompileOptions::default()
        .mask_strategy(MaskStrategy::FAST_RUNTIME)
        .end_tokens([99]),
)?;
# let _ = constraint;
# Ok(())
# }
```

### Completion and forcing

Set `end_tokens` on the final compile call. `is_accepting()` means the
current output is a complete match, possibly with valid continuations. Commit the
sampled end token before stopping. An initially accepting grammar can instead stop
immediately if that is the application's policy.

The read-only `forced_tokens()` query returns a safe library-canonical prefix of
genuinely forced ordinary text. It uses the longest admitted nonempty spelling
and lowest-ID alias, retains exact-event IDs, and stops at root acceptance or an
event boundary that greedy text would cross. Resource limits are reported as errors. This is one
canonical sequence, not an exhaustive tree of equivalent original-ID histories.
Raw-byte `commit_bytes()` remains separate. See [the forcing contract](docs/forcing.md).

### Source formats

Native GLRM has optional semicolons, `grammar NAME { ... }`, explicit `t`/`nt`
declarations, typed externs, scoped ignore, Unicode/byte expressions and regular
intersection/subtraction. Use regular subtraction for prefix-minimal matching, or Lark's regular-body `[lazy]` annotation. Native source contains no numeric-token, special-token-name or
inline JSON literals. Lark accepts those selected compatibility forms only in its
explicit `llm` profile. Unsupported priorities, capture/sampling annotations,
substring generation and regex-internal lazy quantifiers are errors.
