### Composition and reusable grammars

Use `extern token NAME` for a one-event exact-token contract and
`extern grammar NAME` for a general byte/event grammar fragment. Bind source
children to `Grammar`, or cold and independently precompiled children to `Module`.
Neither child needs to be complete before attachment. Modules keep their own
local compilation settings; `Grammar` does not store those settings:

```rust
use glrmask::{Grammar, Module, Result, Vocab};

# fn demo(vocab: &Vocab, marker: u32) -> Result<()> {
let mut child = Module::new(&Grammar::from_glrm(
    r#"extern token MARK start body nt body = "item" MARK"#,
));
child.precompile(vocab)?;
let mut parent = Module::new(&Grammar::from_glrm(
    "extern grammar ITEM start root nt root = ITEM",
));
parent.precompile(vocab)?;
parent.bind("ITEM", &child)?;
parent.bind("ITEM::MARK", glrmask::Token::new(marker))?;
let constraint = parent.compile(vocab)?;
# let _ = constraint;
# Ok(())
# }
```

Binding mutates the receiving handle atomically. `clone()` makes an independently
editable handle sharing immutable internals. Children are captured as snapshots;
editing the original child later does not change existing parents. Repeated
references to one inline grammar share bindings, while separately attached
occurrences can be specialized independently.
`unbound()` exposes outstanding typed paths such as `ITEM::MARK`; `externs()` also
includes bound declarations. Final compilation requires every declared parameter
to be resolved. `Module.save()` and `Module.load()` preserve source, settings,
open bindings and retained work with the same vocabulary identity. Vocabulary
is supplied to root compilation operations, not every cold child constructor.

Exact token sets and compact ranges consume one original ID, never its decoded
spelling. A JSON Schema is an ordinary grammar-valued child. Exact event values
can also bind directly to grammar slots; `Grammar::exact` constructs a standalone
event-set grammar when needed. See the [full guide](GUIDE.md),
[native syntax](docs/glrm-format.md), and [exact-token ranges](docs/exact-token-ranges.md).

`Expr.choice` and `Expr.sequence` combine whole grammar or module leaves without
adding scopes at intermediate grouping operators. The leaves retain independent
bindings, options and any retained work. Use `Grammar.from_expr` for source-only
expressions, and `Module::from_expr` (Rust) or `Module(expr)` (Python) for
configured expressions. See [module lifecycle](docs/module-lifecycle.md).

### Choosing an optimization mode

`FastBuild` favors short cold starts using ordinary Dynamic, `Balanced` uses the
native middle-ground configuration, and `FastRuntime` favors mask-generation speed.
`Auto` is the default. Python exposes the corresponding uppercase names. These are
implementation trade-offs, not different language semantics.
Each Module captures its own `ModuleOptions`. A raw child uses default options,
not its parent's overrides. Unsupported combinations of retained representations
produce an error instead of silently rebuilding or discarding supplied work.

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

Set `end_tokens` on the final compile call. `is_accepting()` means the
current output is a complete match, possibly with valid continuations. Commit the
sampled end token before stopping. An initially accepting grammar can instead stop
immediately if that is the application's policy.

The read-only `forced_tokens()` query returns a safe library-canonical prefix of
genuinely forced ordinary text. It uses the longest admitted nonempty spelling
and lowest-ID alias, retains exact-event IDs, and stops at root acceptance or an
event boundary that greedy text would cross. The correctness-first reference may
construct full masks and reports typed resource failures as errors. This is one
canonical sequence, not an exhaustive tree of equivalent original-ID histories.
Raw-byte `commit_bytes()` remains separate. See [the forcing contract](docs/forcing.md).

### Source formats

Native GLRM has optional semicolons, `grammar NAME { ... }`, explicit `t`/`nt`
declarations, typed externs, scoped ignore, Unicode/byte expressions and regular
intersection/subtraction. Native `shortest { ... }` syntax is removed; the internal
prefix-minimal operation and Lark's regular-body `[lazy]` remain. Native source contains no numeric-token, special-token-name or
inline JSON literals. Lark accepts those selected compatibility forms only in its
explicit `llm` profile. Unsupported priorities, capture/sampling annotations,
substring generation and regex-internal lazy quantifiers are errors.
