### Composition and reusable grammars

Use `extern token NAME` for a one-event token-set contract and
`extern grammar NAME` for a general byte/event grammar fragment. Bind
children to a `Module`. A child must have every extern bound before it is
attached. Modules keep bindings and retained work; strategy values are supplied
to the compile call, and `Grammar` stores no settings:

```rust
use glrmask::{Grammar, Module, Result, Vocab};

# fn demo(vocab: &Vocab, marker: u32) -> Result<()> {
let child_grammar = Grammar::from_glrm(
    r#"extern token MARK start body nt body = "item" MARK"#,
)?;
let mut child = child_grammar.precompile(vocab)?;
child.bind("MARK", marker)?;
let mut parent = Module::new(&Grammar::from_glrm(
    "extern grammar ITEM start root nt root = ITEM",
)?);
parent.bind("ITEM", &child)?;
parent.precompile(vocab)?;
let constraint = parent.compile(vocab)?;
# let _ = constraint;
# Ok(())
# }
```

`grammar.precompile(vocab)` is the module-returning mirror of
`grammar.compile(vocab)`: it is `Module::new(&grammar)` followed by
`Module::precompile`, so composed sub-grammars can be precompiled and bound in
one step.

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
[native syntax](docs/glrm-format.md), and [token sets](docs/exact-token-ranges.md).

`Expr.choice` and `Expr.sequence` combine whole grammar, module, or token leaves
without adding scopes at intermediate grouping operators. `Expr.literal` matches
exact bytes. The pieces retain independent bindings and any retained work.
Expressions can be bound directly to module externs.

### Choosing a mask strategy

`MaskStrategy` chooses how masks are computed:

- `ParserDwa` builds a parser DWA for the grammar. Slowest to compile, with the fastest masks.
- `TokenTrie` computes each mask by walking the vocabulary. Fastest to compile, with slower masks.
- `Auto`, the default, reuses any precompiled work; otherwise it uses `ParserDwa`, or `TokenTrie` when that would be too large.

`FAST_MASKS` and `FAST_COMPILE` are aliases for `ParserDwa` and `TokenTrie`.

Two compile options tune `TokenTrie`. Both are off by default:

- `template_dfas` parses with template DFAs instead of an LR table. Slower to compile, with faster masks.
- `merge_equivalent_tokens` merges tokens that the grammar treats the same, so masks walk fewer tokens. It requires `template_dfas`.

`BoundaryMaskStrategy` chooses how masks are computed at grammar boundaries, where a grammar meets a bound extern. `ParserDwa` is slower to compile, with faster masks, and fails with `ErrorKind::ResourceLimit` if it would be too large. `TokenTrie` is faster to compile, with slower masks. `Auto`, the default, uses `ParserDwa`, or `TokenTrie` when that would be too large.

Options are passed to each `compile` and `precompile` call (in Python, as keyword arguments) and apply to the whole compile, including bound grammars. With `TokenTrie`, parts already precompiled with `ParserDwa` keep their masks; only the rest is compiled with `TokenTrie`, using template DFAs. With `merge_equivalent_tokens` on, every part is recompiled.

```rust
# use glrmask::{CompileOptions, Grammar, MaskStrategy, Result, Vocab};
# fn demo(grammar: &Grammar, vocab: &Vocab) -> Result<()> {
let constraint = grammar.compile_with(
    vocab,
    CompileOptions::default()
        .mask_strategy(MaskStrategy::TokenTrie)
        .template_dfas(true),
)?;
# let _ = constraint;
# Ok(())
# }
```

### Completion and forcing

To require an end-of-sequence token, put it after the grammar with `Grammar::from_expr(Expr::sequence(...))`. The token is allowed once the grammar is complete. After it is committed the mask is empty. If the token has no bytes, add its ID with `with_special_tokens`.

`is_accepting()` means the input so far is a complete match, which may still allow more input. `is_terminated()` says whether anything more can be committed.

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
inline JSON literals. Lark accepts those selected compatibility forms. Unsupported priorities, capture/sampling annotations,
substring generation and regex-internal lazy quantifiers are errors.
