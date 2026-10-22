# Reusable modules and expressions

The GLRMask v1 API structures grammar processing into three stages:
`Grammar` → `Module` → `Constraint`, plus `Expr` for composing pieces.

`Grammar` describes an immutable source language without compilation settings
or bindings. Use `grammar.compile(vocab)` for a closed source grammar and
`Module` when a grammar needs independently configured settings, late binding,
or reusable precompilation work. A `Constraint` is always fully runnable:
starting a state, calculating a mask, or committing a token never compiles
deferred grammar definitions.

## Construction and local settings

Rust wraps a source grammar with `Module::new` and configures settings via
methods on the module:

```rust
use glrmask::{Grammar, MaskStrategy, Module};

fn configured(source: &Grammar<'_>) -> glrmask::Result<Module> {
    let mut module = Module::new(source);
    module.set_mask_strategy(MaskStrategy::FAST_RUNTIME)?;
    // The source grammar is unchanged.
    Ok(module)
}
```

Creating a module does not require a vocabulary or perform grammar compilation.
It captures an owned source snapshot. Settings belong to the module, never to the
source `Grammar`. Setters (`set_mask_strategy`, `set_boundary_mask_strategy`)
update the receiving module in place and return a `Result<()>`.

Python uses constructor keywords and properties:

```python
from glrmask import Grammar, MaskStrategy, Module

source = Grammar.from_glrm('start root nt root = /[0-9]+/')
module = Module(source, mask_strategy=MaskStrategy.FAST_RUNTIME)
```

Settings on a module apply to its own compilation unit. Attaching that module to
a parent does not change its settings. An attached child must have all of its
own externs resolved before attachment. Implicit modules (a bare `Grammar` or
`Expr` passed as a binding) use default settings, not the parent's settings.
Final root controls, such as end tokens, are supplied to the root compilation
call via `CompileOptions` (Python: `end_tokens` keyword) and apply only to that
root compilation.

## Open preparation and late binding

Binding and precompilation update a module in place. Binding captures the child
as an independent snapshot at that moment; later changes to the original child
do not rewrite a previously bound parent. In Rust, use `clone()` to specialize
an independent snapshot; in Python, use `copy()`.

This complete Rust example retains useful work before its child is known:

```rust
use glrmask::{Grammar, Module, Vocab};

fn main() -> glrmask::Result<()> {
    let vocab = Vocab::new(vec![
        (0, b"[".to_vec()), (1, b"7".to_vec()), (2, b"]".to_vec()),
        (3, b"[7]".to_vec()),
    ]);
    let source = Grammar::from_glrm(
        r#"extern grammar ITEM start root nt root = "[" ITEM "]""#,
    );
    let mut parent = Module::new(&source);
    assert_eq!(parent.externs().unbound().len(), 1);
    parent.precompile(&vocab)?;

    let saved = parent.save();
    let mut loaded = Module::load(&saved, Some(&vocab))?;
    loaded.bind("ITEM", Grammar::from_glrm("start root nt root = /[0-9]+/"))?;
    assert!(loaded.externs().unbound().is_empty());

    let constraint = loaded.compile(&vocab)?;
    let mut state = constraint.start();
    state.commit_token(3)?;
    assert!(state.is_accepting());
    Ok(())
}
```

The corresponding Python workflow is:

```python
from glrmask import Grammar, Module, Vocab

vocab = Vocab.from_id_to_bytes({0: b"[", 1: b"7", 2: b"]", 3: b"[7]"})
source = Grammar.from_glrm('extern grammar ITEM start root nt root = "[" ITEM "]"')
parent = Module(source)
parent.precompile(vocab)             # Real work; an open extern is allowed.
saved = parent.save()                # Bytes, including retained work.
loaded = Module.load(saved, vocab=vocab)
loaded.bind("ITEM", Grammar.from_glrm('start root nt root = /[0-9]+/'))
constraint = loaded.compile(vocab)   # Performs any remaining required work.
state = constraint.start()
state.commit_token(3)
assert state.is_accepting()
```

`precompile` is optional. Final `compile` accepts a wholly cold module or a
module containing already prepared children. It rejects unresolved externs
explicitly. Passing a vocabulary once to a root operation supplies it to the
whole graph; callers do not pass one to every constructor.

### Mask strategy and composition

`MaskStrategy` replaces `Optimization` and controls mask generation:
- `Auto` (default): reuses retained `ParserDwa` work, else retained `TokenClassTrie` work, else builds `ParserDwa`. It never selects `TokenTrie`.
- `ParserDwa` (alias `FAST_RUNTIME`): static parser-DWA masks (slowest build, fastest masks).
- `TokenTrie` (alias `FAST_BUILD`): dynamic token-trie walk (fastest build).
- `TokenClassTrie`: token-trie walk over token equivalence classes.

`TokenTrie` compiles the whole composition as one grammar: bound children are inlined at source level, so `precompile` retains no work for `TokenTrie` modules. `Auto` parts join a `TokenTrie` composition; an explicit `ParserDwa` or `TokenClassTrie` part cannot be combined with a `TokenTrie` part, and compilation reports an error.

`BoundaryMaskStrategy` selects the boundary parser for crossings between composed
grammars:
- `ParserDwa` (default, alias `FAST_RUNTIME`): static boundary parser; faster crossings. An over-budget build is a typed compile error.
- `Dynamic` (alias `FAST_BUILD`): segmented dynamic boundary parser; faster build.

Explicit strategy requests never fall back silently. If an engine cannot build
a part or cannot be linked with neighbouring parts, compilation errors
explicitly naming the strategy. Precompilation retains work without discarding
existing work, so preparation can only accelerate compilation.

## Expressions and independently scoped leaves

`Expr.choice` describes context-free alternation, and `Expr.sequence`
describes concatenation. `Expr.literal` matches exact bytes without whitespace
or ignore rules inserted between parts.

In Python:

```python
from glrmask import Expr, Grammar, Module, Vocab

vocab = Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"c", 3: b"!"})
a = Grammar.from_glrm('start root nt root = "a"')
b = Grammar.from_glrm('start root nt root = "b"')
c = Grammar.from_glrm('start root nt root = "c"')
lit = Expr.literal("!")
expression = Expr.sequence(Expr.choice(a, b), c, lit)

parent = Module(Grammar.from_glrm('extern grammar SUB start root nt root = SUB'))
parent.bind(SUB=expression)
constraint = parent.compile(vocab)
```

In Rust:

```rust
use glrmask::{Expr, Grammar, Module, Vocab};

fn alternatives(vocab: &Vocab) -> glrmask::Result<glrmask::Constraint> {
    let a = Grammar::from_glrm(r#"start root nt root = "a""#);
    let b = Grammar::from_glrm(r#"start root nt root = "b""#);
    let c = Grammar::from_glrm(r#"start root nt root = "c""#);
    let expression = Expr::sequence([
        Expr::choice([&a, &b]),
        Expr::from(&c),
        Expr::literal("!"),
    ]);

    let mut parent = Module::new(&Grammar::from_glrm(
        "extern grammar SUB start root nt root = SUB",
    ));
    parent.bind("SUB", expression)?;
    parent.compile(vocab)
}
```

Whole grammar leaves retain their own lexer and ignore rules. Nested expression
operators add no extra namespace. Leaves can be `Grammar`, `Module`, `Expr`,
`Token`, or `ExactTokens`. In Python, passing a plain string raises `TypeError`;
use `Expr.literal`. A part with unbound externs is an error: Python's
`Expr.choice` and `Expr.sequence` raise `ValueError` immediately, and Rust
reports it when the expression is bound or compiled.

An empty sequence denotes epsilon. An empty choice matches nothing.

## Exact token events and errors

`Token` and `ExactTokens` denote exact original-ID events, not their decoded
text. A grammar-valued extern accepts `Module`, `Grammar`, `Expr`, `Token`, or
`ExactTokens` directly without requiring a separate grammar conversion. A
token-valued extern accepts `Token` or `ExactTokens`.

Binding is atomic and mutates in place: unknown extern names, repeated bindings,
wrong value kinds, or children with open externs leave the module unchanged.
Each extern is bound once on the module that declared it; qualified `::` paths
are not supported.

Introspection is provided by `m.externs()` (Python: `m.externs`), which returns
an `Externs` view of all declared externs on the module in declaration order,
and `m.externs().unbound()` (Python: `m.externs.unbound`), which reports only
unresolved externs. Each `Extern` exposes `name()`, `kind()`, and `span()`.

`PreparedGrammar`, `link()`, `requirements`, `Grammar.bind`, `Grammar.exact`,
`BoundaryTriggerDetail`, and `Optimization` are removed from the public API.
In Python, `clone()` is replaced by `copy()`.
