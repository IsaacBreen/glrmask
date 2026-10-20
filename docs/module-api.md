# Reusable modules and expressions

This API is an unreleased main-branch feature. Use a build of the matching
source revision until it is included in a published package.

`Grammar` describes a language. It does not own compilation options. Use
`grammar.compile(vocab)` for a closed source grammar and `Module` when a grammar
needs independently configured children, late binding, or reusable compilation
work. A `Constraint` is always fully runnable: starting a state, calculating a
mask, or committing a token never compiles deferred grammar definitions.

## Construction and local settings

Rust uses one default constructor and an instance method for configured copies:

```rust
use glrmask::{Grammar, Module, ModuleOptions, Optimization};

fn configured(source: &Grammar<'_>) -> glrmask::Result<Module> {
    let original = Module::new(source);
    let configured = original.with_options(
        ModuleOptions::default().optimization(Optimization::FastRuntime),
    )?;
    // The source and original module are unchanged.
    Ok(configured)
}
```

Creating a module does not require a vocabulary or perform grammar compilation.
It captures an owned source snapshot. `Grammar::module()` is the equivalent
default convenience. `with_options` returns an independent configured snapshot;
`set_options` changes the receiving module in place. Both preserve existing
bindings and independently configured descendants. An incompatible change to
options already used for retained local work is an error, not an implicit
discard-and-rebuild operation. The original value survives a failed update.

Python uses its constructor and keyword arguments rather than copying Rust's
associated-function syntax:

```python
from glrmask import Grammar, Module, Optimization

source = Grammar.from_glrm('start root nt root = /[0-9]+/')
module = Module(source, optimization=Optimization.FAST_RUNTIME)
```

Options on a module apply to its own compilation unit. Attaching that module to
a different parent does not change its settings. A raw source attached directly
uses its own default configuration, not the parent's configuration. Create an
explicit configured child when its options should differ. Final root controls,
such as end tokens, are supplied to the final compilation call and are not
inherited by children.

## Open preparation and late binding

Binding and precompilation update a module in place. Binding captures the child
at that moment; later changes to the original child do not rewrite a previously
bound parent. Clone or copy a module to specialize independent branches.

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
    assert_eq!(parent.unbound()?.len(), 1);
    parent.precompile(&vocab)?;

    let saved = parent.save();
    let mut loaded = Module::load_with_vocab(&saved, &vocab)?;
    loaded.bind("ITEM", Grammar::from_glrm("start root nt root = /[0-9]+/"))?;
    assert!(loaded.unbound()?.is_empty());

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

vocab = Vocab.from_dict({"[": 0, "7": 1, "]": 2, "[7]": 3})
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

`precompile` is optional. Final `compile` accepts a wholly cold module and a
module containing already prepared children. It rejects unresolved externs and
incompatible vocabulary or compilation representations explicitly. Passing a
vocabulary once to a root operation supplies it to the whole graph; callers do
not pass one to every constructor. A loaded file may retain a target identity
and compiled data, while a newly bound raw child still needs the full vocabulary
for compilation, so final compilation takes the vocabulary explicitly.

### Representation compatibility

Local options are real requests, not suggestions silently replaced by the
parent's mode. Auto, Balanced and FastRuntime modules use native template
representations and can be combined with one another. FastBuild keeps the
ordinary Dynamic representation. Retained Dynamic/LR and native-template
components cannot be mixed across an attachment edge; this is rejected before
compilation rather than converting or discarding the supplied work.

Homogeneous Dynamic composition uses the existing retained-component linker.
Its unsupported nullable or global-ignore boundaries still produce explicit
capability errors. A FastBuild Lark unit combining inline schema expansion with
open imports also has a capability error. These constraints apply to reusable
component composition, not a claim that a source language is invalid: direct
whole-source `Grammar.compile` retains its existing compilation route. Choosing
native-compatible local options before precompilation provides the general
reusable graph path. There is no new public parser-backend switch.

The environment used during later loading cannot silently change the
representation of an existing artifact. Resource or target errors do not alter
the receiving module.

The serialized module may be open, cold, or mixed. Loading restores its stored
state; it does not compile the source or silently replace invalid retained
machinery with newly compiled data. Use `Module::load_with_vocab` in Rust or the
`vocab` argument to `Module.load` in Python to check the intended target.

## Expressions and independently scoped leaves

`Expr.choice` describes unordered language alternatives, and `Expr.sequence`
describes concatenation. They are not prioritized parsing or separate decoding
calls. A single original model token can cross a grammar-leaf boundary.

```python
from glrmask import Expr, Grammar, Module

a = Grammar.from_glrm('start root nt root = "a"')
b = Grammar.from_glrm('start root nt root = "b"')
c = Grammar.from_glrm('start root nt root = "c"')
expression = Expr.sequence([Expr.choice([a, b]), c])
module = Module(expression)
```

In Rust, mixed operand types require an explicit common `Expr` conversion:

```rust
use glrmask::{Expr, Grammar, IntoExpr, Module};

fn alternatives() -> glrmask::Result<Module> {
    let a = Grammar::from_glrm(r#"start root nt root = "a""#);
    let b = Grammar::from_glrm(r#"start root nt root = "b""#);
    let c = Grammar::from_glrm(r#"start root nt root = "c""#);
    let expression = Expr::sequence([
        Expr::choice([(&a).into_expr(), (&b).into_expr()]),
        (&c).into_expr(),
    ]);
    Module::from_expr(&expression)
}
```

Whole imported grammars retain their own lexer and ignore rules. Nested
expression operators add no extra namespace. Leaves are addressed in their
stable traversal order as `leaf0`, `leaf1`, and so on; reassociating the same
ordered leaves does not add path segments. Two appearances of an open source
are independent occurrences, even when their definitions or compiled local work
are shared. `unbound()` / Python `.unbound` reports the actual qualified paths
to use for late binding. `externs()` / `.externs` also includes bound declarations.

Configured or precompiled `Module` values are valid expression leaves and retain
their options and compiled work. `Grammar::from_expr` and `Grammar.from_expr`
are source-only conversions: they reject expressions that would lose nondefault
configuration or retained work. Use the module conversion for those expressions.
Source-only expressions can bind directly into a `Grammar`; module binding also
accepts configured expressions. No explicit wrapper is needed for normal binding.

An empty sequence denotes epsilon. An empty choice is rejected when materialized
rather than silently assigned another language. The initial expression API has
no named-operand requirement or general rule-building DSL.

## Exact token events and errors

`Token` and `ExactTokens` still denote exact original-ID events, not their decoded
text. A grammar-valued extern may accept such an event through the existing
exact-event adaptation. A token-valued extern cannot accept arbitrary text or a
general grammar. Vocabulary-bound token sets must match the target mapping,
including exact-only entries.

Unknown, repeated, wrong-kind and incompatible-target bindings leave the module
unchanged. A failed option update or failed precompilation likewise preserves
the receiver. `UnboundExtern` exposes each outstanding declaration's kind,
qualified path and source span. State predicates remain `is_accepting` and
`is_rejected`; an accepting prefix can still have continuations, and an empty
token mask does not itself mean rejection.

The old `PreparedGrammar` API is a migration surface, not the canonical name for
the new flexible object. It should not be substituted for `Module` in new examples:
its historical construction phase and accepted child types are different.
