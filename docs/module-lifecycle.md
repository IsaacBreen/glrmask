# Reusable modules and expressions

These APIs require an unreleased build of the matching main-branch source.
See [the module API guide](module-api.md) for the representation-compatibility
matrix and Python examples.

`Grammar` describes a source language. It does not store compilation settings.
`Module` owns a snapshot of that language, its external bindings and any
retained compilation work. `Constraint` is the runnable result.

Creating a cold module requires no vocabulary and performs no grammar
compilation. Binding a child captures its current state, not a live mutable
reference. Later changes to either handle do not affect previously captured
snapshots. A failed binding leaves the receiving handle unchanged.

## Local options and root options

Rust uses `CompileOptions` for the mask strategy, boundary strategy and the
final root's stopping policy. Strategy values are compile settings, not module
state: one value applies to the whole compile, including every composed child.

The instance method `with_options` returns a configured snapshot. `set_options`
updates an existing module in place. Both reject changes incompatible with
locally retained work, without discarding that work. A cold parent's options can
still change when an independently configured child has been precompiled.
Equivalent unchanged options are accepted. Neither operation puts options on
the original `Grammar`.

```rust
use glrmask::{Grammar, Module, ModuleOptions, Optimization, Result, Vocab};

# fn example(vocab: &Vocab) -> Result<()> {
let source = Grammar::from_glrm(r#"
    extern grammar ITEM
    start list
    nt list = "[" ITEM "]"
"#);
let options = ModuleOptions::default().optimization(Optimization::Balanced);
let mut parent = Module::new(&source).with_options(options)?;
parent.bind("ITEM", Grammar::from_ebnf(r#"start ::= "null""#))?;
let constraint = parent.compile(vocab)?;
# let _ = constraint;
# Ok(())
# }
```

## Optional advance compilation

`precompile(vocab)` performs the reusable work currently possible on every known
definition. It is valid while external slots remain unbound. It changes the
module in place and keeps already applicable work. It does not promise that
binding-dependent final work has been performed before those bindings exist.

`compile(vocab)` returns a runnable constraint whether the module is cold,
partially precompiled or mixed. Every external declaration must be bound. The
operation preflights the graph and target before doing expensive work. It never
asks the caller to precompile each newly attached child separately.

```rust
use glrmask::{Grammar, Module, Result, Vocab};

# fn reuse(vocab: &Vocab) -> Result<()> {
let mut parent = Module::new(&Grammar::from_glrm(r#"
    extern grammar TOOL
    start message
    nt message = "call(" TOOL ")"
"#));
parent.precompile(vocab)?;
assert_eq!(parent.unbound()?[0].name(), "TOOL");
let saved = parent.save();

let mut restored = Module::load(saved)?;
restored.bind("TOOL", Grammar::from_ebnf(r#"start ::= "ping""#))?;
let constraint = restored.compile(vocab)?;
# let _ = constraint;
# Ok(())
# }
```

The vocabulary is supplied once per root compilation operation, not once per
child. Saved work records its original token-ID/byte mapping and special-token
event domain. An incompatible vocabulary is an error; matching decoded
spellings alone are not sufficient. A cold child may still require the full
vocabulary even when its parent has already been precompiled.

Module serialization preserves source, attachment snapshots, target identity
and retained work. Loading does not compile. Malformed graph
references, excessive expansion and unsupported versions are errors. Saving a
module with unbound slots does not make it runnable.

## Expressions and separately bindable leaves

`Expr.choice` is unordered context-free alternation, not prioritized choice.
`Expr.sequence` is language concatenation, not independent tokenization of each
operand. A single model-token spelling may cross several leaf boundaries.

Nested grouping operators create no new lexer/ignore scope and no additional
public path segment. Each imported grammar leaf retains its own scope. Leaves
are addressed deterministically as `leaf0`, `leaf1`, and so on, in left-to-right
order; attaching an expression introduces the normal parent-slot prefix.

```rust
use glrmask::{Expr, Grammar, IntoExpr, Module, Result};

# fn expression() -> Result<()> {
let open = Grammar::from_glrm("extern grammar ITEM start s nt s = ITEM");
let separator = Grammar::from_ebnf(r#"start ::= ",""#);
let expression = Expr::sequence([
    open.clone().into_expr(),
    separator.into_expr(),
    open.into_expr(),
]);
let mut module = Module::from_expr(&expression)?;
module.bind("leaf0::ITEM", Grammar::from_ebnf(r#"start ::= "a""#))?;
module.bind("leaf2::ITEM", Grammar::from_ebnf(r#"start ::= "b""#))?;
assert!(module.unbound()?.is_empty());
# Ok(())
# }
```

Rust collections containing different operand types need an explicit common
`Expr` representation; `IntoExpr` provides that conversion. A configured or
precompiled `Module` may be an expression leaf, retaining its bindings and work.
Converting such an expression into a source-only `Grammar` is an error rather
than silently discarding information. Source-only expressions support
`Grammar::from_expr` and direct source binding.

An empty sequence denotes epsilon. Empty choice is rejected when materializing
an expression as source or a module. Unknown or duplicate bindings, including
duplicate qualified paths, are errors. General grammars cannot fill atomic token
slots; a token set can fill a grammar-valued slot without becoming decoded
text.

## Diagnostics and execution

`unbound()` returns only unresolved external paths. `externs()` returns all
declarations, including bound slots and declarations in attached occurrences.
Records include a qualified name, declared kind and original source coordinates
when available. Traversal and expansion limits protect callers from unbounded
diagnostic output.

No deferred grammar compilation occurs during `Constraint.start()`, masking or
token commitment. `is_accepting()` means that the prefix is a match now, not that generation has
terminated. A failed commit returns an error and kills the state: later commits
fail and the mask is empty. These runtime contracts are unchanged by the module
lifecycle.
