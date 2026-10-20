GLRMask has four core types:

- `Grammar`: the source-language definition, without compilation settings.
- `Module`: a reusable, independently configured grammar with bindings and any retained compilation work. It may still have unresolved externs.
- `Constraint`: a fully runnable grammar compiled against a specific vocabulary.
- `ConstraintState`: the mutable state for a single generated sequence.

Use direct `Grammar.compile` for a closed source, or a `Module` when local
settings, late-bound children, or retained work are needed. `Expr.choice` and
`Expr.sequence` group grammar leaves without adding an extra lexer scope at each
operator. See [modules and expressions](https://github.com/IsaacBreen/glrmask/blob/main/docs/module-api.md)
for construction, open precompilation, and persistence examples.
