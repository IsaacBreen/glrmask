GLRMask has four core types:

- `Grammar`: source grammar semantics and bindings.
- `PreparedGrammar`: reusable vocabulary-specific compiled machinery, which can still have unresolved bindings.
- `Constraint`: a closed, runnable grammar compiled against a vocabulary.
- `ConstraintState`: the mutable state for one generated sequence.

Compile a `Grammar` directly, or use `prepare()` and `link()` for reusable components. Create a state with `start()`, then compute the next-token mask, sample an allowed token, and commit it.

`Grammar` and `PreparedGrammar` keep reusable definitions with mutable bindings. `bind()` fills one declared slot on the handle. Clones and previously attached children retain independent binding snapshots; a failed bind leaves the handle unchanged.

See the [full guide](https://glrmask.com/guide/) for composition, persistence, optimization and special-token details.
