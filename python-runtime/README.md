# `_glrmask_runtime`

Python binding for the execution-only `glrmask-runtime` crate.

It does not parse JSON Schema, Lark, GLRM, EBNF, regexes, or vocabularies.
The main `_glrmask` compiler binding produces a versioned runtime artifact:

```python
artifact = compiled_constraint.save_runtime_artifact()
runtime_constraint = _glrmask_runtime.Constraint.load(artifact)
state = runtime_constraint.start()
```

`ConstraintState.fill_mask()` and `fill_mask_timed_ns()` fill a caller-owned
packed `numpy.int32` word buffer. `commit_token()` and
`commit_token_timed_ns()` execute the corresponding sampled vocabulary token.
This is the path used by CFA's `glrmask_runtime` framework.

The loaded `Constraint` keeps the immutable executor and rebuilt caches in
memory. Calling `start()` creates a fresh session without deserializing the
artifact again.

Build locally with:

```sh
maturin develop --release --manifest-path python-runtime/Cargo.toml
```
