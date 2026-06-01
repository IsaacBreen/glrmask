# Future compile-repair guide for Chunk 10

This chunk intentionally did not compile.  When compile repair begins, follow
this order.

## 1. Module discovery

Run:

```text
cargo check
```

Resolve errors about missing modules first.  Do not begin with type errors.

Expected touched modules:

```text
runtime/artifact/mod.rs
runtime/mod.rs
compile/pipeline/finalize.rs
```

## 2. Visibility errors

If a method moved to `artifact/caches.rs` is private but called from another
runtime module, decide whether it is truly part of the cache-builder API.  If it
is, use `pub(crate)`.  If it is not, move the caller instead.

## 3. Import warnings

Because the crate denies warnings, remove unused imports only after module/type
errors are fixed.  Do not flatten the new module structure to silence warnings.

## 4. Serialization derive errors

If the envelope derive fails, check whether `Constraint` still derives both
`Serialize` and `Deserialize` in `compiled.rs`.

## 5. Runtime behavior tests

After the crate checks, run minimal round-trip tests:

1. compile tiny EBNF;
2. save;
3. load;
4. compare first mask;
5. commit same token sequence through both original and loaded constraint.

## 6. Do not optimize yet

No performance tuning belongs in compile repair.  The goal is structural
correctness first.
