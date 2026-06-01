# Next compile repair map after Chunk 08

Chunk 08 is structural and intentionally not compile-driven.  When compilation
starts, use this order.  Do not revert the architecture to fix an import.

## 1. Module declarations

Check that every directory module has exactly one `mod.rs` and no conflicting
sibling `.rs` file.  For JSON Schema this means:

```text
load/mod.rs
normalize/mod.rs
lower/mod.rs
lower/array/mod.rs
lower/string/mod.rs
lower/number/mod.rs
lower/object/mod.rs
schema/mod.rs
tests/mod.rs
```

## 2. Import paths

Expected path changes:

```text
super::ast::*              -> super::schema::*
super::error::*            -> super::diagnostics::*
super::combinators::*      -> super::normalize::*
super::string::*           -> super::lower::string::* from tests
```

Inside `lower/object/mod.rs`, `super::string` is correct because object and
string are sibling modules under `lower`.

## 3. Visibility

If a compile error says a moved helper is private, do not make it globally public
by default.  Use this hierarchy:

1. private `fn`,
2. `pub(super) fn`,
3. `pub(crate) fn`,
4. public API only if root crate users need it.

## 4. Expected first errors

Likely first compile errors will be:

- stale `super::ast` references in moved files,
- stale `super::string` references from tests,
- module privacy of `lower::string`,
- duplicate or missing re-export of `SchemaType`,
- references to `normalize` functions from object lowering.

Fix these mechanically.

## 5. What not to do

Do not:

1. move combinator code back into `lower`,
2. merge schema structs back into a monolithic `ast.rs`,
3. put environment variable reads inside lowerer functions,
4. move tests back to a flat `tests.rs` solely to fix module paths,
5. hide exactness uncertainty by deleting docs.

The shape is the target; compiler repair should conform to the shape.
