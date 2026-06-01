# Module boundary invariants

## `schema/`

Allowed imports:

- `std` collections and formatting traits;
- `serde_json::Value` only for storing already-loaded `const`/`enum` values.

Forbidden imports:

- `GrammarExpr`;
- `Lowerer`;
- automata/tokenizer/parser/runtime modules;
- environment variables;
- regex engines.

Invariant: `schema/` represents constraints over JSON values.  It never chooses a
text encoding strategy.

## `load/`

Allowed imports:

- `serde_json::{Value, Map}`;
- `schema::*`;
- `diagnostics::*`.

Forbidden imports:

- `GrammarExpr`;
- `Lowerer`;
- `regex_syntax` and grammar regex construction helpers;
- tokenizer/automata/runtime modules.

Invariant: `load/` either constructs typed schema syntax, records reference
locations, or rejects unsupported syntax.  It never broadens by emitting a
looser grammar because it does not emit grammar at all.

## `normalize/`

Allowed imports:

- `schema::*`;
- limited lowerer hooks only when the existing implementation requires them;
- diagnostics for exact/broadening decisions.

Forbidden imports:

- raw JSON `Value` keyword parsing;
- tokenizer/automata/runtime modules;
- env-var reads.

Invariant: every function in `normalize/` is a schema-denotation rewrite or
comparison.  Each rewrite must have an exactness or broadening argument.

## `lower/`

Allowed imports:

- `schema::*`;
- `GrammarExpr` and `NamedGrammar`;
- regex construction utilities;
- importer options and diagnostics.

Forbidden imports:

- raw JSON Schema keyword parsing outside exact-literal emission;
- compile-pipeline internals;
- runtime `Constraint`/`ConstraintState`.

Invariant: `lower/` is the only JSON importer phase that maps from value-level
schema constraints to encoded JSON text grammars.

## `tests/`

Allowed imports:

- internal importer modules;
- compile/runtime APIs needed for end-to-end acceptance checks.

Invariant: tests should be sorted by semantic feature, not by the file where a
bug happened.
