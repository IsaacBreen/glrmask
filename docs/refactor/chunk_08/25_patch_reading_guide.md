# Patch reading guide

Read the patch in this order:

1. `src/import/json_schema/mod.rs` — shows the new stage boundary.
2. `src/import/json_schema/schema/` — shows the typed model split.
3. `src/import/json_schema/load/` — shows raw JSON parsing separated from
   grammar lowering.
4. `src/import/json_schema/lower/mod.rs` — shows the lowerer context moved under
   a lowering namespace.
5. `src/import/json_schema/lower/{array,string,number,object,combinators}.rs` —
   show domain lowerers moved under the lowerer namespace.
6. `docs/refactor/chunk_08/*` — explains semantic claims and follow-up tasks.

Do not start by reading `lower/object.rs`; it remains huge.  The purpose of this
chunk is to isolate it so that the next object-specific chunk can safely split it
without also changing schema loading.
