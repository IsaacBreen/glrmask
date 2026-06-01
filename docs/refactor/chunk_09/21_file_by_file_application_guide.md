# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## File-by-file application guide

### `src/lib.rs`

Add the parser module declaration after the import/front-end modules.  Keep it crate-private.

### `src/parser/mod.rs`

Create a short module-level document explaining that parser-domain machinery is shared by compile and runtime.

### `src/parser/glr/mod.rs`

Declare `accumulator`, `advance`, `analysis`, `labels`, and `table`.  Do not declare `parser`.

### `src/compiler/glr/mod.rs`

Replace the old implementation module declaration with a hidden compatibility shim.  Do not leave old source files under `src/compiler/glr`.

### `src/parser/glr/advance/mod.rs`

Keep public crate-private entry points visible.  Move long internal bodies to included fragments.  The facade should show the algorithmic reading order.

### `src/parser/glr/table/options.rs`

Define `GLRTableOptions`.  Preserve old env variable names and disable-flag polarities.

### `src/parser/glr/table/optimize.rs`

Make the file a pass index.  It should not be the 3500-line implementation body anymore.

### `src/parser/glr/analysis.rs`

Make the file an analysis index.  It should not be a 2800-line mixed normalization implementation anymore.
