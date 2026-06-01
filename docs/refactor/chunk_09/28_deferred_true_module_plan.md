# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Converting textual fragments into true modules later

### Step 1: Make dependencies explicit

For each fragment, list every helper it calls outside itself.  Convert wildcard imports into explicit imports.

### Step 2: Choose visibility

Use the narrowest visibility that works:

- private within module when possible;
- `pub(super)` for sibling fragments;
- `pub(crate)` only when another subsystem genuinely needs it.

### Step 3: Move tests with their implementation

Large `tests.rs` fragments should become true `#[cfg(test)] mod tests` inside the relevant module.

### Step 4: Compile after each fragment family

Do not convert all includes at once.  Suggested order:

1. advance fragments;
2. analysis fragments;
3. optimizer guarded fragments;
4. optimizer table-pass fragments;
5. table build/mod/row split.

### Step 5: Delete include facade comments

Once true modules exist, the facade should use `mod foo;` declarations rather than `include!`.
