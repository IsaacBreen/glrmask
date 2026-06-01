# Compile repair strategy
When compilation begins, do not flatten the tree to silence errors.  Repair in
this order:

1. Fix module declarations and imports.
2. Fix visibility for helpers that moved across sibling modules.
3. Fix doc links and rustdoc paths.
4. Run rustfmt.
5. Run unit tests for JSON importer only.
6. Run all tests.
7. Only then split the remaining large files internally.

Any compile error that suggests moving code back to flat files should be treated
as a design smell in the repair, not a reason to undo Chunk 08.
