# Risk register

## Risk: visibility broadening

Moving helpers into sibling modules required `pub(super)` on several items. Later cleanup should reduce visibility where possible.

## Risk: shared imports hide dependencies

The current split uses `use super::*` to keep the source movement low-risk. Later cleanup should replace this with explicit imports.

## Risk: fast-path coupling remains dense

`fast_path.rs` is still large. This chunk isolates fast paths from the rest of Commit but does not yet split each fast path into its own file.

## Risk: profiled implementation duplicates logic

`profiled.rs` still contains substantial duplicated control flow. A future chunk should consider a trace-observer abstraction so profiled/unprofiled paths share more structure.

## Risk: no compile pass yet

Per instruction, this chunk did not compile. The source shape is intentional, but mechanical compile repair remains a later phase.
