# Cross-chunk dependency map

## Depends on Chunk 03

Chunk 03 created compile pipeline phases.  Chunk 10 now tightens the last phase:
`finalize_runtime_constraint` packages semantic parts instead of constructing
cache fields directly.

## Depends on Chunk 05

Parser DWA construction now has a clearer subsystem.  Chunk 10 treats the Parser
DWA as a semantic artifact field.

## Depends on Chunk 06

Scan relation / CanMatch cleanup clarified that CanMatch weights have final
internal coordinates.  Chunk 10 documents those coordinates in artifact
`token_space.rs`.

## Depends on Chunk 09

GLR machinery moved under `parser/glr`.  Chunk 10 stores the `GLRTable` as a
semantic parser artifact and does not move parser implementation code.

## Enables Chunk 11

Now that cache finalization has moved out, `runtime/constraint.rs` can be split
without also carrying cache rebuild code.

## Enables Chunk 12

Mask code can be reviewed against a clearer artifact API: it consumes semantic
fields and caches but should not build them.

## Enables Chunk 13

Commit code can be reviewed against the same artifact API, especially tokenizer
fast transitions and template DFAs.

## Enables Chunk 14

Template DFA storage is localized.  The next template-DFA subsystem pass can
promote stack-effect recognizers without searching through artifact aliases.
