# Design alternatives rejected

## Alternative A: leave scan code under `compiler::stages`

Rejected because the paper objects should be findable by name.  A reader looking
for `CanMatch` should not have to know historical stage names.

## Alternative B: merge scan execution and scan-relation construction

Rejected because runtime scanning and compile-time relation construction are
different kinds of computation.  One evaluates a chosen byte string from one
state.  The other constructs a global relation over many states and tokens.

## Alternative C: reuse Terminal-DWA token classes

Rejected as mathematically unsound.  Completed terminal behavior does not imply
partial future-completion behavior.

## Alternative D: expose `src/scan` publicly now

Rejected because `ScanOutcome` and related wrappers are currently internal
vocabulary, not stable API.  They should become public only if the library later
commits to exposing lexer scan traces.

## Alternative E: delete the legacy materializer immediately

Rejected because it is a useful validation oracle while the grouped sweep is
being moved.  Remove it only after executable tests prove the grouped path.

## Alternative F: put all materialization helpers in one `common.rs`

Rejected for this chunk because it can become a junk drawer.  If compile repair
shows genuine shared helpers, extract a narrow `materialize_common.rs` later.

## Alternative G: rename everything in one mega-pass

Rejected because scan relation, runtime mask, runtime commit, and template DFAs
all touch lexer scanning.  Doing every rename now would make the patch harder to
review.  Chunk 06 focuses on the boundary and leaves secondary comment sweeps for
later chunks.
