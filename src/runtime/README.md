# Runtime namespace

The runtime namespace contains the immutable compiled artifact and the mutable
state machine used during decoding.

The publication-facing mathematical split is:

1. `artifact/` stores the immutable compiled object: parser table, tokenizer,
   Parser DWA, scan relation, token-space maps, and rebuilt caches.
2. `state/` stores one mutable generated-prefix frontier and its nonsemantic
   scratch/cache buffers.
3. `mask/` implements the paper's **Mask** operation: read the frontier, walk
   the Parser DWA, combine encountered weights, and materialize a vocabulary
   bitset.
4. `commit/` implements the paper's **Commit** operation: consume bytes,
   enumerate completed terminal boundaries, and replace the frontier with the
   parser/tokenizer successor frontier.
5. `mask_mapping.rs` contains the final internal-token-to-original-token
   materialization quotient.  It is still a single file and is the next natural
   runtime file to split after this chunk.

A runtime state has semantic fields and nonsemantic fields.  The semantic field
is the map from tokenizer state ids to parser GSS frontiers.  Generation counts,
mask caches, and scratch buffers are performance machinery and may be discarded
without changing the accepted language.
