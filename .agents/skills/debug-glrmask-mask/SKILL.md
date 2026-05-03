---
name: debug-glrmask-mask
description: Trace glrmask2 mask/commit mismatches and false positive or false negative masks to their root cause. Use when commit_bytes or commit_token disagrees with mask(), when GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE fails, or when a token is accepted or rejected incorrectly and the answer likely depends on tokenizer states, possible_matches, internal token classes, terminal DWA, parser DWA, L1/L2P partitioning, or constraint-vocab remapping.
---

# Debug GLRMask Mask

## Goal

Find the first layer where semantics diverge. Do not add post-filters around `mask()` or relax assertions to hide a mismatch.

## Workflow

1. Establish the oracle: compare `mask()`, `commit_token`, and `commit_bytes` on the same prefix and token. Treat `commit_bytes` as the byte-level parser oracle until proven otherwise.
2. Run the focused release test and capture output to `/tmp/*.log`; release mode matters for timing and FFI-adjacent work.
3. Toggle diagnostic build paths to localize the layer:
   - `GLRMASK_NO_PARTITION=1`: bypass split L1/L2P terminal-DWA construction.
   - `GLRMASK_FORCE_ALL_L2P=1`: keep partitioning but route all terminals through L2P.
   - `GLRMASK_PM_BRUTE_FORCE=1`: replace optimized constraint possible-matches.
   - `GLRMASK_PM_TRIE_CLASS_BUILD=0`: bypass trie-class possible-match grouping.
   - `GLRMASK_DEBUG_PROFILE=1 GLRMASK_DEBUG_TERMINAL_MAPPINGS=1`: print terminal ids, L1/L2P classification, partition sizes, and DWA profiles.
4. If possible-matches toggles do not affect the oracle, debug parser-DWA weights and terminal-DWA/id-map remapping before runtime mask expansion.
5. If `GLRMASK_NO_PARTITION=1` or `GLRMASK_FORCE_ALL_L2P=1` changes the oracle, inspect `src/compiler/stages/id_map_and_terminal_dwa/{partition,merge,l1,l2p}.rs`. Compare token/state maps and DWA weights before and after L1/L2P merge.
6. Keep diagnostics temporary unless they are generally useful and gated by an env var. Remove one-off prints before committing the fix.

## L1 Terminal DWA Pitfall

L1 state equivalence can be valid for whole-token walks from a start state without being valid for arbitrary suffix walks after the first byte. If the L1 builder splits tokens into first byte plus suffix, it must not walk the suffix from a merged internal representative unless the equivalence relation is proven suffix-closed. A robust pattern is: use concrete DFA target states for suffix traversal, then map the final state back to the L1 representative/TSID space.

This failure shape often looks bizarre in an MRE: duplicate vocab bytes, token ordering, and source-looking survivor strings may be load-bearing because they perturb internal token ids, state reps, and range-set profiles enough to expose a bad representative choice.

The L1 transition relation is a simple whole-token predicate: for each original tokenizer start state and LLM token, run the whole token bytes, keep terminals whose match width equals the token length, then add possible-future terminals from the token end state. Optimized L1 construction may compress IDs, but it must be equivalent to that predicate.

Do not coarsen L1 tokenizer states from a token sample. A sampled "confirmation" can merge states that agree on the sample but differ on an unsampled token/terminal pair. This produces order-sensitive false negatives where `commit_token` succeeds through the concrete tokenizer state but the parser DWA weight was built from a representative whose end-state terminal signature is missing the needed terminal. Exact bounded state equivalence is acceptable; sampled post-coarsening is not.

When debugging L1, compare the concrete full-token end-state terminal signature with the signature used in the built transition weight. If they differ, suspect state-map coarsening or representative use before suspecting runtime mask filtering.

## Root-Cause Standard

The final explanation should name:

- the exact layer that first drops or admits the token;
- the invariant violated, in terms of original ids vs internal ids, tokenizer-state ids, possible-match token space, or DWA weight semantics;
- why the weird MRE details are load-bearing;
- why the fix restores the invariant without masking the symptom.

Update this skill as new reusable diagnostics or invariants are discovered.
