---
name: debug-glrmask-mask
description: Trace glrmask2 mask/commit mismatches and false positive or false negative masks to their root cause. Use when commit_bytes or commit_token disagrees with mask(), when GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE fails, or when a token is accepted or rejected incorrectly and the answer likely depends on tokenizer states, possible_matches, internal token classes, terminal DWA, parser DWA, L1/L2P partitioning, or constraint-vocab remapping.
---

# Debug GLRMask Mask

## Goal

Find the first layer where `mask()`, `commit_token`, and `commit_bytes` diverge. Do not add post-filters or relax assertions to hide a mismatch.

## Workflow

1. Establish the oracle on the same prefix/token. Treat `commit_bytes` as the byte-level parser oracle until proven otherwise.
2. Reproduce in release mode and tee long output to `/tmp/*.log`.
3. Localize with one toggle at a time:
   - `GLRMASK_FORCE_ALL_L2P=1`: keep partitioning but route all terminals through L2P.
   - `GLRMASK_PM_BRUTE_FORCE=1`: replace optimized constraint possible-matches.
   - `GLRMASK_PM_TRIE_CLASS_BUILD=0`: bypass trie-class possible-match grouping.
4. If possible-matches toggles do not affect the oracle, inspect parser-DWA weights, terminal-DWA weights, id-map remapping, and runtime mask expansion in that order.
5. If partition toggles change the oracle, inspect `src/compiler/stages/id_map_and_terminal_dwa/{partition,merge,l1,l2p}.rs` and compare original ids, internal ids, tokenizer-state ids, and DWA weights before/after merge.

## High-Value Pitfalls

### L1 Whole-Token Semantics

For each original tokenizer start state and LLM token, L1 must run the whole token bytes, keep only terminal matches whose width equals the token length, then add possible-future terminals from the concrete token end state. Optimized construction may compress ids, but the transition weight must be equivalent to this predicate.

### L1 State Equivalence

Do not coarsen L1 tokenizer states from samples. Merge states only when their exact whole-vocab terminal-signature profiles are equal. Hashes can bucket candidates, but exact equality must prove every final merge. See `notes/2026-05-03-l1-sampled-equivalence-bug.md` for the o1052 failure history.

### Representative Suffix Walks

Whole-token equivalence from a start state is not necessarily suffix-closed. If L1 splits a token into first byte plus suffix, walk suffix bytes from the concrete DFA target state, then map only the final result back into compressed TSID/id spaces.

### Weird MRE Sensitivity

If duplicate vocab bytes, token ordering, or irrelevant-looking string edits are load-bearing, suspect representative choice, id maps, range-set profiles, or cross-partition remapping before suspecting the literal token text.

## Root-Cause Standard

Name the exact layer that first drops or admits the token, the invariant violated, why odd MRE details are load-bearing if relevant, and why the fix restores the invariant rather than masking the symptom.
