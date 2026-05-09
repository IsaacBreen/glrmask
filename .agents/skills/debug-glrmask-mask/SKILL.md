---
name: debug-glrmask-mask
description: Trace glrmask2 mask/commit mismatches and false positive or false negative masks to their root cause. Use when commit_bytes or commit_token disagrees with mask(), when GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE fails, or when a token is accepted or rejected incorrectly and the answer likely depends on tokenizer states, possible_matches, internal token classes, terminal DWA, parser DWA, L1/L2P partitioning, or constraint-vocab remapping.
---

# Debug GLRMask Mask

## Goal

Find the first layer where `mask()`, `commit_token`, and `commit_bytes` diverge. Do not add post-filters or relax assertions to hide a mismatch.

## Workflow

1. Establish the oracle on the same prefix/token. Treat `commit_bytes` as the byte-level parser oracle until proven otherwise.
2. Before proposing any grammar, importer, or lowering change, classify the failure:
   - If `commit_bytes(token_bytes)` accepts but `mask()` or `commit_token(token_id)` rejects, the matched language is already correct. The bug is in mask/token construction, parser-DWA, terminal-DWA/id-map, possible-matches, runtime mask expansion, or token remapping. Do not "fix" it by changing grammar shape, terminal grouping, factoring, chunking, or lowering.
   - If `commit_bytes(token_bytes)` rejects and the token should be legal, then investigate source/importer/grammar semantics. Only then is a grammar or importer edit a candidate fix.
3. Reproduce in release mode and tee long output to `/tmp/*.log`.
4. Localize with one toggle at a time:
   - `GLRMASK_FORCE_ALL_L2P=1`: keep partitioning but route all terminals through L2P.
   - `GLRMASK_PM_BRUTE_FORCE=1`: replace optimized constraint possible-matches.
   - `GLRMASK_PM_TRIE_CLASS_BUILD=0`: bypass trie-class possible-match grouping.
5. If possible-matches toggles do not affect the oracle, inspect parser-DWA weights, terminal-DWA weights, id-map remapping, and runtime mask expansion in that order.
6. If partition toggles change the oracle, inspect `src/compiler/stages/id_map_and_terminal_dwa/{partition,merge,l1,l2p}.rs` and compare original ids, internal ids, tokenizer-state ids, and DWA weights before/after merge.

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

Symptom-suppressing changes are not root causes. If a bug disappears after a
local rewrite, alternate lowering, feature toggle, disabled optimization,
different data shape, extra wrapper, changed recursion direction, broader
grammar, narrower vocab, or other behavior-preserving transformation, treat that
as a workaround until you prove why the original failing path broke. The
investigation is not complete until it names the first artifact where the
expected invariant stops holding: source input, normalized representation,
flattened CFG, nullability/FIRST data, LR/GLR item set, parse table action,
parser-DWA weight, terminal-DWA/id-map output, possible-matches, or runtime mask
expansion.

The mask must be a function of the language accepted from the current parser
state, not of incidental grammar presentation. A change that only alters parse
structure, terminal boundaries, helper nonterminals, factoring, chunk sizes, or
where bytes are fused into terminals is not a valid fix for a mask mismatch
unless the `commit_bytes` oracle proves the old structure accepted the wrong
language. If equivalent-language grammars produce different masks, fix the
compiler/runtime layer that made the mask structure-dependent.

Before accepting a fix for a parser/mask mismatch:

- Preserve or add a regression that fails on the original bad path, preferably
  with minimized input, configuration, vocab/data, prefix, and token.
- Include a same-prefix/same-token `commit_bytes` check in the regression or in
  the saved investigation log, and state whether the failure is language
  semantics or mask/token construction.
- Compare the failing path and the proposed fixed path at the first downstream
  artifact where they diverge.
- Explain why the proposed fix restores the violated invariant. Do not report
  success merely because a transformed input or alternate code path makes the
  symptom pass.
- If the fix intentionally normalizes away a problematic shape, state the
  remaining footgun explicitly and either fix the lower layer or add coverage
  that prevents the bad path from re-entering through public inputs.
