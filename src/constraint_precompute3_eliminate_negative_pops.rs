// src/constraint_precompute3_eliminate_negative_pops.rs
//
// New design: Negative-pop elimination that respects ordering and "stack can change"
// between operations. We do not reorder operations globally. Instead, we cancel
// negative/positive run pairs locally, using set-intersection semantics for checks.
//
// This file provides:
// - Public trie-level APIs (stubs/todo!()) to be wired to graph rewriting.
//
// - Fully implemented reference, stack-only algorithms:
//    * stack_eliminate_internal_negative_pops: cancels internal negative/positive run pairs
//      in a single stack, respecting ordering and using a user-supplied intersection
//      predicate to detect mismatches (which eliminate the whole stack).
//    * stack_eliminate_trailing_negative_pops: removes trailing negative pops (and zero pops).
//
// The stack reference functions are intended for testing and validation and will guide the
// future trie-level implementation.
//
// Notes on check semantics:
// - We expect callers to supply an `intersect` closure that returns true iff the "checks"
//   on two positions are compatible. In tests, we model a check as a BTreeSet<usize> where
//   an empty set denotes "unconstrained" (i.e., a universal set). The intersection then
//   uses the rule: empty ∩ X = X (non-empty) ⇒ match; non-empty ∩ non-empty empty ⇒ mismatch.

use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};

/// Perform negative-pop elimination on the trie:
/// - First eliminate internal negative pops by canceling negative/positive run pairs.
/// - Then eliminate trailing negative pops at the end of stacks.
/// This orchestrator delegates to the two trie-level transforms (currently todo!).
pub fn eliminate_negative_pops<EK, EV, T, FGet, FReplace, FIntersect>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    mut get_pop: FGet,
    mut replace_pop: FReplace,
    mut intersect_checks: FIntersect,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FReplace: FnMut(&EK, isize) -> EK,
    FIntersect: FnMut(&EK, &EK) -> bool,
{
    // Stage 1: eliminate internal negative pops (graph-level)
    eliminate_internal_negative_pops_on_trie(
        god,
        roots,
        &mut get_pop,
        &mut replace_pop,
        &mut intersect_checks,
    );

    // Stage 2: eliminate trailing negatives (graph-level)
    eliminate_trailing_negative_pops_on_trie(god, roots, &mut get_pop);
}

/// Graph-level transform: eliminate internal negative pops by pairwise cancellation
/// of negative/positive runs (negative on the left, positive on the right), using
/// the provided closures to read/modify pop and test check compatibility.
///
/// Not implemented yet.
pub fn eliminate_internal_negative_pops_on_trie<EK, EV, T, FGet, FReplace, FIntersect>(
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    _get_pop: &mut FGet,
    _replace_pop: &mut FReplace,
    _intersect_checks: &mut FIntersect,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FReplace: FnMut(&EK, isize) -> EK,
    FIntersect: FnMut(&EK, &EK) -> bool,
{
    // TODO: Implement the graph-level version by scanning paths and performing local rewrites.
    // Strategy sketch:
    // - Enumerate stacks (paths) or perform local rewrites along edges while preserving
    //   branching semantics. Use on-the-fly cloning where needed to avoid mutating shared nodes.
    // - For each negative/positive run pair (local to a path segment), test compatibility via
    //   intersect_checks, and cancel pops up to the min of run totals, producing remainders.
    // - Eliminate stacks exhibiting mismatches (remove the paths/edges).
    todo!()
}

/// Graph-level transform: remove trailing negative pops at the ends of stacks.
/// Not implemented yet.
pub fn eliminate_trailing_negative_pops_on_trie<EK, EV, T, FGet>(
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    _get_pop: &mut FGet,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
{
    // TODO: Implement the graph-level version by pruning or neutralizing trailing negative
    // edges on all terminal paths. Also remove edges with pop == 0.
    todo!()
}

/// Reference stack function: eliminate internal negative pops by canceling adjacent
/// negative/positive run pairs.
/// Returns:
/// - Some(new_stack) if no mismatches were found.
/// - None if any run pair exhibited a mismatch (entire stack eliminated).
pub fn stack_eliminate_internal_negative_pops<EK, FGet, FReplace, FIntersect>(
    stack: Vec<EK>,
    mut get_pop: FGet,
    mut replace_pop: FReplace,
    mut intersect_checks: FIntersect,
) -> Option<Vec<EK>>
where
    EK: Clone,
    FGet: FnMut(&EK) -> isize,
    FReplace: FnMut(&EK, isize) -> EK,
    FIntersect: FnMut(&EK, &EK) -> bool,
{
    // Remove zero-pop items eagerly: they are no-ops and don't influence run pairing.
    let mut cleaned: Vec<EK> = Vec::with_capacity(stack.len());
    for ek in stack {
        if get_pop(&ek) != 0 {
            cleaned.push(ek);
        }
    }

    // Buffers for current negative and positive runs.
    let mut out: Vec<EK> = Vec::with_capacity(cleaned.len());
    let mut neg_buf: Vec<EK> = Vec::new();
    let mut pos_buf: Vec<EK> = Vec::new();
    let mut in_pos = false;

    // Helper: process a completed neg/pos pair, returning (leftover_neg, leftover_pos).
    fn process_pair<EK, FGet, FReplace, FIntersect>(
        mut neg_buf: Vec<EK>,
        mut pos_buf: Vec<EK>,
        get_pop: &mut FGet,
        replace_pop: &mut FReplace,
        intersect_checks: &mut FIntersect,
    ) -> Option<(Vec<EK>, Vec<EK>)>
    where
        EK: Clone,
        FGet: FnMut(&EK) -> isize,
        FReplace: FnMut(&EK, isize) -> EK,
        FIntersect: FnMut(&EK, &EK) -> bool,
    {
        // Create a positive-pop reversed copy of the negative run.
        let mut neg_rev: Vec<EK> = Vec::with_capacity(neg_buf.len());
        for ek in neg_buf.iter().rev() {
            let p = get_pop(ek);
            debug_assert!(p < 0);
            neg_rev.push(replace_pop(ek, -p));
        }

        // The positive run remains as-is (pops > 0 expected).
        // Compute realized check positions for both runs:
        // Positions are cumulative sums; we only record positions where an item carries a check.
        // Intersections are checked for overlapping positions (if any); failures eliminate the stack.
        use std::collections::BTreeMap;
        let mut neg_map: BTreeMap<usize, &EK> = BTreeMap::new();
        let mut pos_map: BTreeMap<usize, &EK> = BTreeMap::new();

        let mut cum = 0usize;
        for ek in &neg_rev {
            let p = get_pop(ek);
            debug_assert!(p > 0);
            cum += p as usize;
            // Record at boundary
            neg_map.insert(cum, ek);
        }
        cum = 0;
        for ek in &pos_buf {
            let p = get_pop(ek);
            debug_assert!(p > 0);
            cum += p as usize;
            pos_map.insert(cum, ek);
        }

        // Check pairwise compatibility via intersection on overlapping positions.
        let mut neg_it = neg_map.iter().peekable();
        let mut pos_it = pos_map.iter().peekable();
        while let (Some((npos, nek)), Some((ppos, pek))) = (neg_it.peek(), pos_it.peek()) {
            if npos == ppos {
                // Overlapping boundary: must be compatible
                let ok = intersect_checks(nek, pek);
                if !ok {
                    return None;
                }
                neg_it.next();
                pos_it.next();
            } else if npos < ppos {
                neg_it.next();
            } else {
                pos_it.next();
            }
        }

        // Determine how much we can cancel (min of the run totals).
        let sum_neg: usize = neg_rev.iter().map(|ek| get_pop(ek).max(0) as usize).sum();
        let sum_pos: usize = pos_buf.iter().map(|ek| get_pop(ek).max(0) as usize).sum();
        let mut cancel_amt = sum_neg.min(sum_pos);

        // Subtract cancel_amt from the fronts of both lists.
        fn subtract_from_front<EK, FGet, FReplace>(
            mut seq: Vec<EK>,
            mut amt: usize,
            get_pop: &mut FGet,
            replace_pop: &mut FReplace,
        ) -> Vec<EK>
        where
            EK: Clone,
            FGet: FnMut(&EK) -> isize,
            FReplace: FnMut(&EK, isize) -> EK,
        {
            if amt == 0 || seq.is_empty() {
                // No changes
                return seq;
            }
            let mut out: Vec<EK> = Vec::with_capacity(seq.len());
            for ek in seq.into_iter() {
                if amt == 0 {
                    out.push(ek);
                    continue;
                }
                let p = get_pop(&ek);
                debug_assert!(p > 0);
                let pu = p as usize;
                if pu > amt {
                    let new_p = (pu - amt) as isize;
                    out.push(replace_pop(&ek, new_p));
                    amt = 0;
                } else if pu == amt {
                    // Entire element consumed; drop it.
                    amt = 0;
                } else {
                    // Consume and drop this element entirely.
                    amt -= pu;
                }
            }
            out
        }

        let neg_rev_left = subtract_from_front(neg_rev, cancel_amt, get_pop, replace_pop);
        let pos_left = subtract_from_front(pos_buf, cancel_amt, get_pop, replace_pop);

        // Convert neg_rev_left back to original order with negative pops.
        let mut leftover_neg: Vec<EK> = Vec::with_capacity(neg_rev_left.len());
        for ek in neg_rev_left.into_iter().rev() {
            let p = get_pop(&ek);
            debug_assert!(p > 0);
            leftover_neg.push(replace_pop(&ek, -p));
        }

        Some((leftover_neg, pos_left))
    }

    for ek in cleaned.into_iter() {
        let p = get_pop(&ek);
        if p < 0 {
            // Negative element
            if !in_pos {
                neg_buf.push(ek);
            } else {
                // We just finished a positive run; process neg/pos pair.
                let pair = process_pair(neg_buf, pos_buf, &mut get_pop, &mut replace_pop, &mut intersect_checks)?;
                let (leftover_neg, leftover_pos) = pair;
                out.extend(leftover_pos);
                neg_buf = leftover_neg;
                pos_buf = Vec::new();
                in_pos = false;

                // Start next negative run with this ek
                neg_buf.push(ek);
            }
        } else if p > 0 {
            // Positive element
            if neg_buf.is_empty() && !in_pos {
                // No negative run to pair with: emit directly.
                out.push(ek);
            } else {
                in_pos = true;
                pos_buf.push(ek);
            }
        } else {
            // Zero: skip (already removed) - here for completeness.
        }
    }

    // If we ended in a positive run, resolve the tailing pair.
    if in_pos {
        let pair = process_pair(neg_buf, pos_buf, &mut get_pop, &mut replace_pop, &mut intersect_checks)?;
        let (leftover_neg, leftover_pos) = pair;
        out.extend(leftover_pos);
        neg_buf = leftover_neg;
    }

    // Append any trailing negatives; these remain to be eliminated by the trailing stage.
    out.extend(neg_buf.into_iter());

    // Remove any zeros that might have resurfaced (defensive; shouldn't happen).
    let mut final_out: Vec<EK> = Vec::with_capacity(out.len());
    for ek in out.into_iter() {
        if get_pop(&ek) != 0 {
            final_out.push(ek);
        }
    }

    Some(final_out)
}

/// Reference stack function: remove trailing negative pops and zero-pop items.
/// This does not attempt to cancel anything; it simply trims the tail where pop < 0,
/// and removes any zero-pop elements anywhere.
pub fn stack_eliminate_trailing_negative_pops<EK, FGet>(
    stack: Vec<EK>,
    mut get_pop: FGet,
) -> Vec<EK>
where
    EK: Clone,
    FGet: FnMut(&EK) -> isize,
{
    // First, remove zero-pop items anywhere.
    let mut cleaned: Vec<EK> = Vec::with_capacity(stack.len());
    for ek in stack {
        if get_pop(&ek) != 0 {
            cleaned.push(ek);
        }
    }

    // Then drop trailing negatives.
    let cut = cleaned
        .iter()
        .rposition(|ek| get_pop(ek) >= 0)
        .map(|i| i + 1)
        .unwrap_or(0);
    cleaned.truncate(cut);
    cleaned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PrecomputedNodeContents;
    use crate::datastructures::trie::{Trie, Trie2Index};
    use crate::trie_test_framework::harness;
    use std::collections::{BTreeMap, BTreeSet};

    // Test harness types
    type TestEV = ();
    type TestT = PrecomputedNodeContents;
    type TestGod = GodWrapper<TestEK, TestEV, TestT>;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct TestEK {
        pop: isize,
        // Empty set == "unconstrained" (universal)
        check: BTreeSet<usize>,
    }

    impl TestEK {
        fn new(pop: isize, check_ids: &[usize]) -> Self {
            let mut s = BTreeSet::new();
            for &id in check_ids {
                s.insert(id);
            }
            Self { pop, check: s }
        }
    }

    // Helpers for closures
    fn get_pop(ek: &TestEK) -> isize {
        ek.pop
    }

    fn replace_pop(ek: &TestEK, new_pop: isize) -> TestEK {
        TestEK {
            pop: new_pop,
            check: ek.check.clone(),
        }
    }

    /// Intersection semantics for our tests:
    /// - Empty set is "universal": empty ∩ X = X (non-empty) ⇒ OK; empty ∩ empty ⇒ OK
    /// - Non-empty ∩ Non-empty must be non-empty.
    fn checks_intersect(a: &TestEK, b: &TestEK) -> bool {
        if a.check.is_empty() || b.check.is_empty() {
            return true;
        }
        a.check.iter().any(|x| b.check.contains(x))
    }

    // Convenience to build an EK
    fn ek(pop: isize, ids: &[usize]) -> TestEK {
        TestEK::new(pop, ids)
    }

    // -- Unit tests for stack elimination behavior --

    #[test]
    fn run_pair_full_cancel_with_remainder_positive() {
        // -1 b, -1 c, +1 c, +1 b, +1 a  =>  +1 a
        let input = vec![
            ek(-1, &[1]), // b
            ek(-1, &[2]), // c
            ek(1, &[2]),  // c
            ek(1, &[1]),  // b
            ek(1, &[0]),  // a
        ];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(1, &[0])]);
    }

    #[test]
    fn run_pair_partial_cancel_neg_leftover() {
        // -1 b, -1 c, +1 c  =>  -1 b (leftover; to be trimmed by trailing stage if at end)
        let input = vec![ek(-1, &[1]), ek(-1, &[2]), ek(1, &[2])];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(-1, &[1])]);
    }

    #[test]
    fn run_pair_full_cancel_empty() {
        // -1 c, +1 c  =>  []
        let input = vec![ek(-1, &[2]), ek(1, &[2])];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert!(got.is_empty());
    }

    #[test]
    fn run_pair_mismatch_eliminates_stack() {
        // -1 b, +1 c  => mismatch => None
        let input = vec![ek(-1, &[1]), ek(1, &[2])];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect);
        assert!(got.is_none());
    }

    #[test]
    fn run_pair_partial_cancel_neg_aggregates() {
        // -3 b, +1 c  =>  -2 b
        let input = vec![ek(-3, &[1]), ek(1, &[2])];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(-2, &[1])]);
    }

    #[test]
    fn run_pair_multi_neg_partial_cancel() {
        // -1 a, -3 b, +1 c  =>  -1 a, -2 b
        let input = vec![ek(-1, &[0]), ek(-3, &[1]), ek(1, &[2])];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(-1, &[0]), ek(-2, &[1])]);
    }

    #[test]
    fn run_pair_gap_positive_ignores_missing_slots() {
        // -1 a, -1 b, -1 c, +2 b  =>  -1 a
        let input = vec![ek(-1, &[0]), ek(-1, &[1]), ek(-1, &[2]), ek(2, &[1])];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(-1, &[0])]);
    }

    #[test]
    fn run_pair_gap_negative_ignores_missing_slots() {
        // -2 b, +1 c, +1 b, +1 a  =>  +1 a
        let input = vec![ek(-2, &[1]), ek(1, &[2]), ek(1, &[1]), ek(1, &[0])];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(1, &[0])]);
    }

    #[test]
    fn trailing_negative_pops_are_removed() {
        // After internal elimination, drop trailing negatives (and zeros anywhere).
        let input = vec![ek(1, &[0]), ek(-1, &[1]), ek(0, &[])];
        let trimmed = stack_eliminate_trailing_negative_pops(input, get_pop);
        assert_eq!(trimmed, vec![ek(1, &[0])]);
    }

    #[test]
    fn positive_prefix_without_negative_run_is_preserved() {
        // Starts with positives; internal elimination should leave them alone.
        let input = vec![ek(1, &[1]), ek(2, &[2]), ek(-1, &[3])];
        let mid = stack_eliminate_internal_negative_pops(input.clone(), get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        // Only a positive prefix followed by a trailing negative; no internal pair to cancel.
        assert_eq!(mid, input);
        // Trailing negative drops
        let final_s = stack_eliminate_trailing_negative_pops(mid, get_pop);
        assert_eq!(final_s, vec![ek(1, &[1]), ek(2, &[2])]);
    }

    // --- Graph-level scenario (stack-only validation) ---

    fn new_node(god: &TestGod) -> Trie2Index {
        harness::new_node(god, PrecomputedNodeContents::internal())
    }

    fn add_edge(god: &TestGod, from: Trie2Index, to: Trie2Index, key: TestEK) {
        harness::add_edge(god, from, to, key, ());
    }

    /// Validate the stack pipeline on a complex graph by:
    /// - Extracting all stacks from the trie.
    /// - Applying the stack pipeline (internal + trailing).
    /// - Ensuring the final stacks contain no negative pops and do not mismatch.
    #[test]
    fn test_graph_from_complex_stack_trace() {
        let god = TestGod::new();
        // Nodes (labels copied from earlier example)
        let n16 = new_node(&god); // root
        let n18 = new_node(&god); // end
        let n19 = new_node(&god);
        let n20 = new_node(&god);
        let n21 = new_node(&god);
        let n22 = new_node(&god);
        let n23 = new_node(&god);
        let n24 = new_node(&god);
        let n25 = new_node(&god);
        let n26 = new_node(&god);
        let n27 = new_node(&god);
        let n28 = new_node(&god);
        let n29 = new_node(&god);
        let n30 = new_node(&god);
        let n31 = new_node(&god);
        let n32 = new_node(&god);
        let n33 = new_node(&god);
        let n34 = new_node(&god);
        let n35 = new_node(&god);
        let n36 = new_node(&god);
        let n37 = new_node(&god);
        let n38 = new_node(&god);

        // --- Build graph from the described stack trace ---

        // Branch 1 (from root -> n19)
        add_edge(&god, n16, n19, ek(0, &[]));
        add_edge(&god, n19, n20, ek(0, &[]));

        // Path through n21 (with negative pop)
        add_edge(&god, n20, n21, ek(0, &[0]));
        add_edge(&god, n21, n23, ek(0, &[]));
        add_edge(&god, n23, n25, ek(-1, &[1]));
        add_edge(&god, n25, n27, ek(0, &[]));
        add_edge(&god, n27, n28, ek(0, &[]));
        add_edge(&god, n28, n18, ek(0, &[]));

        // Path through n22
        add_edge(&god, n20, n22, ek(0, &[2]));
        add_edge(&god, n22, n24, ek(2, &[]));
        add_edge(&god, n24, n26, ek(0, &[0])); // Leaf

        // Branch 2 (from root -> n29)
        add_edge(&god, n16, n29, ek(0, &[]));
        add_edge(&god, n29, n30, ek(0, &[]));

        // Path through n31 (with negative pop)
        add_edge(&god, n30, n31, ek(0, &[1]));
        add_edge(&god, n31, n33, ek(0, &[]));
        add_edge(&god, n33, n35, ek(-1, &[2]));
        add_edge(&god, n35, n37, ek(0, &[]));
        add_edge(&god, n37, n38, ek(0, &[]));
        add_edge(&god, n38, n18, ek(0, &[]));

        // Path through n32
        add_edge(&god, n30, n32, ek(0, &[2]));
        add_edge(&god, n32, n34, ek(2, &[]));
        add_edge(&god, n34, n36, ek(0, &[0])); // Leaf

        // Extract stacks
        let stacks = Trie::<TestEK, TestEV, TestT>::get_all_paths(&god, &[n16]);

        // Apply the stack pipeline to each stack: internal cancellation + trailing elimination.
        let mut final_stacks = Vec::new();
        for s in stacks {
            match stack_eliminate_internal_negative_pops(s, get_pop, replace_pop, checks_intersect) {
                None => {
                    // Entire stack eliminated due to mismatch; skip it.
                }
                Some(mid) => {
                    let fin = stack_eliminate_trailing_negative_pops(mid, get_pop);
                    // Assert there are no negative pops in the result
                    for ek in &fin {
                        assert!(
                            ek.pop >= 0,
                            "Final stacks must not contain negative pops: {:?}",
                            fin
                        );
                    }
                    final_stacks.push(fin);
                }
            }
        }

        // For sanity, we expect at least one surviving stack and all are non-negative-only now.
        assert!(
            !final_stacks.is_empty(),
            "Expected some surviving stacks after the elimination pipeline"
        );
        for s in final_stacks {
            for ek in s {
                assert!(ek.pop >= 0);
            }
        }
    }
}
