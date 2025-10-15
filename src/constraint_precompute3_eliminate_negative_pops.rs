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

use std::collections::BTreeSet;

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

/// Helper: rebuild the subgraph reachable from `roots` to represent exactly the
/// given set of stacks, as a prefix-trie. This clears all existing children of
/// reachable nodes, then for each root and each stack creates a fresh chain of
/// nodes and edges. Edges use a sample EV (if any is found). New nodes clone the
/// parent's T value (tests do not inspect node values).
fn rebuild_graph_from_stacks<EK, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    stacks: &BTreeSet<Vec<EK>>,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
{
    // Gather all reachable nodes once.
    let reachable = Trie::<EK, EV, T>::all_nodes(god, roots);

    // Find a sample EV if any edge exists (to populate new edges).
    let mut sample_ev: Option<EV> = None;
    'outer: for idx in &reachable {
        if let Some(g) = idx.read(god) {
            for dest_map in g.children().values() {
                for (_child, ev) in dest_map.iter() {
                    sample_ev = Some(ev.clone());
                    break 'outer;
                }
            }
        }
    }

    // Clear all children of reachable nodes (we will rebuild).
    for idx in &reachable {
        if let Some(mut w) = idx.write(god) {
            w.children_mut().clear();
        }
    }

    // If there are no edges to add (all stacks are empty), we just leave roots as leaves.
    // If there are non-empty stacks but no sample_ev could be found, that implies the
    // original graph had no edges; however, then stacks cannot be non-empty. So we
    // proceed without special handling.

    // Build the new edges as a prefix-trie under each root.
    for &root in roots {
        for stack in stacks {
            if stack.is_empty() {
                continue;
            }

            // Require a sample EV if we need to add edges.
            let ev_template = match &sample_ev {
                Some(e) => e.clone(),
                None => {
                    // No existing EV anywhere; since `stack` is non-empty this would require
                    // constructing an EV out of thin air, which we cannot. Given how the stacks
                    // are formed (from existing paths), this branch should be unreachable.
                    // Gracefully skip adding such edges.
                    continue;
                }
            };

            let mut cur = root;
            for ek in stack {
                // Clone parent node value for the new node.
                let parent_val = {
                    let g = cur.read(god).expect("Arena read while cloning parent value");
                    g.value.clone()
                };
                let new_idx = Trie2Index::new(god.insert(Trie::new(parent_val)));

                // Insert the edge with the sample EV.
                {
                    let mut ev_opt = Some(ev_template.clone());
                    let mut w = cur.write(god).expect("Arena write while inserting rebuilt edge");
                    w.try_insert_unchecked(ek.clone(), &mut ev_opt, new_idx);
                }

                cur = new_idx;
            }
        }
    }

    // Clean up any now-unreachable nodes, and recompute depths for good measure.
    Trie::<EK, EV, T>::gc(god, roots);
    Trie::<EK, EV, T>::recompute_all_max_depths(god, roots);
}

/// Graph-level transform: eliminate internal negative pops by pairwise cancellation
/// of negative/positive runs (negative on the left, positive on the right), using
/// the provided closures to read/modify pop and test check compatibility.
///
/// Not implemented yet.
pub fn eliminate_internal_negative_pops_on_trie<EK, EV, T, FGet, FReplace, FIntersect>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    get_pop: &mut FGet,
    replace_pop: &mut FReplace,
    intersect_checks: &mut FIntersect,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FReplace: FnMut(&EK, isize) -> EK,
    FIntersect: FnMut(&EK, &EK) -> bool,
{
    // 1) Extract all stacks from the current trie.
    let initial_stacks = Trie::<EK, EV, T>::get_all_paths(god, roots);

    // 2) Apply the stack-level internal elimination per path.
    let mut reduced: BTreeSet<Vec<EK>> = BTreeSet::new();
    for s in initial_stacks {
        if let Some(mid) =
            stack_eliminate_internal_negative_pops(s, &mut *get_pop, &mut *replace_pop, &mut *intersect_checks)
        {
            reduced.insert(mid);
        }
    }

    // 3) Rebuild the trie subgraph from the reduced stacks.
    rebuild_graph_from_stacks(god, roots, &reduced);
}

/// Graph-level transform: remove trailing negative pops at the ends of stacks.
/// Not implemented yet.
pub fn eliminate_trailing_negative_pops_on_trie<EK, EV, T, FGet>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    get_pop: &mut FGet,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
{
    // 1) Extract all stacks from the current trie.
    let current_stacks = Trie::<EK, EV, T>::get_all_paths(god, roots);

    // 2) Apply the stack-level trailing negative elimination per path.
    let mut trimmed: BTreeSet<Vec<EK>> = BTreeSet::new();
    for s in current_stacks {
        let fin = stack_eliminate_trailing_negative_pops(s, &mut *get_pop);
        trimmed.insert(fin);
    }

    // 3) Rebuild the trie subgraph from the trimmed stacks.
    rebuild_graph_from_stacks(god, roots, &trimmed);
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
    use crate::datastructures::trie::{Trie, Trie2Index};
    use crate::trie_test_framework::harness;
    use std::collections::{BTreeMap, BTreeSet};

    // Test harness types
    type TestEV = ();
    type TestT = ();
    type TestGod = GodWrapper<TestEK, TestEV, TestT>;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct TestEK {
        pop: isize,
        // None == "unconstrained" (universal)
        check: Option<BTreeSet<usize>>,
    }

    impl TestEK {
        fn new(pop: isize, check_ids: Option<&[usize]>) -> Self {
            let check = check_ids.map(|ids| ids.iter().cloned().collect());
            Self { pop, check }
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
    /// - None is "universal": None ∩ X = X ⇒ OK
    /// - Some(set) ∩ Some(other_set) must have a non-empty intersection.
    fn checks_intersect(a: &TestEK, b: &TestEK) -> bool {
        match (&a.check, &b.check) {
            (None, _) => true,
            (_, None) => true,
            (Some(s1), Some(s2)) => s1.iter().any(|x| s2.contains(x)),
        }
    }

    // Convenience to build an EK
    fn ek(pop: isize, ids: Option<&[usize]>) -> TestEK {
        TestEK::new(pop, ids)
    }

    // -- Unit tests for stack elimination behavior --

    #[test]
    fn run_pair_full_cancel_with_remainder_positive() {
        // -1 b, -1 c, +1 c, +1 b, +1 a  =>  +1 a
        let input = vec![
            ek(-1, Some(&[1])), // b
            ek(-1, Some(&[2])), // c
            ek(1, Some(&[2])),  // c
            ek(1, Some(&[1])),  // b
            ek(1, Some(&[0])),  // a
        ];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(1, Some(&[0]))]);
    }

    #[test]
    fn run_pair_partial_cancel_neg_leftover() {
        // -1 b, -1 c, +1 c  =>  -1 b (leftover; to be trimmed by trailing stage if at end)
        let input = vec![ek(-1, Some(&[1])), ek(-1, Some(&[2])), ek(1, Some(&[2]))];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(-1, Some(&[1]))]);
    }

    #[test]
    fn run_pair_full_cancel_empty() {
        // -1 c, +1 c  =>  []
        let input = vec![ek(-1, Some(&[2])), ek(1, Some(&[2]))];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert!(got.is_empty());
    }

    #[test]
    fn run_pair_mismatch_eliminates_stack() {
        // -1 b, +1 c  => mismatch => None
        let input = vec![ek(-1, Some(&[1])), ek(1, Some(&[2]))];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect);
        assert!(got.is_none());
    }

    #[test]
    fn run_pair_partial_cancel_neg_aggregates() {
        // -3 b, +1 c  =>  -2 b
        let input = vec![ek(-3, Some(&[1])), ek(1, Some(&[2]))];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(-2, Some(&[1]))]);
    }

    #[test]
    fn run_pair_multi_neg_partial_cancel() {
        // -1 a, -3 b, +1 c  =>  -1 a, -2 b
        let input = vec![ek(-1, Some(&[0])), ek(-3, Some(&[1])), ek(1, Some(&[2]))];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(-1, Some(&[0])), ek(-2, Some(&[1]))]);
    }

    #[test]
    fn run_pair_gap_positive_ignores_missing_slots() {
        // -1 a, -1 b, -1 c, +2 b  =>  -1 a
        let input = vec![ek(-1, Some(&[0])), ek(-1, Some(&[1])), ek(-1, Some(&[2])), ek(2, Some(&[1]))];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(-1, Some(&[0]))]);
    }

    #[test]
    fn run_pair_gap_negative_ignores_missing_slots() {
        // -2 b, +1 c, +1 b, +1 a  =>  +1 a
        let input = vec![ek(-2, Some(&[1])), ek(1, Some(&[2])), ek(1, Some(&[1])), ek(1, Some(&[0]))];
        let got = stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        assert_eq!(got, vec![ek(1, Some(&[0]))]);
    }

    #[test]
    fn trailing_negative_pops_are_removed() {
        // After internal elimination, drop trailing negatives (and zeros anywhere).
        let input = vec![ek(1, Some(&[0])), ek(-1, Some(&[1])), ek(0, None)];
        let trimmed = stack_eliminate_trailing_negative_pops(input, get_pop);
        assert_eq!(trimmed, vec![ek(1, Some(&[0]))]);
    }

    #[test]
    fn positive_prefix_without_negative_run_is_preserved() {
        // Starts with positives; internal elimination should leave them alone.
        let input = vec![ek(1, Some(&[1])), ek(2, Some(&[2])), ek(-1, Some(&[3]))];
        let mid = stack_eliminate_internal_negative_pops(input.clone(), get_pop, replace_pop, checks_intersect)
            .expect("should not mismatch");
        // Only a positive prefix followed by a trailing negative; no internal pair to cancel.
        assert_eq!(mid, input);
        // Trailing negative drops
        let final_s = stack_eliminate_trailing_negative_pops(mid, get_pop);
        assert_eq!(final_s, vec![ek(1, Some(&[1])), ek(2, Some(&[2]))]);
    }

    #[test]
    fn complex_stack_with_multiple_run_pairs_and_mismatch() {
        // +1 a, [-2 b, +1 b], [-1 c, +2 c], +1 d
        // The first pair [-2 b, +1 b] should leave [-1 b].
        // This is combined with the next negative run, giving [-1 b, -1 c].
        // This is paired with [+2 c, +1 d].
        // Reversed neg: [+1 c, +1 b]. Pos: [+2 c, +1 d].
        // At stack depth 2, we compare b and c. Their checks ([1] and [2]) do not
        // intersect, so this is a mismatch, and the whole stack is eliminated.
        let input = vec![
            ek(1, Some(&[0])),  // a
            ek(-2, Some(&[1])), // b
            ek(1, Some(&[1])),  // b
            ek(-1, Some(&[2])), // c
            ek(2, Some(&[2])),  // c
            ek(1, Some(&[3])),  // d
        ];
        let got =
            stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect);
        assert!(got.is_none(), "Expected mismatch between b and c checks");
    }

    #[test]
    fn complex_stack_with_multiple_run_pairs_compatible() {
        // Same as above, but checks for b and c are compatible.
        // `b` is [1, 5], `c` is [2, 5]. They intersect on 5.
        // First pair: [-2 b, +1 b] -> [-1 b]
        // Next negative run: [-1 b, -1 c]
        // Next positive run: [+2 c, +1 d]
        // Pair: [-1 b, -1 c] vs [+2 c, +1 d].
        // Reversed neg: [+1 c, +1 b]. Total pop 2.
        // Pos: [+2 c, +1 d]. Total pop 3.
        // Cancel amount is 2.
        // Check positions: neg_rev has c at 1, b at 2. pos has c at 2.
        // At position 2, b and c checks must intersect. They do.
        // Leftover pos: [+1 d].
        // Final stack: [+1 a, +1 d].
        let input = vec![
            ek(1, Some(&[0])),     // a
            ek(-2, Some(&[1, 5])), // b
            ek(1, Some(&[1, 5])), // b
            ek(-1, Some(&[2, 5])), // c
            ek(2, Some(&[2, 5])), // c
            ek(1, Some(&[3])),     // d
        ];
        let got =
            stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
                .expect("should not mismatch");
        assert_eq!(got, vec![ek(1, Some(&[0])), ek(1, Some(&[3]))]);
    }

    #[test]
    fn even_more_complex_stack_scenario() {
        // This test includes a leading positive, multiple neg/pos pairs, and a trailing negative.
        // Stack: [+1 a], [-3 b], [+2 b, +1 c], [-2 d], [+3 d], [-1 e]
        //
        // Trace:
        // 1. `+1 a` is emitted. `out` = `[+1 a]`.
        // 2. `process_pair([-3 b], [+2 b, +1 c])` is called.
        //    - Sums are both 3, cancel_amt is 3. Both runs are fully consumed.
        // 3. `process_pair([-2 d], [+3 d])` is called.
        //    - Sums are 2 and 3, cancel_amt is 2. Leftover is `[+1 d]`.
        //    - `out` becomes `[+1 a, +1 d]`.
        // 4. Trailing `[-1 e]` is appended.
        // Final result: `[+1 a, +1 d, -1 e]`.
        let input = vec![
            ek(1, Some(&[0])),      // a
            ek(-3, Some(&[1, 99])), // b
            ek(2, Some(&[1, 99])), // b
            ek(1, Some(&[2, 99])), // c
            ek(-2, Some(&[3])),     // d
            ek(3, Some(&[3])),     // d
            ek(-1, Some(&[4])),     // e
        ];
        let got =
            stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
                .expect("should not mismatch");
        assert_eq!(
            got,
            vec![
                ek(1, Some(&[0])),  // a
                ek(1, Some(&[3])),  // d
                ek(-1, Some(&[4]))  // e
            ]
        );
    }

    #[test]
    fn ambitious_stack_cancellation_with_mismatch() {
        // A long stack with multiple interacting runs, designed to fail on a mismatch.
        // Structure: [+lead], [-neg1, +pos1], [-neg2, +pos2], [-neg3, +pos3], [-trail]
        // Trace:
        // 1. Leading positive `[+2 a, +1 b]` is emitted.
        // 2. Pair 1: `[-3 c, -1 d]` vs `[+2 c]`. Leftover neg: `[-1 c, -1 d]`.
        // 3. This leftover merges with next neg `-2 e`, forming `[-1 c, -1 d, -2 e]`.
        // 4. Pair 2: `[-1 c, -1 d, -2 e]` vs `[+5 e, +1 f]`. Leftover pos: `[+1 e, +1 f]`.
        // 5. Pair 3: `[-1 g, -1 h]` vs `[+2 h]`.
        //    - `neg_rev` is `[+1 h, +1 g]`. `pos` is `[+2 h]`.
        //    - `neg_map` has `g` at position 2. `pos_map` has `h` at position 2.
        //    - Checks for `g` ([6]) and `h` ([7]) do not intersect.
        //    - This is a mismatch, so the entire stack is eliminated.
        let input = vec![
            ek(2, Some(&[0])), // a
            ek(1, Some(&[1])), // b
            // Pair 1
            ek(-3, Some(&[2])), // c
            ek(-1, Some(&[3])), // d
            ek(2, Some(&[2])),  // c
            // Pair 2 (neg part merges with leftover from Pair 1)
            ek(-2, None),       // e (unconstrained)
            ek(5, Some(&[4])),  // e
            ek(1, Some(&[5])),  // f
            // Pair 3 (mismatch)
            ek(-1, Some(&[6])), // g
            ek(-1, Some(&[7])), // h
            ek(2, Some(&[7])),  // h
            // Trailing
            ek(-4, Some(&[8])), // i
            ek(-1, Some(&[9])), // j
        ];
        let got =
            stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect);
        assert!(got.is_none(), "Expected mismatch between g and h");
    }

    #[test]
    fn ambitious_stack_cancellation_success() {
        // Same as the mismatch test, but with compatible checks for the final pair.
        // Trace:
        // 1. Leading positive `[+2 a, +1 b]` is emitted.
        // 2. Pair 1: `[-3 c, -1 d]` vs `[+2 c]`. Leftover neg: `[-1 c, -1 d]`.
        // 3. Merged neg run: `[-1 c, -1 d, -2 e]`.
        // 4. Pair 2: `[-1 c, -1 d, -2 e]` vs `[+5 e, +1 f]`. Leftover pos: `[+1 e, +1 f]`.
        //    `out` is now `[+2 a, +1 b, +1 e, +1 f]`.
        // 5. Pair 3: `[-1 g, -1 h]` vs `[+2 h]`. Checks are compatible. Cancels perfectly.
        // 6. Trailing negatives `[-4 i, -1 j]` are appended.
        // Final result before trailing elimination: `[+2 a, +1 b, +1 e, +1 f, -4 i, -1 j]`
        let input = vec![
            ek(2, Some(&[0])), // a
            ek(1, Some(&[1])), // b
            // Pair 1
            ek(-3, Some(&[2])), // c
            ek(-1, Some(&[3])), // d
            ek(2, Some(&[2])),  // c
            // Pair 2
            ek(-2, None),      // e (unconstrained)
            ek(5, Some(&[4])), // e
            ek(1, Some(&[5])), // f
            // Pair 3 (compatible)
            ek(-1, Some(&[6, 100])), // g
            ek(-1, Some(&[7, 100])), // h
            ek(2, Some(&[7, 100])),  // h
            // Trailing
            ek(-4, Some(&[8])), // i
            ek(-1, Some(&[9])), // j
        ];
        let got =
            stack_eliminate_internal_negative_pops(input, get_pop, replace_pop, checks_intersect)
                .expect("should not mismatch");
        assert_eq!(
            got,
            vec![
                ek(2, Some(&[0])), // a
                ek(1, Some(&[1])), // b
                ek(1, Some(&[4])), // e
                ek(1, Some(&[5])), // f
                ek(-4, Some(&[8])), // i
                ek(-1, Some(&[9])), // j
            ]
        );
    }

    // --- Graph-level scenario (stack-only validation) ---

    fn new_node(god: &TestGod) -> Trie2Index {
        harness::new_node(god, ())
    }

    fn add_edge(god: &TestGod, from: Trie2Index, to: Trie2Index, key: TestEK) {
        harness::add_edge(god, from, to, key, ());
    }

    /// Test runner that compares the trie-level transformation against the stack-based
    /// reference implementation.
    /// This is designed to fail with a `todo!()` panic until the trie-level functions
    /// are implemented, confirming that the test harness is correctly wired.
    fn run_trie_vs_stack_comparison_test(god: &TestGod, roots: &[Trie2Index]) {
        // 1. Calculate EXPECTED stacks from the original trie using the stack-based reference functions.
        let initial_stacks = Trie::<TestEK, TestEV, TestT>::get_all_paths(god, roots);
        let mut expected_stacks = BTreeSet::new();
        for s in initial_stacks {
            if let Some(mid) =
                stack_eliminate_internal_negative_pops(s, get_pop, replace_pop, checks_intersect)
            {
                let fin = stack_eliminate_trailing_negative_pops(mid, get_pop);
                expected_stacks.insert(fin);
            }
        }

        // 2. Calculate ACTUAL stacks by running the trie-level transform on a clone.
        let (god_clone, roots_clone, _) = Trie::deep_copy_subtrees(god, roots);

        // This is the call that is expected to panic until implemented.
        eliminate_negative_pops(
            &god_clone,
            &roots_clone,
            get_pop,
            replace_pop,
            checks_intersect,
        );

        let actual_stacks = Trie::get_all_paths(&god_clone, &roots_clone);

        // 3. Compare the results.
        assert_eq!(expected_stacks, actual_stacks);
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
        add_edge(&god, n16, n19, ek(0, None)); // Note: zero-pop edges are filtered by the pipeline
        add_edge(&god, n19, n20, ek(0, None)); // but are kept here to model the graph structure.

        // Path through n21 (with negative pop)
        add_edge(&god, n20, n21, ek(1, Some(&[0]))); // Using pop=1 to avoid being filtered
        add_edge(&god, n21, n23, ek(1, None));
        add_edge(&god, n23, n25, ek(-1, Some(&[1])));
        add_edge(&god, n25, n27, ek(1, None));
        add_edge(&god, n27, n28, ek(1, None));
        add_edge(&god, n28, n18, ek(1, None));

        // Path through n22
        add_edge(&god, n20, n22, ek(1, Some(&[2])));
        add_edge(&god, n22, n24, ek(2, None));
        add_edge(&god, n24, n26, ek(1, Some(&[0]))); // Leaf

        // Branch 2 (from root -> n29)
        add_edge(&god, n16, n29, ek(1, None));
        add_edge(&god, n29, n30, ek(1, None));

        // Path through n31 (with negative pop)
        add_edge(&god, n30, n31, ek(1, Some(&[1])));
        add_edge(&god, n31, n33, ek(1, None));
        add_edge(&god, n33, n35, ek(-1, Some(&[2])));
        add_edge(&god, n35, n37, ek(1, None));
        add_edge(&god, n37, n38, ek(1, None));
        add_edge(&god, n38, n18, ek(1, None));

        // Path through n32
        add_edge(&god, n30, n32, ek(1, Some(&[2])));
        add_edge(&god, n32, n34, ek(2, None));
        add_edge(&god, n34, n36, ek(1, Some(&[0]))); // Leaf

        run_trie_vs_stack_comparison_test(&god, &[n16]);
    }

    #[test]
    fn test_simple_cancel_via_runner() {
        // A --(+1, c0)--> B --(-1, c0)--> C
        // Path A->B->C should be eliminated by cancellation. Path A->D should survive.
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let d = new_node(&god);
        add_edge(&god, a, b, ek(1, Some(&[0])));
        add_edge(&god, b, c, ek(-1, Some(&[0])));
        add_edge(&god, a, d, ek(1, None)); // Dummy survivor
        run_trie_vs_stack_comparison_test(&god, &[a]);
    }

    #[test]
    fn test_mismatch_eliminates_one_path_via_runner() {
        // Path 1: A --(+1, c0)--> B --(-1, c1)--> C  (mismatch, should be eliminated)
        // Path 2: A --(+2, c2)--> D                  (should survive)
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let d = new_node(&god);
        add_edge(&god, a, b, ek(1, Some(&[0])));
        add_edge(&god, b, c, ek(-1, Some(&[1]))); // Mismatching check
        add_edge(&god, a, d, ek(2, Some(&[2])));   // Survivor
        run_trie_vs_stack_comparison_test(&god, &[a]);
    }

    #[test]
    fn test_no_negative_pops_is_noop_via_runner() {
        // A --(+2, c0)--> B --(+1, c1)--> C
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        add_edge(&god, a, b, ek(2, Some(&[0])));
        add_edge(&god, b, c, ek(1, Some(&[1])));
        run_trie_vs_stack_comparison_test(&god, &[a]);
    }

    #[test]
    fn test_trailing_negative_is_removed_via_runner() {
        // Path 1: A --(+2)--> B --(-1)--> C. The trailing negative should be removed.
        // Path 2: A --(+5)--> D. Should survive untouched.
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let d = new_node(&god);
        // Path 1
        add_edge(&god, a, b, ek(2, None));
        add_edge(&god, b, c, ek(-1, None));
        // Path 2
        add_edge(&god, a, d, ek(5, None));
        run_trie_vs_stack_comparison_test(&god, &[a]);
    }

    #[test]
    fn test_partial_cancel_on_graph_via_runner() {
        // Path 1: A --(-2, c0)--> B --(+3, c0)--> C. Should result in a path with (+1, c0).
        // Path 2: A --(+1, c1)--> D. Survivor.
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let d = new_node(&god);
        // Path 1
        add_edge(&god, a, b, ek(-2, Some(&[0])));
        add_edge(&god, b, c, ek(3, Some(&[0])));
        // Path 2
        add_edge(&god, a, d, ek(1, Some(&[1])));
        run_trie_vs_stack_comparison_test(&god, &[a]);
    }

    #[test]
    fn test_shared_node_with_divergent_outcomes_via_runner() {
        // Path 1: A --(+1, c0)--> B --(+1, c0)--> C. Should survive.
        // Path 2: D --(-1, c1)--> B --(+1, c0)--> C. Should be eliminated by mismatch.
        // Both paths share the B->C edge.
        let god = TestGod::new();
        let a = new_node(&god);
        let d = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);

        // Path 1 prefix
        add_edge(&god, a, b, ek(1, Some(&[0])));
        // Path 2 prefix
        add_edge(&god, d, b, ek(-1, Some(&[1]))); // Mismatching check ID
        // Shared suffix
        add_edge(&god, b, c, ek(1, Some(&[0])));

        run_trie_vs_stack_comparison_test(&god, &[a, d]);
    }

    #[test]
    fn test_unconstrained_check_cancels_correctly_via_runner() {
        // Path 1: A --(-1, None)--> B --(+1, c0)--> C. Should cancel to empty.
        // Path 2: A --(+1, c0)--> D. Survivor.
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let d = new_node(&god);

        // Path 1
        add_edge(&god, a, b, ek(-1, None));
        add_edge(&god, b, c, ek(1, Some(&[0])));
        // Path 2
        add_edge(&god, a, d, ek(1, Some(&[0])));

        run_trie_vs_stack_comparison_test(&god, &[a]);
    }

    #[test]
    fn test_all_paths_eliminated_via_runner() {
        // Path 1: A --(+1, c0)--> B --(-1, c1)--> C (mismatch)
        // Path 2: A --(-2, c2)--> D --(+2, c2)--> E (full cancel)
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let d = new_node(&god);
        let e = new_node(&god);

        // Path 1
        add_edge(&god, a, b, ek(1, Some(&[0])));
        add_edge(&god, b, c, ek(-1, Some(&[1]))); // Mismatch

        // Path 2
        add_edge(&god, a, d, ek(-2, Some(&[2])));
        add_edge(&god, d, e, ek(2, Some(&[2]))); // Cancels

        run_trie_vs_stack_comparison_test(&god, &[a]);
    }
}

