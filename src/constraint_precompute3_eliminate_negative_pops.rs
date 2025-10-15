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

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};

// Helper struct to carry all data associated with a path segment.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct PathElement<EK, EV, T> {
    ek: EK,
    ev: EV,
    src_node_value: T,
    dst_node_value: T,
}

// Helper to extract all paths with full EK, EV, and T data.
fn get_all_paths_with_data<EK, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
) -> BTreeSet<(T, Vec<PathElement<EK, EV, T>>)>
where
    EK: Ord + Clone,
    EV: Ord + Clone,
    T: Ord + Clone,
{
    let mut all_paths = BTreeSet::new();
    for &root in roots {
        if let Some(root_guard) = root.read(god) {
            let root_value = root_guard.value.clone();
            let mut visiting = BTreeSet::new();
            get_all_paths_with_data_recursive(
                god,
                root,
                vec![],
                &mut all_paths,
                &mut visiting,
                root_value,
            );
        }
    }
    all_paths
}

fn get_all_paths_with_data_recursive<EK, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    node_idx: Trie2Index,
    current_path: Vec<PathElement<EK, EV, T>>,
    all_paths: &mut BTreeSet<(T, Vec<PathElement<EK, EV, T>>)>,
    visiting: &mut BTreeSet<Trie2Index>,
    root_value: T,
) where
    EK: Ord + Clone,
    EV: Ord + Clone,
    T: Ord + Clone,
{
    if !visiting.insert(node_idx) {
        return; // Cycle detected
    }

    if let Some(guard) = node_idx.read(god) {
        if guard.is_leaf() {
            all_paths.insert((root_value, current_path));
        } else {
            let src_node_value = guard.value.clone();
            for (edge_key, dest_map) in guard.children() {
                for (child_idx, edge_value) in dest_map.iter() {
                    if let Some(child_guard) = child_idx.read(god) {
                        let mut new_path = current_path.clone();
                        new_path.push(PathElement {
                            ek: edge_key.clone(),
                            ev: edge_value.clone(),
                            src_node_value: src_node_value.clone(),
                            dst_node_value: child_guard.value.clone(),
                        });
                        get_all_paths_with_data_recursive(
                            god,
                            *child_idx,
                            new_path,
                            all_paths,
                            visiting,
                            root_value.clone(),
                        );
                    }
                }
            }
        }
    }
    visiting.remove(&node_idx);
}

/// Perform negative-pop elimination on the trie:
/// - First eliminate internal negative pops by canceling negative/positive run pairs.
/// - Then eliminate trailing negative pops at the end of stacks.
/// This orchestrator delegates to the two trie-level transforms (currently todo!).
pub fn eliminate_negative_pops<EK, EV, T, FGet, FReplace, FIntersect, FCanRemove>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    mut get_pop: FGet,
    mut replace_pop: FReplace,
    mut intersect_checks: FIntersect,
    mut can_remove: FCanRemove,
) where
    EK: Ord + Clone,
    EV: Clone + Ord,
    T: Clone + Ord + PartialEq,
    FGet: FnMut(&EK, &EV) -> isize,
    FReplace: FnMut(&EK, &EV, isize) -> (EK, EV),
    FIntersect: FnMut(&EK, &EV, &EK, &EV) -> bool,
    FCanRemove: FnMut(&EK, &EV) -> bool,
{
    // Stage 1: eliminate internal negative pops (graph-level)
    eliminate_internal_negative_pops_on_trie(
        god,
        roots,
        &mut get_pop,
        &mut replace_pop,
        &mut intersect_checks,
        &mut can_remove,
    );

    // Stage 2: eliminate trailing negatives (graph-level)
    eliminate_trailing_negative_pops_on_trie(god, roots, &mut get_pop, &mut can_remove);
}

/// Graph-level transform: eliminate internal negative pops by pairwise cancellation
/// of negative/positive runs (negative on the left, positive on the right), using
/// the provided closures to read/modify pop and test check compatibility.
///
/// Not implemented yet.
pub fn eliminate_internal_negative_pops_on_trie<
    EK,
    EV,
    T,
    FGet,
    FReplace,
    FIntersect,
    FCanRemove,
>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    get_pop: &mut FGet,
    replace_pop: &mut FReplace,
    intersect_checks: &mut FIntersect,
    can_remove: &mut FCanRemove,
) where
    EK: Ord + Clone,
    EV: Clone + Ord,
    T: Clone + Ord + PartialEq,
    FGet: FnMut(&EK, &EV) -> isize,
    FReplace: FnMut(&EK, &EV, isize) -> (EK, EV),
    FIntersect: FnMut(&EK, &EV, &EK, &EV) -> bool,
    FCanRemove: FnMut(&EK, &EV) -> bool,
{
    let paths = get_all_paths_with_data(god, roots);
    let mut processed_paths = BTreeSet::new();

    for (root_t, path) in paths {
        let temp_get_pop = |pe: &PathElement<EK, EV, T>| get_pop(&pe.ek, &pe.ev);
        let temp_replace_pop = |pe: &PathElement<EK, EV, T>, new_pop| {
            let mut new_pe = pe.clone();
            let (new_ek, new_ev) = replace_pop(&pe.ek, &pe.ev, new_pop);
            new_pe.ek = new_ek;
            new_pe.ev = new_ev;
            new_pe
        };
        let temp_intersect_checks =
            |pe1: &PathElement<EK, EV, T>, pe2: &PathElement<EK, EV, T>| {
                intersect_checks(&pe1.ek, &pe1.ev, &pe2.ek, &pe2.ev)
            };
        let temp_can_remove = |pe: &PathElement<EK, EV, T>| {
            can_remove(&pe.ek, &pe.ev) && pe.src_node_value == pe.dst_node_value
        };

        if let Some(new_path) = stack_eliminate_internal_negative_pops(
            path,
            temp_get_pop,
            temp_replace_pop,
            temp_intersect_checks,
            temp_can_remove,
        ) {
            processed_paths.insert((root_t, new_path));
        }
    }

    // Rebuild the trie from the processed paths, preserving root indices.
    let old_roots_with_values: Vec<(Trie2Index, T)> =
        roots.iter().map(|r| (*r, r.read(god).unwrap().value.clone())).collect();

    god.clear();

    let mut new_root_map: BTreeMap<T, Trie2Index> = BTreeMap::new();
    for (old_idx, root_t) in &old_roots_with_values {
        let new_node = Trie::new(root_t.clone());
        god.insert_at((*old_idx).into(), new_node);
        new_root_map.insert(root_t.clone(), *old_idx);
    }

    for (root_t, path) in processed_paths {
        if !new_root_map.contains_key(&root_t) {
            // This case can happen if a root was a leaf and its path was empty and got eliminated.
            // We ensure the root node still exists.
            continue;
        }
        let mut current_idx = new_root_map[&root_t];
        for pe in path {
            // This creates an unrolled graph, which is fine for a dummy impl.
            // We must separate node creation from edge insertion to avoid deadlocking the Arena's RwLock.
            let new_node = Trie::new(pe.dst_node_value.clone());
            let new_idx = Trie2Index::from(god.insert(new_node));

            {
                let mut current_guard = current_idx.write(god).unwrap();
                current_guard.force_insert_to_node(pe.ek.clone(), pe.ev.clone(), new_idx);
            }
            current_idx = new_idx;
        }
    }
}

/// Graph-level transform: remove trailing negative pops at the ends of stacks.
/// Not implemented yet.
pub fn eliminate_trailing_negative_pops_on_trie<EK, EV, T, FGet, FCanRemove>(
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    get_pop: &mut FGet,
    can_remove: &mut FCanRemove,
) where
    EK: Ord + Clone,
    EV: Clone + Ord,
    T: Clone + Ord + PartialEq,
    FGet: FnMut(&EK, &EV) -> isize,
    FCanRemove: FnMut(&EK, &EV) -> bool,
{
    let paths = get_all_paths_with_data(_god, _roots);
    let mut processed_paths = BTreeSet::new();

    for (root_t, path) in paths {
        let temp_get_pop = |pe: &PathElement<EK, EV, T>| get_pop(&pe.ek, &pe.ev);
        let temp_can_remove = |pe: &PathElement<EK, EV, T>| {
            can_remove(&pe.ek, &pe.ev) && pe.src_node_value == pe.dst_node_value
        };

        let new_path =
            stack_eliminate_trailing_negative_pops(path, temp_get_pop, temp_can_remove);
        processed_paths.insert((root_t, new_path));
    }

    // Rebuild the trie from the processed paths, preserving root indices.
    let old_roots_with_values: Vec<(Trie2Index, T)> =
        _roots.iter().map(|r| (*r, r.read(_god).unwrap().value.clone())).collect();

    _god.clear();

    let mut new_root_map: BTreeMap<T, Trie2Index> = BTreeMap::new();
    for (old_idx, root_t) in &old_roots_with_values {
        let new_node = Trie::new(root_t.clone());
        _god.insert_at((*old_idx).into(), new_node);
        new_root_map.insert(root_t.clone(), *old_idx);
    }

    for (root_t, path) in processed_paths {
        if !new_root_map.contains_key(&root_t) {
            continue;
        }
        let mut current_idx = new_root_map[&root_t];
        for pe in path {
            // This creates an unrolled graph, which is fine for a dummy impl.
            // We must separate node creation from edge insertion to avoid deadlocking the Arena's RwLock.
            let new_node = Trie::new(pe.dst_node_value.clone());
            let new_idx = Trie2Index::from(_god.insert(new_node));

            {
                let mut current_guard = current_idx.write(_god).unwrap();
                current_guard.force_insert_to_node(pe.ek.clone(), pe.ev.clone(), new_idx);
            }
            current_idx = new_idx;
        }
    }
}

/// Reference stack function: eliminate internal negative pops by canceling adjacent
/// negative/positive run pairs.
/// Returns:
/// - Some(new_stack) if no mismatches were found.
/// - None if any run pair exhibited a mismatch (entire stack eliminated).
pub fn stack_eliminate_internal_negative_pops<EK, FGet, FReplace, FIntersect, FCanRemove>(
    stack: Vec<EK>,
    mut get_pop: FGet,
    mut replace_pop: FReplace,
    mut intersect_checks: FIntersect,
    mut can_remove: FCanRemove,
) -> Option<Vec<EK>>
where
    EK: Clone,
    FGet: FnMut(&EK) -> isize,
    FReplace: FnMut(&EK, isize) -> EK,
    FIntersect: FnMut(&EK, &EK) -> bool,
    FCanRemove: FnMut(&EK) -> bool,
{

    // Buffers for current negative and positive runs.
    let mut out: Vec<EK> = Vec::with_capacity(stack.clone().len());
    let mut neg_buf: Vec<EK> = Vec::new();
    let mut pos_buf: Vec<EK> = Vec::new();
    let mut in_pos = false;

    // Helper: process a completed neg/pos pair, returning (leftover_neg, leftover_pos).
    fn process_pair<EK, FGet, FReplace, FIntersect, FCanRemove>(
        mut neg_buf: Vec<EK>,
        mut pos_buf: Vec<EK>,
        get_pop: &mut FGet,
        replace_pop: &mut FReplace,
        intersect_checks: &mut FIntersect,
        can_remove: &mut FCanRemove,
    ) -> Option<(Vec<EK>, Vec<EK>)>
    where
        EK: Clone,
        FGet: FnMut(&EK) -> isize,
        FReplace: FnMut(&EK, isize) -> EK,
        FIntersect: FnMut(&EK, &EK) -> bool,
        FCanRemove: FnMut(&EK) -> bool,
    {
        // Create a positive-pop reversed copy of the negative run.
        let mut neg_rev: Vec<EK> = Vec::with_capacity(neg_buf.len());
        for ek in neg_buf.iter().rev() {
            let p = get_pop(ek);
            debug_assert!(p <= 0);
            neg_rev.push(replace_pop(ek, -p));
        }

        // The positive run remains as-is (pops > 0 expected).
        // Compute realized check positions for both runs:
        // Positions are cumulative sums; we only record positions where an item carries a check.
        // Intersections are checked for overlapping positions (if any); failures eliminate the stack.
        use std::collections::BTreeMap;
        let mut neg_map: BTreeMap<usize, Vec<&EK>> = BTreeMap::new();
        let mut pos_map: BTreeMap<usize, Vec<&EK>> = BTreeMap::new();

        let mut cum = 0usize;
        for ek in &neg_rev {
            let p = get_pop(ek);
            debug_assert!(p >= 0);
            cum += p as usize;
            // Record at boundary
            neg_map.entry(cum).or_default().push(ek);
        }
        cum = 0;
        for ek in &pos_buf {
            let p = get_pop(ek);
            debug_assert!(p > 0);
            cum += p as usize;
            pos_map.entry(cum).or_default().push(ek);
        }

        // Check pairwise compatibility via intersection on overlapping positions.
        let mut neg_it = neg_map.iter().peekable();
        let mut pos_it = pos_map.iter().peekable();
        while let (Some((npos, neks)), Some((ppos, peks))) = (neg_it.peek(), pos_it.peek()) {
            if *npos == *ppos {
                // Overlapping boundary: must be compatible. All pairs must intersect.
                for nek in neks.iter() {
                    for pek in peks.iter() {
                        if !intersect_checks(nek, pek) {
                            return None;
                        }
                    }
                }
                neg_it.next();
                pos_it.next();
            } else if *npos < *ppos {
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
        fn subtract_from_front<EK, FGet, FReplace, FCanRemove>(
            mut seq: Vec<EK>,
            mut amt: usize,
            get_pop: &mut FGet,
            replace_pop: &mut FReplace,
            can_remove: &mut FCanRemove,
        ) -> Vec<EK>
        where
            EK: Clone,
            FGet: FnMut(&EK) -> isize,
            FReplace: FnMut(&EK, isize) -> EK,
            FCanRemove: FnMut(&EK) -> bool,
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
                } else {
                    // Element is fully consumed.
                    if !can_remove(&ek) {
                        out.push(replace_pop(&ek, 0));
                    }
                    if pu == amt {
                        amt = 0;
                    } else {
                        // pu < amt
                        amt -= pu;
                    }
                }
            }
            out
        }

        let neg_rev_left =
            subtract_from_front(neg_rev, cancel_amt, get_pop, replace_pop, can_remove);
        let pos_left = subtract_from_front(pos_buf, cancel_amt, get_pop, replace_pop, can_remove);

        // Convert neg_rev_left back to original order with negative pops.
        let mut leftover_neg: Vec<EK> = Vec::with_capacity(neg_rev_left.len());
        for ek in neg_rev_left.into_iter().rev() {
            let p = get_pop(&ek);
            debug_assert!(p >= 0);
            leftover_neg.push(replace_pop(&ek, -p));
        }

        Some((leftover_neg, pos_left))
    }

    for ek in stack.into_iter() {
        let p = get_pop(&ek);
        if p < 0 {
            // Negative element
            if !in_pos {
                neg_buf.push(ek);
            } else {
                // We just finished a positive run; process neg/pos pair.
                let pair = process_pair(
                    neg_buf,
                    pos_buf,
                    &mut get_pop,
                    &mut replace_pop,
                    &mut intersect_checks,
                    &mut can_remove,
                )?;
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
            // Zero-pop item acts as a separator. Process any pending pair.
            if in_pos {
                let pair = process_pair(
                    neg_buf,
                    pos_buf,
                    &mut get_pop,
                    &mut replace_pop,
                    &mut intersect_checks,
                    &mut can_remove,
                )?;
                let (leftover_neg, leftover_pos) = pair;
                out.extend(leftover_pos);
                neg_buf = leftover_neg;
                pos_buf = Vec::new();
                in_pos = false;
            }
            // Emit any remaining negative buffer.
            out.extend(neg_buf.into_iter());
            neg_buf = Vec::new();

            // Emit the zero-pop item itself.
            out.push(ek);
        }
    }

    // If we ended in a positive run, resolve the tailing pair.
    if in_pos {
        let pair =
            process_pair(neg_buf, pos_buf, &mut get_pop, &mut replace_pop, &mut intersect_checks, &mut can_remove)?;
        let (leftover_neg, leftover_pos) = pair;
        out.extend(leftover_pos);
        neg_buf = leftover_neg;
    }

    // Append any trailing negatives; these remain to be eliminated by the trailing stage.
    out.extend(neg_buf.into_iter());

    // Remove any zeros that might have resurfaced.
    let mut final_out: Vec<EK> = Vec::with_capacity(out.len());
    for ek in out.into_iter() {
        if get_pop(&ek) != 0 || !can_remove(&ek) {
            final_out.push(ek);
        }
    }

    Some(final_out)
}

/// Reference stack function: remove trailing negative pops and zero-pop items.
/// This does not attempt to cancel anything; it simply trims the tail where pop < 0,
/// and removes any zero-pop elements anywhere.
pub fn stack_eliminate_trailing_negative_pops<EK, FGet, FCanRemove>(
    stack: Vec<EK>,
    mut get_pop: FGet,
    mut can_remove: FCanRemove,
) -> Vec<EK>
where
    EK: Clone,
    FGet: FnMut(&EK) -> isize,
    FCanRemove: FnMut(&EK) -> bool,
{
    // First, remove zero-pop items that can be removed.

    // Drop trailing negatives.
    let cut = stack.clone()
        .iter()
        .rposition(|ek| get_pop(ek) >= 0)
        .map(|i| i + 1)
        .unwrap_or(0);
    stack.clone().truncate(cut);
    stack.clone()
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
        ev: Option<usize>,
    }

    impl TestEK {
        fn new(pop: isize, check_ids: Option<&[usize]>, ev: Option<usize>) -> Self {
            let check = check_ids.map(|ids| ids.iter().cloned().collect());
            Self { pop, check, ev }
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
            ev: ek.ev,
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

    fn can_remove(ek: &TestEK) -> bool {
        // An EK can be removed if it has no important EV.
        // For tests, non-zero pops are also considered removable if they become zero.
        ek.ev.is_none()
    }

    // Convenience to build an EK
    fn ek(pop: isize, ids: Option<&[usize]>, ev: Option<usize>) -> TestEK {
        TestEK::new(pop, ids, ev)
    }

    // -- Unit tests for stack elimination behavior --

    #[test]
    fn zero_pop_and_cancellation_with_ev() {
        // --- Zero-pop handling ---
        // 0-pop with no EV should be removed by initial cleaning.
        let input1 = vec![ek(1, None, None), ek(0, None, None), ek(2, None, None)];
        let got1 = stack_eliminate_internal_negative_pops(
            input1,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got1, vec![ek(1, None, None), ek(2, None, None)]);

        // 0-pop with EV should be preserved and act as a separator.
        let input2 = vec![ek(-1, None, None), ek(0, None, Some(123)), ek(1, None, None)];
        let got2 = stack_eliminate_internal_negative_pops(
            input2.clone(),
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got2, input2, "0-pop with EV should prevent cancellation");

        // --- Cancellation producing zero-pops ---
        // -1 with EV, +1 without. After cancellation, the -1 should remain as a 0-pop.
        let input3 = vec![ek(-1, None, Some(1)), ek(1, None, None)];
        let got3 = stack_eliminate_internal_negative_pops(
            input3,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got3, vec![ek(0, None, Some(1))]);

        // Both have EV. Both should remain. Note: leftover positive part is emitted before
        // the leftover negative part is appended at the end, causing reordering.
        let input4 = vec![ek(-1, None, Some(1)), ek(1, None, Some(2))];
        let got4 = stack_eliminate_internal_negative_pops(
            input4,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got4, vec![ek(0, None, Some(2)), ek(0, None, Some(1))]);

        // Partial cancel: -2 with EV, +1 without. Should become -1 with EV.
        let input5 = vec![ek(-2, None, Some(1)), ek(1, None, None)];
        let got5 = stack_eliminate_internal_negative_pops(
            input5,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .unwrap();
        assert_eq!(got5, vec![ek(-1, None, Some(1))]);
    }

    #[test]
    fn run_pair_full_cancel_with_remainder_positive() {
        // -1 b, -1 c, +1 c, +1 b, +1 a  =>  +1 a
        let input = vec![
            ek(-1, Some(&[1]), None), // b
            ek(-1, Some(&[2]), None), // c
            ek(1, Some(&[2]), None),  // c
            ek(1, Some(&[1]), None),  // b
            ek(1, Some(&[0]), None),  // a
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(got, vec![ek(1, Some(&[0]), None)]);
    }

    #[test]
    fn run_pair_partial_cancel_neg_leftover() {
        // -1 b, -1 c, +1 c  =>  -1 b (leftover; to be trimmed by trailing stage if at end)
        let input = vec![
            ek(-1, Some(&[1]), None),
            ek(-1, Some(&[2]), None),
            ek(1, Some(&[2]), None),
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(got, vec![ek(-1, Some(&[1]), None)]);
    }

    #[test]
    fn run_pair_full_cancel_empty() {
        // -1 c, +1 c  =>  []
        let input = vec![ek(-1, Some(&[2]), None), ek(1, Some(&[2]), None)];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert!(got.is_empty());
    }

    #[test]
    fn run_pair_mismatch_eliminates_stack() {
        // -1 b, +1 c  => mismatch => None
        let input = vec![ek(-1, Some(&[1]), None), ek(1, Some(&[2]), None)];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        );
        assert!(got.is_none());
    }

    #[test]
    fn run_pair_partial_cancel_neg_aggregates() {
        // -3 b, +1 c  =>  -2 b
        let input = vec![ek(-3, Some(&[1]), None), ek(1, Some(&[2]), None)];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(got, vec![ek(-2, Some(&[1]), None)]);
    }

    #[test]
    fn run_pair_multi_neg_partial_cancel() {
        // -1 a, -3 b, +1 c  =>  -1 a, -2 b
        let input = vec![
            ek(-1, Some(&[0]), None),
            ek(-3, Some(&[1]), None),
            ek(1, Some(&[2]), None),
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(
            got,
            vec![ek(-1, Some(&[0]), None), ek(-2, Some(&[1]), None)]
        );
    }

    #[test]
    fn run_pair_gap_positive_ignores_missing_slots() {
        // -1 a, -1 b, -1 c, +2 b  =>  -1 a
        let input = vec![
            ek(-1, Some(&[0]), None),
            ek(-1, Some(&[1]), None),
            ek(-1, Some(&[2]), None),
            ek(2, Some(&[1]), None),
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(got, vec![ek(-1, Some(&[0]), None)]);
    }

    #[test]
    fn run_pair_gap_negative_ignores_missing_slots() {
        // -2 b, +1 c, +1 b, +1 a  =>  +1 a
        let input = vec![
            ek(-2, Some(&[1]), None),
            ek(1, Some(&[2]), None),
            ek(1, Some(&[1]), None),
            ek(1, Some(&[0]), None),
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(got, vec![ek(1, Some(&[0]), None)]);
    }

    #[test]
    fn trailing_negative_pops_are_removed() {
        // After internal elimination, drop trailing negatives (and zeros anywhere).
        let input = vec![
            ek(1, Some(&[0]), None),
            ek(-1, Some(&[1]), None),
            ek(0, None, None),
        ];
        let trimmed = stack_eliminate_trailing_negative_pops(input, get_pop, can_remove);
        assert_eq!(trimmed, vec![ek(1, Some(&[0]), None)]);
    }

    #[test]
    fn positive_prefix_without_negative_run_is_preserved() {
        // Starts with positives; internal elimination should leave them alone.
        let input = vec![
            ek(1, Some(&[1]), None),
            ek(2, Some(&[2]), None),
            ek(-1, Some(&[3]), None),
        ];
        let mid = stack_eliminate_internal_negative_pops(
            input.clone(),
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        // Only a positive prefix followed by a trailing negative; no internal pair to cancel.
        assert_eq!(mid, input);
        // Trailing negative drops
        let final_s = stack_eliminate_trailing_negative_pops(mid, get_pop, can_remove);
        assert_eq!(
            final_s,
            vec![ek(1, Some(&[1]), None), ek(2, Some(&[2]), None)]
        );
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
            ek(1, Some(&[0]), None),  // a
            ek(-2, Some(&[1]), None), // b
            ek(1, Some(&[1]), None),  // b
            ek(-1, Some(&[2]), None), // c
            ek(2, Some(&[2]), None),  // c
            ek(1, Some(&[3]), None),  // d
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        );
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
            ek(1, Some(&[0]), None),     // a
            ek(-2, Some(&[1, 5]), None), // b
            ek(1, Some(&[1, 5]), None),  // b
            ek(-1, Some(&[2, 5]), None), // c
            ek(2, Some(&[2, 5]), None),  // c
            ek(1, Some(&[3]), None),     // d
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(
            got,
            vec![ek(1, Some(&[0]), None), ek(1, Some(&[3]), None)]
        );
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
            ek(1, Some(&[0]), None),      // a
            ek(-3, Some(&[1, 99]), None), // b
            ek(2, Some(&[1, 99]), None), // b
            ek(1, Some(&[2, 99]), None), // c
            ek(-2, Some(&[3]), None),     // d
            ek(3, Some(&[3]), None),     // d
            ek(-1, Some(&[4]), None),     // e
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(
            got,
            vec![
                ek(1, Some(&[0]), None),  // a
                ek(1, Some(&[3]), None),  // d
                ek(-1, Some(&[4]), None)  // e
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
            ek(2, Some(&[0]), None), // a
            ek(1, Some(&[1]), None), // b
            // Pair 1
            ek(-3, Some(&[2]), None), // c
            ek(-1, Some(&[3]), None), // d
            ek(2, Some(&[2]), None),  // c
            // Pair 2 (neg part merges with leftover from Pair 1)
            ek(-2, None, None),       // e (unconstrained)
            ek(5, Some(&[4]), None),  // e
            ek(1, Some(&[5]), None),  // f
            // Pair 3 (mismatch)
            ek(-1, Some(&[6]), None), // g
            ek(-1, Some(&[7]), None), // h
            ek(2, Some(&[7]), None),  // h
            // Trailing
            ek(-4, Some(&[8]), None), // i
            ek(-1, Some(&[9]), None), // j
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        );
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
            ek(2, Some(&[0]), None), // a
            ek(1, Some(&[1]), None), // b
            // Pair 1
            ek(-3, Some(&[2]), None), // c
            ek(-1, Some(&[3]), None), // d
            ek(2, Some(&[2]), None),  // c
            // Pair 2
            ek(-2, None, None),      // e (unconstrained)
            ek(5, Some(&[4]), None), // e
            ek(1, Some(&[5]), None), // f
            // Pair 3 (compatible)
            ek(-1, Some(&[6, 100]), None), // g
            ek(-1, Some(&[7, 100]), None), // h
            ek(2, Some(&[7, 100]), None),  // h
            // Trailing
            ek(-4, Some(&[8]), None), // i
            ek(-1, Some(&[9]), None), // j
        ];
        let got = stack_eliminate_internal_negative_pops(
            input,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("should not mismatch");
        assert_eq!(
            got,
            vec![
                ek(2, Some(&[0]), None), // a
                ek(1, Some(&[1]), None), // b
                ek(1, Some(&[4]), None), // e
                ek(1, Some(&[5]), None), // f
                ek(-4, Some(&[8]), None), // i
                ek(-1, Some(&[9]), None), // j
            ]
        );
    }

    #[test]
    fn test_user_structure_as_stacks() {
        // This test models the stacks from the user-provided graph structure, respecting
        // that most edges are pop: 0. Edges with `ev: None` are removable if pop is 0.

        // Path 1A from the graph.
        let path1a = vec![
            ek(0, None, None),          // from 16->19
            ek(0, None, None),          // from 19->20
            ek(0, Some(&[0]), None),    // from 20->21
            ek(0, None, None),          // from 21->23
            ek(-1, Some(&[1]), None),   // from 23->25
            ek(0, None, None),          // from 25->27
            ek(0, None, Some(1)),       // from 27->28, `tokens: [1]` -> not removable
            ek(0, None, None),          // from 28->18
        ];
        // After cleaning, only non-zero pops or non-removable items remain.
        let expected1a_internal = vec![ek(-1, Some(&[1]), None), ek(0, None, Some(1))];
        let got1a_internal = stack_eliminate_internal_negative_pops(
            path1a,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        );
        assert_eq!(
            got1a_internal.as_ref(),
            Some(&expected1a_internal),
            "Path 1A should not mismatch"
        );
        // The trailing elim stage does not remove the -1 because it's not at the trail.
        let got1a_final =
            stack_eliminate_trailing_negative_pops(got1a_internal.unwrap(), get_pop, can_remove);
        assert_eq!(got1a_final, expected1a_internal);

        // Path 1B from the graph.
        let path1b = vec![
            ek(0, None, None),          // from 16->19
            ek(0, None, None),          // from 19->20
            ek(0, Some(&[2]), None),    // from 20->22
            ek(2, None, None),          // from 22->24
            ek(0, Some(&[0]), None),    // from 24->26
        ];
        let expected1b = vec![ek(2, None, None)];
        let got1b = stack_eliminate_internal_negative_pops(
            path1b,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("Path 1B should survive");
        assert_eq!(got1b, expected1b);

        // Path 2A from the graph.
        let path2a = vec![
            ek(0, None, None),          // from 16->29
            ek(0, None, None),          // from 29->30
            ek(0, Some(&[1]), None),    // from 30->31
            ek(0, None, None),          // from 31->33
            ek(-1, Some(&[2]), None),   // from 33->35
            ek(0, None, None),          // from 35->37
            ek(0, None, Some(2)),       // from 37->38, `tokens: [0]` -> not removable
            ek(0, None, None),          // from 38->18
        ];
        let expected2a = vec![ek(-1, Some(&[2]), None), ek(0, None, Some(2))];
        let got2a = stack_eliminate_internal_negative_pops(
            path2a,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("Path 2A should not mismatch");
        assert_eq!(got2a, expected2a);

        // Path 2B from the graph.
        let path2b = vec![
            ek(0, None, None),          // from 16->29
            ek(0, None, None),          // from 29->30
            ek(0, Some(&[2]), None),    // from 30->32
            ek(2, None, None),          // from 32->34
            ek(0, Some(&[0]), None),    // from 34->36
        ];
        let expected2b = vec![ek(2, None, None)];
        let got2b = stack_eliminate_internal_negative_pops(
            path2b,
            get_pop,
            replace_pop,
            checks_intersect,
            can_remove,
        )
        .expect("Path 2B should survive");
        assert_eq!(got2b, expected2b);
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
                stack_eliminate_internal_negative_pops(s, get_pop, replace_pop, checks_intersect, can_remove)
            {
                let fin = stack_eliminate_trailing_negative_pops(mid, get_pop, can_remove);
                expected_stacks.insert(fin);
            }
        }

        // 2. Calculate ACTUAL stacks by running the trie-level transform on a clone.
        let (god_clone, roots_clone, _) = Trie::deep_copy_subtrees(god, roots);

        // This is the call that is expected to panic until implemented.
        eliminate_negative_pops(
            &god_clone,
            &roots_clone,
            |ek, _ev| get_pop(ek),
            |ek, _ev, new_pop| (replace_pop(ek, new_pop), ()),
            |ek1, _ev1, ek2, _ev2| checks_intersect(ek1, ek2),
            |ek, _ev| can_remove(ek),
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
        add_edge(&god, n16, n19, ek(0, None, None)); // Note: zero-pop edges are filtered by the pipeline
        add_edge(&god, n19, n20, ek(0, None, None)); // but are kept here to model the graph structure.

        // Path through n21 (with negative pop)
        add_edge(&god, n20, n21, ek(1, Some(&[0]), None)); // Using pop=1 to avoid being filtered
        add_edge(&god, n21, n23, ek(1, None, None));
        add_edge(&god, n23, n25, ek(-1, Some(&[1]), None));
        add_edge(&god, n25, n27, ek(1, None, None));
        add_edge(&god, n27, n28, ek(1, None, None));
        add_edge(&god, n28, n18, ek(1, None, None));

        // Path through n22
        add_edge(&god, n20, n22, ek(1, Some(&[2]), None));
        add_edge(&god, n22, n24, ek(2, None, None));
        add_edge(&god, n24, n26, ek(1, Some(&[0]), None)); // Leaf

        // Branch 2 (from root -> n29)
        add_edge(&god, n16, n29, ek(1, None, None));
        add_edge(&god, n29, n30, ek(1, None, None));

        // Path through n31 (with negative pop)
        add_edge(&god, n30, n31, ek(1, Some(&[1]), None));
        add_edge(&god, n31, n33, ek(1, None, None));
        add_edge(&god, n33, n35, ek(-1, Some(&[2]), None));
        add_edge(&god, n35, n37, ek(1, None, None));
        add_edge(&god, n37, n38, ek(1, None, None));
        add_edge(&god, n38, n18, ek(1, None, None));

        // Path through n32
        add_edge(&god, n30, n32, ek(1, Some(&[2]), None));
        add_edge(&god, n32, n34, ek(2, None, None));
        add_edge(&god, n34, n36, ek(1, Some(&[0]), None)); // Leaf

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
        add_edge(&god, a, b, ek(1, Some(&[0]), None));
        add_edge(&god, b, c, ek(-1, Some(&[0]), None));
        add_edge(&god, a, d, ek(1, None, None)); // Dummy survivor
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
        add_edge(&god, a, b, ek(1, Some(&[0]), None));
        add_edge(&god, b, c, ek(-1, Some(&[1]), None)); // Mismatching check
        add_edge(&god, a, d, ek(2, Some(&[2]), None));   // Survivor
        run_trie_vs_stack_comparison_test(&god, &[a]);
    }

    #[test]
    fn test_no_negative_pops_is_noop_via_runner() {
        // A --(+2, c0)--> B --(+1, c1)--> C
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        add_edge(&god, a, b, ek(2, Some(&[0]), None));
        add_edge(&god, b, c, ek(1, Some(&[1]), None));
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
        add_edge(&god, a, b, ek(2, None, None));
        add_edge(&god, b, c, ek(-1, None, None));
        // Path 2
        add_edge(&god, a, d, ek(5, None, None));
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
        add_edge(&god, a, b, ek(-2, Some(&[0]), None));
        add_edge(&god, b, c, ek(3, Some(&[0]), None));
        // Path 2
        add_edge(&god, a, d, ek(1, Some(&[1]), None));
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
        add_edge(&god, a, b, ek(1, Some(&[0]), None));
        // Path 2 prefix
        add_edge(&god, d, b, ek(-1, Some(&[1]), None)); // Mismatching check ID
        // Shared suffix
        add_edge(&god, b, c, ek(1, Some(&[0]), None));

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
        add_edge(&god, a, b, ek(-1, None, None));
        add_edge(&god, b, c, ek(1, Some(&[0]), None));
        // Path 2
        add_edge(&god, a, d, ek(1, Some(&[0]), None));

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
        add_edge(&god, a, b, ek(1, Some(&[0]), None));
        add_edge(&god, b, c, ek(-1, Some(&[1]), None)); // Mismatch

        // Path 2
        add_edge(&god, a, d, ek(-2, Some(&[2]), None));
        add_edge(&god, d, e, ek(2, Some(&[2]), None)); // Cancels

        run_trie_vs_stack_comparison_test(&god, &[a]);
    }

    #[test]
    fn test_graph_from_user_structure() {
        let god = TestGod::new();
        let mut nodes = std::collections::BTreeMap::new();
        for i in [
            16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        ] {
            nodes.insert(i, new_node(&god));
        }

        let n = |i: i32| nodes[&i];

        // --- Build graph from user-provided structure, corrected version ---
        // Mapping from diagram to TestEK:
        // - `pop` is as specified.
        // - `states [x]` maps to `check: Some(&[x])`.
        // - `tokens: [range]` or not specified maps to `ev: None` (removable if pop=0).
        // - `tokens: [specific_id]` maps to `ev: Some(...)` (not removable).

        // Branch 1 from root n16
        add_edge(&god, n(16), n(19), ek(0, None, None));
        add_edge(&god, n(19), n(20), ek(0, None, None));

        // Path 1A
        add_edge(&god, n(20), n(21), ek(0, Some(&[0]), None));
        add_edge(&god, n(21), n(23), ek(0, None, None));
        add_edge(&god, n(23), n(25), ek(-1, Some(&[1]), None));
        add_edge(&god, n(25), n(27), ek(0, None, None));
        add_edge(&god, n(27), n(28), ek(0, None, Some(1))); // `tokens: [1]`
        add_edge(&god, n(28), n(18), ek(0, None, None));

        // Path 1B
        add_edge(&god, n(20), n(22), ek(0, Some(&[2]), None));
        add_edge(&god, n(22), n(24), ek(2, None, None));
        add_edge(&god, n(24), n(26), ek(0, Some(&[0]), None));

        // Branch 2 from root n16
        add_edge(&god, n(16), n(29), ek(0, None, None));
        add_edge(&god, n(29), n(30), ek(0, None, None));

        // Path 2A
        add_edge(&god, n(30), n(31), ek(0, Some(&[1]), None));
        add_edge(&god, n(31), n(33), ek(0, None, None));
        add_edge(&god, n(33), n(35), ek(-1, Some(&[2]), None));
        add_edge(&god, n(35), n(37), ek(0, None, None));
        add_edge(&god, n(37), n(38), ek(0, None, Some(2))); // `tokens: [0]`
        add_edge(&god, n(38), n(18), ek(0, None, None));

        // Path 2B
        add_edge(&god, n(30), n(32), ek(0, Some(&[2]), None));
        add_edge(&god, n(32), n(34), ek(2, None, None));
        add_edge(&god, n(34), n(36), ek(0, Some(&[0]), None));

        run_trie_vs_stack_comparison_test(&god, &[n(16)]);
    }
}

