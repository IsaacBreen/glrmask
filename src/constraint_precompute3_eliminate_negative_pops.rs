// src/constraint_precompute3_eliminate_negative_pops.rs
//
// New approach: do NOT reorder operations. Negative pops represent pushes
// that must be canceled in-order against following positive pops.
// We handle elimination by pairing runs (negative on the left, positive on the right).
//
// This module exposes a high-level entry point `eliminate_negative_pops` that will
// eventually perform the in-trie transform, but for now it just delegates to two
// yet-to-be-implemented functions (left as todo!()):
//   - eliminate_internal_negative_pops(...)
//   - remove_trailing_negative_pops(...)
//
// We DO implement the reference (single-stack) versions of both stages in the test
// module below. Those serve as the "golden" semantics and are used by tests.
//
// Do not modify src/trie.rs.
// The previous "bubble" logic and its tests are removed, as the problem statement
// clarified the correct semantics (order matters; no reordering).

use crate::datastructures::trie::{GodWrapper, Trie2Index};

/// High-level pipeline that will eliminate negative pops in a trie graph:
/// 1) Eliminate internal negatives by pairing negative/positive runs in order.
/// 2) Remove any trailing negative pops that cannot be paired.
///
/// This is a stub for the graph transformation. The actual in-trie rewriting logic
/// is intentionally left unimplemented in this commit (todo!()).
///
/// The reference, single-stack implementation lives in the tests module and
/// exercises the full semantic logic independently of the trie shape.
pub fn eliminate_negative_pops<EK, EV, T, FGetPop, FSetPop, FExtractCheck, FBuildKey, FIntersect, FMergeEV>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    mut _get_pop: FGetPop,
    mut _set_pop_like: FSetPop,
    mut _extract_check: FExtractCheck,
    mut _build_key_from: FBuildKey,
    mut _intersect_checks: FIntersect,
    mut _merge_ev: FMergeEV,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGetPop: FnMut(&EK) -> isize,
    FSetPop: FnMut(&EK, isize) -> EK,
    FExtractCheck: FnMut(&EK) -> Option<()>, // Placeholder type for now; real impl will define its own domain.
    FBuildKey: FnMut(isize, Option<()>) -> EK, // Placeholder builder signature for now.
    FIntersect: FnMut(Option<()>, Option<()>) -> Option<()>,
    FMergeEV: FnMut(&mut EV, EV),
{
    // Stage 1: eliminate internal negative pops by pairing negative/positive runs.
    eliminate_internal_negative_pops(
        god,
        roots,
        &mut _get_pop,
        &mut _set_pop_like,
        &mut _extract_check,
        &mut _build_key_from,
        &mut _intersect_checks,
        &mut _merge_ev,
    );

    // Stage 2: remove trailing negative pops that cannot be paired.
    remove_trailing_negative_pops(
        god,
        roots,
        &mut _get_pop,
        &mut _set_pop_like,
        &mut _extract_check,
        &mut _build_key_from,
        &mut _intersect_checks,
        &mut _merge_ev,
    );
}

/// Stage 1 (graph): eliminate internal negative pops by pairing negative/positive runs in order.
///
/// Intentionally left unimplemented in this commit. We provide a fully implemented
/// stack-based reference in the test module to pin down semantics.
pub fn eliminate_internal_negative_pops<EK, EV, T, FGetPop, FSetPop, FExtractCheck, FBuildKey, FIntersect, FMergeEV>(
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    _get_pop: &mut FGetPop,
    _set_pop_like: &mut FSetPop,
    _extract_check: &mut FExtractCheck,
    _build_key_from: &mut FBuildKey,
    _intersect_checks: &mut FIntersect,
    _merge_ev: &mut FMergeEV,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGetPop: FnMut(&EK) -> isize,
    FSetPop: FnMut(&EK, isize) -> EK,
    FExtractCheck: FnMut(&EK) -> Option<()>,
    FBuildKey: FnMut(isize, Option<()>) -> EK,
    FIntersect: FnMut(Option<()>, Option<()>) -> Option<()>,
    FMergeEV: FnMut(&mut EV, EV),
{
    todo!("Graph transform not implemented yet. Reference (single-stack) version is tested and lives in this module's tests.");
}

/// Stage 2 (graph): remove trailing negative pops that cannot be paired.
///
/// Intentionally left unimplemented in this commit. We provide a fully implemented
/// stack-based reference in the test module to pin down semantics.
pub fn remove_trailing_negative_pops<EK, EV, T, FGetPop, FSetPop, FExtractCheck, FBuildKey, FIntersect, FMergeEV>(
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    _get_pop: &mut FGetPop,
    _set_pop_like: &mut FSetPop,
    _extract_check: &mut FExtractCheck,
    _build_key_from: &mut FBuildKey,
    _intersect_checks: &mut FIntersect,
    _merge_ev: &mut FMergeEV,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGetPop: FnMut(&EK) -> isize,
    FSetPop: FnMut(&EK, isize) -> EK,
    FExtractCheck: FnMut(&EK) -> Option<()>,
    FBuildKey: FnMut(isize, Option<()>) -> EK,
    FIntersect: FnMut(Option<()>, Option<()>) -> Option<()>,
    FMergeEV: FnMut(&mut EV, EV),
{
    todo!("Graph transform not implemented yet. Reference (single-stack) version is tested and lives in this module's tests.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PrecomputedNodeContents;
    use crate::datastructures::trie::{Trie, Trie2Index};
    use crate::trie_test_framework::harness;

    use std::collections::BTreeSet;

    // -----------------------------
    // Test EK type and helpers
    // -----------------------------
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct TestEK {
        pop: isize,
        check: Option<usize>,
    }

    impl TestEK {
        fn new(pop: isize, check: Option<usize>) -> Self {
            TestEK { pop, check }
        }
    }

    type TestEV = ();
    type TestT = PrecomputedNodeContents;
    type TestGod = GodWrapper<TestEK, TestEV, TestT>;

    fn new_node(god: &TestGod) -> Trie2Index {
        harness::new_node(god, PrecomputedNodeContents::internal())
    }

    fn add_edge(god: &TestGod, from: Trie2Index, to: Trie2Index, key: TestEK) {
        harness::add_edge(god, from, to, key, ());
    }

    // -----------------------------
    // Reference (single-stack) implementation
    // -----------------------------

    // Utility: remove zero-pop edges (no-ops).
    fn drop_zero_pops(mut stack: Vec<TestEK>) -> Vec<TestEK> {
        stack.retain(|ek| ek.pop != 0);
        stack
    }

    // Intersection logic where a gap (None) is a wildcard.
    // - Mismatch (Some(a) vs Some(b) where a != b) -> None
    // - Match (Some(a) vs Some(a)) -> Some(Some(a))
    // - Gap (None vs Some(a)) -> Some(Some(a))
    // - Double Gap (None vs None) -> Some(None)
    fn intersect_checks_gappy(a: Option<usize>, b: Option<usize>) -> Option<Option<usize>> {
        match (a, b) {
            (Some(va), Some(vb)) if va != vb => None, // Mismatch
            (Some(_), Some(_)) => Some(a),           // Match
            (None, Some(_)) => Some(b),              // Neg has gap, take Pos's check
            (Some(_), None) => Some(a),              // Pos has gap, take Neg's check
            (None, None) => Some(None),              // Both have gap
        }
    }

    // A helper to consume pops from a run up to a certain amount.
    // Returns the remaining items.
    fn consume_pops(run: Vec<TestEK>, amount_to_consume: isize) -> Vec<TestEK> {
        let mut remaining = Vec::new();
        let mut consumed = 0;
        for mut ek in run {
            if consumed >= amount_to_consume {
                remaining.push(ek);
                continue;
            }
            let pop = ek.pop;
            let can_consume = amount_to_consume - consumed;
            if pop <= can_consume {
                consumed += pop;
            // This item is fully consumed, do not add to remaining.
            } else {
                ek.pop -= can_consume;
                consumed = amount_to_consume;
                remaining.push(ek);
            }
        }
        remaining
    }

    fn cancel_run_pair(
        neg_run: &[TestEK],
        pos_run: &[TestEK],
        intersect: &impl Fn(Option<usize>, Option<usize>) -> Option<Option<usize>>,
    ) -> Result<(Vec<TestEK>, Vec<TestEK>), ()> {
        // Convert negative run to positive counts and reverse for alignment
        let mut neg_rev: Vec<TestEK> = neg_run.iter().rev().cloned().collect();
        for ek in &mut neg_rev {
            debug_assert!(ek.pop < 0);
            ek.pop = -ek.pop;
        }
        let pos: Vec<TestEK> = pos_run.to_vec();

        // Totals and overlap
        let sum_neg: isize = neg_rev.iter().map(|e| e.pop).sum();
        let sum_pos: isize = pos.iter().map(|e| e.pop).sum();
        let overlap = sum_neg.min(sum_pos);

        // Build realized check maps
        use std::collections::BTreeMap;
        let mut neg_map: BTreeMap<isize, Option<usize>> = BTreeMap::new();
        let mut pos_map: BTreeMap<isize, Option<usize>> = BTreeMap::new();
        let mut acc = 0;
        for ek in &neg_rev {
            acc += ek.pop;
            if ek.check.is_some() {
                neg_map.insert(acc, ek.check);
            }
        }
        acc = 0;
        for ek in &pos {
            acc += ek.pop;
            if ek.check.is_some() {
                pos_map.insert(acc, ek.check);
            }
        }

        // Validate pairwise intersections along the overlap frontier.
        for s in 1..=overlap {
            let a = neg_map.get(&s).cloned().flatten();
            let b = pos_map.get(&s).cloned().flatten();
            if intersect(a, b).is_none() {
                return Err(()); // mismatch => eliminate entire stack
            }
        }

        // Reconstruct the remaining runs by consuming the overlap.
        let mut neg_after_consume = consume_pops(neg_rev, overlap);
        let pos_after_consume = consume_pops(pos, overlap);

        // Restore negative sign and original order for the negative run leftovers
        neg_after_consume.reverse();
        for ek in &mut neg_after_consume {
            ek.pop = -ek.pop;
        }

        Ok((neg_after_consume, pos_after_consume))
    }

    // Reference stage 1: eliminate internal negatives by pairing runs (neg on left, pos on right).
    fn eliminate_internal_pops_stack(
        stack: Vec<TestEK>,
        intersect: impl Fn(Option<usize>, Option<usize>) -> Option<Option<usize>>,
    ) -> Option<Vec<TestEK>> {
        let cleaned = drop_zero_pops(stack);
        if cleaned.is_empty() {
            return Some(vec![]);
        }

        let mut out: Vec<TestEK> = Vec::with_capacity(cleaned.len());
        let mut i = 0usize;

        while i < cleaned.len() {
            let cur = cleaned[i];
            if cur.pop < 0 {
                // Collect the negative run [i..j)
                let mut j = i + 1;
                while j < cleaned.len() && cleaned[j].pop < 0 {
                    j += 1;
                }
                // If no following positive run, we cannot eliminate; just copy the run.
                if j >= cleaned.len() || cleaned[j].pop <= 0 {
                    out.extend_from_slice(&cleaned[i..j]);
                    i = j;
                    continue;
                }
                // Collect the positive run [j..k)
                let mut k = j + 1;
                while k < cleaned.len() && cleaned[k].pop > 0 {
                    k += 1;
                }

                let neg_run = &cleaned[i..j];
                let pos_run = &cleaned[j..k];

                match cancel_run_pair(neg_run, pos_run, &intersect) {
                    Ok((neg_left, pos_left)) => {
                        out.extend(neg_left);
                        out.extend(pos_left);
                        i = k;
                    }
                    Err(()) => return None,
                }
            } else if cur.pop > 0 {
                // Copy a positive run through unchanged.
                let mut j = i + 1;
                while j < cleaned.len() && cleaned[j].pop > 0 {
                    j += 1;
                }
                out.extend_from_slice(&cleaned[i..j]);
                i = j;
            } else {
                i += 1;
            }
        }
        Some(out)
    }

    // Reference stage 2: remove trailing negative pops.
    fn remove_trailing_negative_pops_stack(mut stack: Vec<TestEK>) -> Vec<TestEK> {
        stack = drop_zero_pops(stack);
        while let Some(last) = stack.last() {
            if last.pop < 0 {
                stack.pop();
            } else {
                break;
            }
        }
        stack
    }

    // Reference full pipeline over a single stack.
    fn eliminate_negative_pops_stack(
        stack: Vec<TestEK>,
        intersect: impl Fn(Option<usize>, Option<usize>) -> Option<Option<usize>>,
    ) -> Option<Vec<TestEK>> {
        let stage1 = eliminate_internal_pops_stack(stack, intersect)?;
        let stage2 = remove_trailing_negative_pops_stack(stage1);
        Some(stage2)
    }

    // -----------------------------
    // Unit tests for the reference stack implementation
    // -----------------------------

    #[test]
    fn example_reordering_not_allowed_but_ok() {
        let input = vec![
            TestEK::new(1, Some('c' as usize)),
            TestEK::new(1, Some('b' as usize)),
            TestEK::new(-1, Some('d' as usize)),
        ];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy).unwrap();
        let expected = vec![
            TestEK::new(1, Some('c' as usize)),
            TestEK::new(1, Some('b' as usize)),
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn cancel_symmetric_multiple_items() {
        let input = vec![
            TestEK::new(-1, Some('b' as usize)),
            TestEK::new(-1, Some('c' as usize)),
            TestEK::new(1, Some('c' as usize)),
            TestEK::new(1, Some('b' as usize)),
            TestEK::new(1, Some('a' as usize)),
        ];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy).unwrap();
        let expected = vec![TestEK::new(1, Some('a' as usize))];
        assert_eq!(got, expected);
    }

    #[test]
    fn partial_cancellation_leaves_trailing_negative() {
        let input = vec![
            TestEK::new(-1, Some('b' as usize)),
            TestEK::new(-1, Some('c' as usize)),
            TestEK::new(1, Some('c' as usize)),
        ];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy).unwrap();
        assert!(got.is_empty(), "Expected [-1 b] from stage 1, then empty from stage 2");
    }

    #[test]
    fn cancel_fully_and_disappear() {
        let input = vec![TestEK::new(-1, Some('c' as usize)), TestEK::new(1, Some('c' as usize))];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn reduce_pop_counts_with_gaps() {
        let input = vec![
            TestEK::new(-3, Some('b' as usize)),
            TestEK::new(1, Some('c' as usize)),
        ];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy).unwrap();
        assert!(got.is_empty(), "Expected [-2 b] from stage 1, then empty from stage 2");
    }

    #[test]
    fn reduce_pop_counts_multiple_negatives() {
        let input = vec![
            TestEK::new(-1, Some('a' as usize)),
            TestEK::new(-3, Some('b' as usize)),
            TestEK::new(1, Some('c' as usize)),
        ];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy).unwrap();
        assert!(got.is_empty(), "Expected [-1 a, -2 b] from stage 1, then empty from stage 2");
    }

    #[test]
    fn positive_gap_is_wildcard() {
        let input = vec![
            TestEK::new(-1, Some('a' as usize)),
            TestEK::new(-1, Some('b' as usize)),
            TestEK::new(-1, Some('c' as usize)),
            TestEK::new(2, Some('b' as usize)),
        ];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy).unwrap();
        assert!(got.is_empty(), "Expected [-1 a] from stage 1, then empty from stage 2");
    }

    #[test]
    fn negative_gap_is_wildcard() {
        let input = vec![
            TestEK::new(-2, Some('b' as usize)),
            TestEK::new(1, Some('c' as usize)),
            TestEK::new(1, Some('b' as usize)),
            TestEK::new(1, Some('a' as usize)),
        ];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy).unwrap();
        let expected = vec![TestEK::new(1, Some('a' as usize))];
        assert_eq!(got, expected);
    }

    #[test]
    fn strict_mismatch_eliminates_stack() {
        let input = vec![TestEK::new(-1, Some('b' as usize)), TestEK::new(1, Some('c' as usize))];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy);
        assert!(got.is_none());
    }

    #[test]
    fn none_vs_none_cancels_cleanly() {
        let input = vec![TestEK::new(-1, None), TestEK::new(1, None)];
        let got = eliminate_negative_pops_stack(input, intersect_checks_gappy).unwrap();
        assert!(got.is_empty());
    }

    // -----------------------------
    // Graph-level smoke test from complex stack trace (kept, adjusted)
    // -----------------------------
    #[test]
    fn test_graph_from_complex_stack_trace() {
        let god = TestGod::new();
        let n16 = new_node(&god);
        let n18 = new_node(&god);
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

        add_edge(&god, n16, n19, TestEK::new(0, None));
        add_edge(&god, n19, n20, TestEK::new(0, None));
        add_edge(&god, n20, n21, TestEK::new(0, Some(0)));
        add_edge(&god, n21, n23, TestEK::new(0, None));
        add_edge(&god, n23, n25, TestEK::new(-1, Some(1)));
        add_edge(&god, n25, n27, TestEK::new(0, None));
        add_edge(&god, n27, n28, TestEK::new(0, None));
        add_edge(&god, n28, n18, TestEK::new(0, None));
        add_edge(&god, n20, n22, TestEK::new(0, Some(2)));
        add_edge(&god, n22, n24, TestEK::new(2, None));
        add_edge(&god, n24, n26, TestEK::new(0, Some(0)));
        add_edge(&god, n16, n29, TestEK::new(0, None));
        add_edge(&god, n29, n30, TestEK::new(0, None));
        add_edge(&god, n30, n31, TestEK::new(0, Some(1)));
        add_edge(&god, n31, n33, TestEK::new(0, None));
        add_edge(&god, n33, n35, TestEK::new(-1, Some(2)));
        add_edge(&god, n35, n37, TestEK::new(0, None));
        add_edge(&god, n37, n38, TestEK::new(0, None));
        add_edge(&god, n38, n18, TestEK::new(0, None));
        add_edge(&god, n30, n32, TestEK::new(0, Some(2)));
        add_edge(&god, n32, n34, TestEK::new(2, None));
        add_edge(&god, n34, n36, TestEK::new(0, Some(0)));

        let roots = vec![n16];
        let initial_stacks: BTreeSet<Vec<TestEK>> = Trie::get_all_paths(&god, &roots);

        for stack in initial_stacks {
            let transformed = eliminate_negative_pops_stack(stack.clone(), intersect_checks_gappy);
            let transformed = transformed.expect("Reference pipeline eliminated a stack due to mismatch");
            assert!(
                transformed.iter().all(|ek| ek.pop >= 0),
                "Found negative pop after pipeline: {:?}",
                transformed
            );
        }
    }
}
