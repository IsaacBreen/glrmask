// src/constraint_precompute3_eliminate_negative_pops.rs
use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};
use crate::datastructures::EntryApi;
use std::iter::FromIterator;

pub fn eliminate_negative_pops<EK, EV, T, FGet, FMake, FMerge>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    mut get_pop: FGet,
    mut make_key: FMake,
    mut merge_ev: FMerge,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FMake: FnMut(&EK, isize) -> EK,
    FMerge: FnMut(&mut EV, EV),
{
    bubble_up_negative_pops(god, roots, &mut get_pop, &mut make_key, &mut merge_ev);
    neutralize_remaining_negative_pops(god, roots, &mut get_pop, &mut make_key);
}

pub fn bubble_up_negative_pops<EK, EV, T, FGet, FMake, FMerge>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    get_pop: &mut FGet,
    make_key: &mut FMake,
    merge_ev: &mut FMerge,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FMake: FnMut(&EK, isize) -> EK,
    FMerge: FnMut(&mut EV, EV),
{
    todo!()
}

pub fn neutralize_remaining_negative_pops<EK, EV, T, FGet, FMake>(
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    _get_pop: &mut FGet,
    _make_key: &mut FMake,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FMake: FnMut(&EK, isize) -> EK,
{
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PrecomputedNodeContents;
    use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};
    use std::collections::BTreeSet;

    // --- Test Helpers ---
    type TestEK = (isize, usize);
    type TestEV = ();
    type TestT = PrecomputedNodeContents;
    type TestGod = GodWrapper<TestEK, TestEV, TestT>;

    fn new_node(god: &TestGod) -> Trie2Index {
        Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())))
    }

    fn add_edge(god: &TestGod, from: Trie2Index, key: TestEK, to: Trie2Index, val: TestEV) {
        let mut from_w = from.write(god).unwrap();
        from_w
            .children_mut()
            .entry(key)
            .or_default()
            .insert(to, val);
    }

    // --- Type Aliases for Readability ---
    type Path = Vec<TestEK>;
    type PathSet = BTreeSet<Path>;

    // --- Tests ---

    // A simple stack "program" simulator to validate the two-instruction rewrite equivalence.
    //
    // Representation:
    // - The stack is conceptually [0, 1, 2, ..., len-1].
    // - The "dot" (•) position is an integer in [0, len] that marks a slot between elements.
    //   Initially pos = len, matching the doc’s "0 1 2 3 •".
    // - pop k with k >= 0 moves the dot left by k (pos -= k).
    // - pop k with k < 0 moves the dot right by -k (pos += -k).
    // - check x verifies that pos > 0 and the value to the left is x (i.e., x == pos - 1),
    //   because the stack element at index i is the value i.
    //
    // The function returns Some(final_pos) if all checks pass and all positions stay within [0, len].
    // Otherwise returns None.
    fn run_program_with_checks(len: usize, program: &[(isize, Option<usize>)]) -> Option<usize> {
        let mut pos: isize = len as isize;
        let len_is = len as isize;
        for &(pop, check_opt) in program {
            // Apply pop
            if pop >= 0 {
                pos -= pop as isize;
            } else {
                pos += (-pop) as isize;
            }
            // Bounds check
            if pos < 0 || pos > len_is {
                return None;
            }
            // Check (if any)
            if let Some(x) = check_opt {
                if pos == 0 {
                    return None; // no element to the left to check
                }
                let expected_left_val = (pos - 1) as usize;
                if x != expected_left_val {
                    return None;
                }
            }
        }
        Some(pos as usize)
    }

    // Helper to compute the check values x and y that will pass for a given (len, n, m).
    // Returns (x, y) where:
    // - After pop n, check x will pass.
    // - After then pop m, check y will pass.
    //
    // Precondition: positions must remain in-bounds; tests use "safe" pairs.
    fn checks_for_pair(len: usize, n: isize, m: isize) -> Option<(usize, usize)> {
        let mut pos: isize = len as isize;
        // After first pop n
        pos += if n >= 0 { -(n as isize) } else { (-n) as isize };
        if pos <= 0 || pos > len as isize {
            return None;
        }
        let x = (pos - 1) as usize;
        // After second pop m
        pos += if m >= 0 { -(m as isize) } else { (-m) as isize };
        if pos <= 0 || pos > len as isize {
            return None;
        }
        let y = (pos - 1) as usize;
        Some((x, y))
    }

    #[test]
    fn rewrite_equivalence_example_from_doc() {
        // Original:
        // - pop 3, check 0
        // - pop -2, check 2
        let len = 4; // stack: [0,1,2,3], pos starts at 4 ("0 1 2 3 •")
        let program_orig = vec![(3, Some(0)), (-2, Some(2))];

        // Rewritten (as per description):
        // - pop 1, check 2
        // - pop 2, check 0
        // - pop -2 (unconditional)
        let program_rewritten = vec![(1, Some(2)), (2, Some(0)), (-2, None)];

        let end_orig = run_program_with_checks(len, &program_orig)
            .expect("original program should pass");
        let end_rewr = run_program_with_checks(len, &program_rewritten)
            .expect("rewritten program should pass");
        assert_eq!(end_orig, end_rewr, "final positions must match");
    }

    #[test]
    fn rewrite_equivalence_multiple_pairs_sane_bounds() {
        // Validate the transformation:
        //   (pop n, check x); (pop m, check y)
        // becomes
        //   (pop n+m, check y); (pop -m, check x); (pop m) [unconditional]
        //
        // for several "safe" (n, m) pairs where positions stay within [0, len].
        let len = 16; // big enough buffer for the chosen pairs
        let pairs: &[(isize, isize)] = &[
            (3, -2),
            (4, -2),
            (1, -1),
            (5, -3),
        ];
        for &(n, m) in pairs {
            let (x, y) = checks_for_pair(len, n, m)
                .unwrap_or_else(|| panic!("Pair (n={n}, m={m}) out of bounds for len={len}"));
            let original = vec![(n, Some(x)), (m, Some(y))];
            let rewritten = vec![(n + m, Some(y)), (-m, Some(x)), (m, None)];
            let end1 = run_program_with_checks(len, &original)
                .unwrap_or_else(|| panic!("original should pass for (n={n},m={m})"));
            let end2 = run_program_with_checks(len, &rewritten)
                .unwrap_or_else(|| panic!("rewritten should pass for (n={n},m={m})"));
            assert_eq!(
                end1, end2,
                "final positions mismatch for pair (n={n}, m={m})"
            );
        }
    }

    // Spec tests for the graph transformation. These are marked #[ignore]
    // so they won't run until the bubble-up/neutralize functions are implemented.

    // Helper: read all reachable edges’ pop values and assert none are negative.
    fn assert_no_negative_pops_in_graph(god: &TestGod, roots: &[Trie2Index]) {
        let nodes = Trie::<TestEK, TestEV, TestT>::all_nodes(god, roots);
        for idx in nodes {
            let g = idx.read(god).expect("arena read");
            for (ek, dests) in g.children() {
                let pop = ek.0;
                assert!(
                    pop >= 0,
                    "Found negative pop {} on edge key {:?}; graph should be free of negative pops",
                    pop,
                    ek
                );
                assert!(!dests.is_empty(), "edge map should have at least one destination");
            }
        }
    }

    #[ignore]
    #[test]
    fn spec_eliminate_negative_pops_simple_chain() {
        // Build a simple chain A -> B -> C:
        // A --(pop 3, check 0)--> B
        // B --(pop -2, check 2)--> C
        //
        // After elimination, there should be no negative pops anywhere in the reachable subgraph.
        let god: TestGod = GodWrapper::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);

        add_edge(&god, a, (3, 0), b, ());
        add_edge(&god, b, (-2, 2), c, ());

        // Closures for eliminate_negative_pops
        let mut get_pop = |k: &TestEK| k.0;
        let mut make_key = |k: &TestEK, new_pop: isize| (new_pop, k.1);
        let mut merge_ev = |_ev_old: &mut TestEV, _ev_new: TestEV| {};

        eliminate_negative_pops(&god, &[a], &mut get_pop, &mut make_key, &mut merge_ev);

        assert_no_negative_pops_in_graph(&god, &[a]);
    }

    #[ignore]
    #[test]
    fn spec_eliminate_negative_pops_multiple_incoming_edges_to_b() {
        // Troubled pattern to make sure we don't alter B's semantics for other incoming edges:
        //
        //    A --(pop 3, check 0)--> B --(pop -2, check 2)--> C
        //    A --(pop 4, check 1)--> B
        //
        // Expectation (spec):
        // - No negative pops anywhere after elimination.
        // - The path variants from A that previously went through B are preserved in effect,
        //   but with negative pops "bubbled" earlier (possibly via new intermediate nodes),
        //   following the pair rewrite pattern.
        let god: TestGod = GodWrapper::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);

        add_edge(&god, a, (3, 0), b, ());
        add_edge(&god, a, (4, 1), b, ());
        add_edge(&god, b, (-2, 2), c, ());

        let mut get_pop = |k: &TestEK| k.0;
        let mut make_key = |k: &TestEK, new_pop: isize| (new_pop, k.1);
        let mut merge_ev = |_ev_old: &mut TestEV, _ev_new: TestEV| {};

        eliminate_negative_pops(&god, &[a], &mut get_pop, &mut make_key, &mut merge_ev);

        assert_no_negative_pops_in_graph(&god, &[a]);
    }
}
}
