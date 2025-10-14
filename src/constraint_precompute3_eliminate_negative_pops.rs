// src/constraint_precompute3_eliminate_negative_pops.rs
use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};

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
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    _get_pop: FGet,
    _make_key: FMake,
    _merge_ev: FMerge,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FMake: FnMut(&EK, isize) -> EK,
    FMerge: FnMut(&mut EV, EV),
{
    todo!("bubble_up_negative_pops (graph version) not implemented yet");
}

pub fn neutralize_remaining_negative_pops<EK, EV, T, FGet, FMake>(
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    _get_pop: FGet,
    _make_key: FMake,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FMake: FnMut(&EK, isize) -> EK,
{
    todo!("neutralize_remaining_negative_pops (graph version) not implemented yet");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PrecomputedNodeContents;
    use crate::datastructures::trie::{Trie, Trie2Index};
    use std::collections::BTreeSet;

    // --- Test Helpers ---

    type TestEV = ();
    type TestT = PrecomputedNodeContents;
    type TestGod = GodWrapper<TestEK, TestEV, TestT>;

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

    // Canonicalize a stack of edges:
    // - If an unconditional pop is followed by any other pop, they are merged.
    // - Remove unconditional no-ops (pop == 0, check == None).
    fn compress_stack(stack: Vec<TestEK>) -> Vec<TestEK> {
        let mut out: Vec<TestEK> = Vec::new();
        for ek in stack {
            if let Some(last) = out.last_mut() {
                if last.check.is_none() {
                    // If the last operation is an unconditional pop, we can merge it
                    // with the next operation.
                    last.pop += ek.pop;
                    last.check = ek.check;
                    continue;
                }
            }
            out.push(ek);
        }
        out.into_iter()
            .filter(|ek| !(ek.check.is_none() && ek.pop == 0))
            .collect()
    }

    // Apply the pairwise transformation when we see any negative pop as the second in a pair:
    // Given:
    //   - pop n, check x
    //   - pop m, check y   (with m < 0)
    // Replace these two with:
    //   - pop n+m, check y
    //   - pop -m, check x
    //   - pop m, check None
    //
    // This moves the negative "m" to be a trailing unconditional pop directly after
    // the transformed pair. This function applies this transform once for each occurrence
    // where a negative pop appears as the second in a pair while scanning left-to-right.
    fn bubble_up_negative_pops_stack(mut stack: Vec<TestEK>) -> Vec<TestEK> {
        if stack.len() < 2 {
            return stack;
        }
        let mut out: Vec<TestEK> = Vec::new();
        let mut i = 0;
        while i < stack.len() {
            if i + 1 < stack.len() && stack[i + 1].pop < 0 {
                let a = stack[i];
                let b = stack[i + 1];
                // A': pop n+m, check y
                let first = TestEK::new(a.pop + b.pop, b.check);
                // B': pop -m, check x
                let second = TestEK::new(-b.pop, a.check);
                // C': pop m, check None
                let third = TestEK::new(b.pop, None);

                out.push(first);
                out.push(second);
                out.push(third);
                i += 2; // consumed i and i+1
            } else {
                out.push(stack[i]);
                i += 1;
            }
        }
        out
    }

    // Neutralize any remaining trailing unconditional pops by setting them to pop 0.
    // This mirrors the "make them unconditional pop 0" instruction at the end of paths.
    fn neutralize_remaining_negative_pops_stack(mut stack: Vec<TestEK>) -> Vec<TestEK> {
        // If the tail consists of unconditional pops, neutralize them to a single pop 0 None.
        if let Some(last) = stack.last_mut() {
            if last.check.is_none() && last.pop != 0 {
                last.pop = 0;
            }
        }
        stack
    }

    // --- Trie Construction Helpers ---

    fn new_node(god: &TestGod) -> Trie2Index {
        Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())))
    }

    fn add_edge(god: &TestGod, from: Trie2Index, to: Trie2Index, key: TestEK) {
        let mut w = from
            .write(god)
            .expect("Arena write poisoned while adding edge");
        let mut ev: Option<TestEV> = Some(());
        w.try_insert(key, &mut ev, to);
    }

    fn flatten_trie_to_stacks(god: &TestGod, roots: &[Trie2Index]) -> BTreeSet<Vec<TestEK>> {
        Trie::<TestEK, TestEV, TestT>::get_all_paths(god, roots)
    }

    fn map_to_stacks(
        f: impl Fn(Vec<TestEK>) -> Vec<TestEK>,
        stacks: &BTreeSet<Vec<TestEK>>,
    ) -> BTreeSet<Vec<TestEK>> {
        stacks.iter().map(|s| f(s.clone())).collect()
    }

    // Keep this around for future integration tests; mark as ignored for now
    // because the graph versions are not implemented yet.
    fn run_test(god: &TestGod, roots: &[Trie2Index]) {
        let stacks = flatten_trie_to_stacks(god, roots);

        // bubble
        let bubbled_stacks = map_to_stacks(bubble_up_negative_pops_stack, &stacks);
        bubble_up_negative_pops(
            god,
            roots,
            |ek| ek.pop,
            |ek, new_pop| TestEK::new(new_pop, ek.check),
            |ev1, _ev2| *ev1 = (),
        );
        let bubbled_trie_flattened = flatten_trie_to_stacks(god, roots);
        assert_eq!(
            map_to_stacks(compress_stack, &bubbled_stacks),
            map_to_stacks(compress_stack, &bubbled_trie_flattened)
        );

        // neutralize
        let neutralized_stacks =
            map_to_stacks(neutralize_remaining_negative_pops_stack, &bubbled_stacks);
        neutralize_remaining_negative_pops(
            god,
            roots,
            |ek| ek.pop,
            |ek, new_pop| TestEK::new(new_pop, ek.check),
        );
        let neutralized_trie_flattened = flatten_trie_to_stacks(god, roots);
        assert_eq!(
            map_to_stacks(compress_stack, &neutralized_stacks),
            map_to_stacks(compress_stack, &neutralized_trie_flattened)
        );

        // final check
        let final_stacks = neutralized_trie_flattened;
        for stack in final_stacks {
            for ek in &stack {
                assert!(ek.pop >= 0);
            }
        }
    }

    // --- Unit tests for the stack helpers (explicit outputs) ---

    #[test]
    fn test_compress_stack_merges_unconditional_and_drops_zero() {
        let input = vec![
            TestEK::new(3, None),
            TestEK::new(-2, None),
            TestEK::new(2, None),
            TestEK::new(0, None),
            TestEK::new(1, None),
        ];
        let expected = vec![TestEK::new(4, None)];
        let got = compress_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_compress_stack_respects_checks() {
        let input = vec![
            TestEK::new(1, Some(2)),
            TestEK::new(2, None),
            TestEK::new(3, Some(0)),
            TestEK::new(4, None),
            TestEK::new(-1, None),
        ];
        // Unconditional pops are merged with subsequent pops.
        let expected = vec![
            TestEK::new(1, Some(2)),
            TestEK::new(5, Some(0)), // (2, None) + (3, Some(0))
            TestEK::new(3, None),    // (4, None) + (-1, None)
        ];
        let got = compress_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_bubble_up_negative_pops_stack_simple_pair() {
        // From the problem statement example:
        // [pop 3, check 0], [pop -2, check 2]
        // =>
        // [pop 1, check 2], [pop 2, check 0], [pop -2, check None]
        let input = vec![
            TestEK::new(3, Some(0)),
            TestEK::new(-2, Some(2)),
        ];
        let expected = vec![
            TestEK::new(1, Some(2)),
            TestEK::new(2, Some(0)),
            TestEK::new(-2, None),
        ];
        let got = bubble_up_negative_pops_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_bubble_up_negative_pops_stack_when_second_non_negative_no_change() {
        let input = vec![
            TestEK::new(2, Some(1)),
            TestEK::new(3, Some(2)),
        ];
        let expected = input.clone();
        let got = bubble_up_negative_pops_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_neutralize_remaining_negative_pops_stack_simple() {
        let input = vec![
            TestEK::new(1, Some(2)),
            TestEK::new(2, Some(0)),
            TestEK::new(-2, None),
        ];
        // Neutralize trailing unconditional pop:
        let expected = vec![
            TestEK::new(1, Some(2)),
            TestEK::new(2, Some(0)),
            TestEK::new(0, None),
        ];
        let got = neutralize_remaining_negative_pops_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_neutralize_does_nothing_if_trailing_is_not_unconditional() {
        let input = vec![
            TestEK::new(1, Some(2)),
            TestEK::new(-2, Some(3)),
        ];
        let expected = input.clone();
        let got = neutralize_remaining_negative_pops_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_compress_chain_of_unconditional_pops() {
        let input = vec![
            TestEK::new(3, None),
            TestEK::new(-2, None),
            TestEK::new(1, None),
            TestEK::new(0, None),
        ];
        let expected = vec![TestEK::new(2, None)];
        let got = compress_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_bubble_with_unconditional_first_allows_negative_first() {
        // This documents the current single-pass behavior:
        // A = (0, None), B = (-2, Some(3))
        // => [(-2, Some(3)), (2, None), (-2, None)]
        let input = vec![
            TestEK::new(0, None),
            TestEK::new(-2, Some(3)),
        ];
        let expected = vec![
            TestEK::new(-2, Some(3)),
            TestEK::new(2, None),
            TestEK::new(-2, None),
        ];
        let got = bubble_up_negative_pops_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_stack_pipeline_simple_pair_yields_non_negative_after_neutralize() {
        let input = vec![
            TestEK::new(3, Some(0)),
            TestEK::new(-2, Some(2)),
        ];
        let bubbled = bubble_up_negative_pops_stack(input);
        let neutralized = neutralize_remaining_negative_pops_stack(bubbled);
        // Expect the last to be neutralized and all non-negative pops.
        for ek in neutralized {
            assert!(ek.pop >= 0);
        }
    }

    // --- Graph-level expectations (ignored until the TODOs are implemented) ---

    #[ignore = "Graph-level negative-pop elimination not implemented yet"]
    #[test]
    fn test_example() {
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        add_edge(&god, a, b, TestEK::new(3, Some(0)));
        add_edge(&god, b, c, TestEK::new(-2, Some(2)));
        let roots = vec![a];
        run_test(&god, &roots)
    }

    #[ignore = "Graph-level negative-pop elimination not implemented yet"]
    #[test]
    fn test_graph_branching_from_source_single_negative_pair() {
        // A --(5, c0)--> B --(-2, c2)--> C
        // A --(3, c1)--> B
        //
        // Two distinct paths share the middle node B.
        // The transformation must not mutate B's semantics for other incoming edges.
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);

        // Branching edges into B from A
        add_edge(&god, a, b, TestEK::new(5, Some(0)));
        add_edge(&god, a, b, TestEK::new(3, Some(1)));
        // Negative second in the pair
        add_edge(&god, b, c, TestEK::new(-2, Some(2)));

        let roots = vec![a];
        run_test(&god, &roots);
    }

    #[ignore = "Graph-level negative-pop elimination not implemented yet"]
    #[test]
    fn test_graph_multiple_parents_into_b_single_negative_pair() {
        // A1 --(4, c10)--> B --(-3, c2)--> C
        // A2 --(6, c11)--> B
        //
        // Roots are A1 and A2; ensure transformations for one incoming edge don't break the other.
        let god = TestGod::new();
        let a1 = new_node(&god);
        let a2 = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);

        add_edge(&god, a1, b, TestEK::new(4, Some(10)));
        add_edge(&god, a2, b, TestEK::new(6, Some(11)));
        add_edge(&god, b, c, TestEK::new(-3, Some(2)));

        let roots = vec![a1, a2];
        run_test(&god, &roots);
    }

    #[ignore = "Graph-level negative-pop elimination not implemented yet"]
    #[test]
    fn test_graph_branching_after_b_also_has_positive_branch() {
        // A --(5, c0)--> B --(-2, c2)--> C
        // A --(4, c1)--> B --(1, c7)--> D
        //
        // Verifies that non-negative branches out of B remain unaffected while negative pairs get transformed.
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let d = new_node(&god);

        add_edge(&god, a, b, TestEK::new(5, Some(0)));
        add_edge(&god, a, b, TestEK::new(4, Some(1)));
        add_edge(&god, b, c, TestEK::new(-2, Some(2)));
        add_edge(&god, b, d, TestEK::new(1, Some(7)));

        let roots = vec![a];
        run_test(&god, &roots);
    }

    #[ignore = "Graph-level negative-pop elimination not implemented yet"]
    #[test]
    fn test_graph_no_negative_edges_noop() {
        // A --(2, c0)--> B --(1, c1)--> C
        // No negative pops; transformation should be a no-op (up to compression).
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);

        add_edge(&god, a, b, TestEK::new(2, Some(0)));
        add_edge(&god, b, c, TestEK::new(1, Some(1)));

        let roots = vec![a];
        run_test(&god, &roots);
    }

    #[ignore = "Graph-level negative-pop elimination not implemented yet"]
    #[test]
    fn test_graph_trailing_unconditional_negative_neutralized() {
        // A --(3, c0)--> B --(-2, c2)--> C --(0, None)--> terminal
        // The bubble will make the negative trailing and unconditional; neutralization should set it to zero.
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let t = new_node(&god);

        add_edge(&god, a, b, TestEK::new(3, Some(0)));
        add_edge(&god, b, c, TestEK::new(-2, Some(2)));
        // This edge doesn't change the "negative-trailing" property, but is included to model a terminal no-op branch.
        add_edge(&god, c, t, TestEK::new(0, None));

        let roots = vec![a];
        run_test(&god, &roots);
    }
}
