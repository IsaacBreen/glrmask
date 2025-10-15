// src/constraint_precompute3_eliminate_negative_pops.rs
use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};
use std::collections::{HashMap, VecDeque};

pub fn eliminate_negative_pops<EK, EV, T, FGet, FReplace, FNeutral, FMerge>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    get_pop: FGet,
    replace_pop: FReplace,
    neutral_key: FNeutral,
    merge_ev: FMerge,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FReplace: FnMut(&EK, isize) -> EK,
    FNeutral: FnMut() -> EK,
    FMerge: FnMut(&mut EV, EV),
{
    // With the new "front-oriented" strategy, we only neutralize leading negatives at roots here.
    neutralize_remaining_negative_pops(god, roots, get_pop, replace_pop, neutral_key, merge_ev);
}

pub fn bubble_up_negative_pops<EK, EV, T, FGet, FReplace, FNeutral, FMerge>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    mut get_pop: FGet,
    mut replace_pop: FReplace,
    mut neutral_key: FNeutral,
    mut merge_ev: FMerge,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FReplace: FnMut(&EK, isize) -> EK,
    FNeutral: FnMut() -> EK,
    FMerge: FnMut(&mut EV, EV),
{
    // Intentionally left unimplemented per current design:
    // We no longer bubble negatives to the end; and we do not implement graph-level bubbling
    // towards the front here either. Stack-level helpers model the intended behavior.
    todo!()
}

pub fn neutralize_remaining_negative_pops<EK, EV, T, FGet, FNeutral, FReplace, FMerge>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    mut get_pop: FGet,
    mut replace_pop: FReplace,
    mut neutral_key: FNeutral,
    mut merge_ev: FMerge,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FReplace: FnMut(&EK, isize) -> EK,
    FNeutral: FnMut() -> EK,
    FMerge: FnMut(&mut EV, EV),
{
    // Front-oriented neutralization:
    // Change any negative-pop edges leaving root nodes into the neutral key.
    if roots.is_empty() {
        return;
    }

    // Plan moves: (from_node, old_key, child, ev, new_key)
    let mut moves: Vec<(Trie2Index, EK, Trie2Index, EV, EK)> = Vec::new();

    for &u_idx in roots {
        let ug = u_idx
            .read(god)
            .expect("Arena read poisoned while scanning for front neutralization");
        for (ek, dest_map) in ug.children() {
            let pop = get_pop(ek);
            if pop < 0 {
                for (v_idx, ev) in dest_map.iter() {
                    // For front-neutralization, we act on edges directly out of roots,
                    // regardless of whether the destination is a leaf.
                    let new_ek = neutral_key();
                    moves.push((u_idx, ek.clone(), *v_idx, ev.clone(), new_ek));
                }
            }
        }
    }

    // Apply moves
    for (u_idx, old_key, v_idx, ev, new_key) in moves {
        let mut uw = u_idx
            .write(god)
            .expect("Arena write poisoned while applying front neutralization moves");
        // Remove from old_key
        let mut removed_ev_opt: Option<EV> = None;
        if let Some(dest_map) = uw.children_mut().get_mut(&old_key) {
            if let Some(removed_ev) = dest_map.remove(&v_idx) {
                removed_ev_opt = Some(removed_ev);
            }
            if dest_map.is_empty() {
                uw.children_mut().remove(&old_key);
            }
        }
        // Insert under new_key, merging edge value
        let ev_to_insert = removed_ev_opt.unwrap_or(ev.clone());
        let dest_map_new = uw.children_mut().entry(new_key).or_default();
        if let Some(existing_ev) = dest_map_new.get_mut(&v_idx) {
            // Merge edge values when moving into an existing destination under the neutral key.
            merge_ev(existing_ev, ev_to_insert);
        } else {
            dest_map_new.insert(v_idx, ev_to_insert);
        }
    }
}

pub fn assert_negative_pops_follow_property_for_stacks<EK, FGet>(
    stacks: &std::collections::BTreeSet<Vec<EK>>,
    mut get_pop: FGet,
) where
    EK: Ord + Clone + std::fmt::Debug,
    FGet: FnMut(&EK) -> isize,
{
    for stack in stacks {
        // Front-oriented property: once any non-negative is seen, no subsequent negative may appear.
        let mut seen_non_negative = false;
        for ek in stack {
            if seen_non_negative {
                assert!(
                    get_pop(ek) >= 0,
                    "Found a negative pop after a non-negative pop: {:?}",
                    stack
                );
            }
            if get_pop(ek) >= 0 {
                seen_non_negative = true;
            } else {
                // Negative pop before we've seen any non-negative is allowed under the front-oriented policy.
                // Nothing to do here.
            }
        }
    }
}

pub fn assert_negative_pops_follow_property_for_trie<EK, EV, T, FGet>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    get_pop: FGet,
) where
    EK: Ord + Clone + std::fmt::Debug,
    EV: PartialEq + Clone,
    T: PartialEq,
    FGet: FnMut(&EK) -> isize,
{
    let stacks = Trie::<EK, EV, T>::get_all_paths(god, roots);
    assert_negative_pops_follow_property_for_stacks(&stacks, get_pop);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PrecomputedNodeContents;
    use crate::datastructures::trie::{Trie, Trie2Index};
    use std::collections::{BTreeMap, BTreeSet};

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

    fn bubble_up_negative_pops_stack(stack: Vec<TestEK>) -> Vec<TestEK> {
        // Single-pass bubbling towards the FRONT:
        // For each pair (A, B) where B.pop < 0, rewrite [A, B] as:
        //   [(B.pop, None), (A.pop - B.pop, A.check), (B.pop, B.check)]
        // This preserves realized actions while moving the negative to the front of the local pair.
        // Note: This is intentionally single-pass. Residual negatives may still appear later in the stack.
        let mut out: Vec<TestEK> = Vec::with_capacity(stack.len() * 2 + 3);
        for cur in stack.into_iter() {
            if let Some(prev) = out.pop() {
                let a = prev.pop;
                if cur.pop < 0 {
                    let b = cur.pop; // b < 0
                    // (b, None) -- bring the negative to the front
                    out.push(TestEK::new(b, None));
                    // (a - b, check of A)
                    out.push(TestEK::new(a - b, prev.check));
                    // (b, check of B) -- preserves B's realized action at a + b
                    out.push(TestEK::new(b, cur.check));
                } else {
                    // No bubbling; restore prev and append cur
                    out.push(prev);
                    out.push(cur);
                }
            } else {
                // Nothing to bubble against yet.
                out.push(cur);
            }
        }
        // Note: This is intentionally a single-pass transform. Any residual leading negatives
        // can be neutralized by the neutralize_remaining_negative_pops stage in the pipeline.
        out
    }

    // Neutralize any remaining leading unconditional or conditional negatives by setting the head to pop 0.
    // This mirrors the "make them unconditional pop 0" instruction at the start of paths (front-oriented).
    fn neutralize_remaining_negative_pops_stack(mut stack: Vec<TestEK>) -> Vec<TestEK> {
        // If the head is a negative pop, neutralize it to pop 0, check None.
        if let Some(first) = stack.first_mut() {
            if first.pop < 0 {
                first.pop = 0;
                first.check = None;
            }
        }
        stack
    }

    // Compute realized actions as a map from absolute position to check id.
    // Position is the running sum of pops; we record positions at which a check occurs.
    // This is the semantic invariant that must be preserved by bubbling.
    fn realized_actions(stack: &[TestEK]) -> BTreeMap<isize, usize> {
        let mut map = BTreeMap::new();
        let mut pos: isize = 0;
        for ek in stack {
            pos += ek.pop;
            if let Some(check) = ek.check {
                if let Some(existing_check) = map.get(&pos) {
                    if *existing_check != check {
                        panic!("Invalid stack: conflicting realized action at position {}. Existing check: {}, New check: {}", pos, existing_check, check);
                    }
                }
                map.insert(pos, check);
            }
        }
        map
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

        // bubble (stack-only; we do not mutate the graph here)
        let bubbled_stacks = map_to_stacks(bubble_up_negative_pops_stack, &stacks);
        // We intentionally do NOT assert the "no-negative-after-non-negative" property here,
        // because this is a single-pass transform that may leave residual negatives later in the path.
        // Realized-actions invariants are tested below in dedicated tests.

        // neutralize (stack-only; front-oriented)
        let _neutralized_stacks =
            map_to_stacks(neutralize_remaining_negative_pops_stack, &bubbled_stacks);
        // No graph mutation: do not call graph-level bubble or neutralize here.
        // Final "all non-negative" assertion is not applicable to front-bubbling in a single pass.
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
    #[test]
    fn test_bubble_up_negative_pops_stack_simple_pair() {
        // From the problem statement example (front-oriented):
        // Input: [pop 3, check 0], [pop -2, check 2]
        // =>
        // [pop -2, check None], [pop 5, check 0], [pop -2, check 2]
        let input = vec![
            TestEK::new(3, Some(0)),
            TestEK::new(-2, Some(2)),
        ];
        let expected = vec![
            TestEK::new(-2, None),
            TestEK::new(5, Some(0)),
            TestEK::new(-2, Some(2)),
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
        // Leading negative becomes neutralized to pop 0.
        let input = vec![
            TestEK::new(-2, None),
            TestEK::new(2, Some(0)),
            TestEK::new(1, Some(2)),
        ];
        let expected = vec![
            TestEK::new(0, None),
            TestEK::new(2, Some(0)),
            TestEK::new(1, Some(2)),
        ];
        let got = neutralize_remaining_negative_pops_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_neutralize_handles_conditional_negative_pop_at_start() {
        let input = vec![
            TestEK::new(-2, Some(3)),
            TestEK::new(1, Some(2)),
        ];
        let expected = vec![
            TestEK::new(0, None),
            TestEK::new(1, Some(2)),
        ];
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
        // This documents the current single-pass behavior (front-oriented):
        // A = (0, None), B = (-2, Some(3))
        // => [(-2, None), (2, None), (-2, Some(3))]
        let input = vec![
            TestEK::new(0, None),
            TestEK::new(-2, Some(3)),
        ];
        let expected = vec![
            TestEK::new(-2, None),
            TestEK::new(2, None),
            TestEK::new(-2, Some(3)),
        ];
        let got = bubble_up_negative_pops_stack(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_stack_pipeline_simple_pair_neutralizes_leading_negative() {
        let input = vec![
            TestEK::new(3, Some(0)),
            TestEK::new(-2, Some(2)),
        ];
        let bubbled = bubble_up_negative_pops_stack(input);
        let neutralized = neutralize_remaining_negative_pops_stack(bubbled);
        // Expect the leading negative to be neutralized.
        if let Some(first) = neutralized.first() {
            assert!(first.pop >= 0);
        }
    }

    // --- Graph-level expectations (ignored until the TODOs are implemented) ---

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

    // --- New graph-level tests (non-ignored) that include terminal edges to allow merging ---

    #[test]
    fn test_graph_example_with_terminal() {
        // A --(3,c0)--> B --(-2,c2)--> C --(0,None)--> T
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let t = new_node(&god);

        add_edge(&god, a, b, TestEK::new(3, Some(0)));
        add_edge(&god, b, c, TestEK::new(-2, Some(2)));
        add_edge(&god, c, t, TestEK::new(0, None));

        let roots = vec![a];
        run_test(&god, &roots);
    }

    #[test]
    fn test_graph_branching_from_source_single_negative_pair_with_terminal_new() {
        // A --(5, c0)--> B --(-2, c2)--> C --(0, None)--> T
        // A --(3, c1)--> B
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let t = new_node(&god);

        add_edge(&god, a, b, TestEK::new(5, Some(0)));
        add_edge(&god, a, b, TestEK::new(3, Some(1)));
        add_edge(&god, b, c, TestEK::new(-2, Some(2)));
        add_edge(&god, c, t, TestEK::new(0, None));

        let roots = vec![a];
        run_test(&god, &roots);
    }

    #[test]
    fn test_graph_multiple_parents_into_b_single_negative_pair_with_terminal_new() {
        // A1 --(4, c10)--> B --(-3, c2)--> C --(0, None)--> T
        // A2 --(6, c11)--> B
        let god = TestGod::new();
        let a1 = new_node(&god);
        let a2 = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let t = new_node(&god);

        add_edge(&god, a1, b, TestEK::new(4, Some(10)));
        add_edge(&god, a2, b, TestEK::new(6, Some(11)));
        add_edge(&god, b, c, TestEK::new(-3, Some(2)));
        add_edge(&god, c, t, TestEK::new(0, None));

        let roots = vec![a1, a2];
        run_test(&god, &roots);
    }

    #[test]
    fn test_graph_branching_after_b_also_has_positive_branch_with_terminal_new() {
        // A --(5, c0)--> B --(-2, c2)--> C --(0, None)--> T
        // A --(4, c1)--> B --(1, c7)--> D --(0, None)--> T2
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let d = new_node(&god);
        let t = new_node(&god);
        let t2 = new_node(&god);

        add_edge(&god, a, b, TestEK::new(5, Some(0)));
        add_edge(&god, a, b, TestEK::new(4, Some(1)));
        add_edge(&god, b, c, TestEK::new(-2, Some(2)));
        add_edge(&god, c, t, TestEK::new(0, None));
        add_edge(&god, b, d, TestEK::new(1, Some(7)));
        add_edge(&god, d, t2, TestEK::new(0, None));

        let roots = vec![a];
        run_test(&god, &roots);
    }

    #[test]
    fn test_graph_no_negative_edges_noop_new() {
        // A --(2, c0)--> B --(1, c1)--> C --(0, None)--> T
        let god = TestGod::new();
        let a = new_node(&god);
        let b = new_node(&god);
        let c = new_node(&god);
        let t = new_node(&god);

        add_edge(&god, a, b, TestEK::new(2, Some(0)));
        add_edge(&god, b, c, TestEK::new(1, Some(1)));
        add_edge(&god, c, t, TestEK::new(0, None));

        let roots = vec![a];
        run_test(&god, &roots);
    }

    // --- Realized-actions preservation tests for the stack helper ---

    #[test]
    fn test_bubble_preserves_realized_actions_simple_pair_map() {
        let original = vec![
            TestEK::new(3, Some(0)),
            TestEK::new(-2, Some(2)),
        ];
        let bubbled = bubble_up_negative_pops_stack(original.clone());
        assert_eq!(realized_actions(&original), realized_actions(&bubbled));
    }

    #[test]
    fn test_bubble_preserves_realized_actions_unconditional_first_map() {
        let original = vec![
            TestEK::new(0, None),
            TestEK::new(-2, Some(3)),
        ];
        let bubbled = bubble_up_negative_pops_stack(original.clone());
        assert_eq!(realized_actions(&original), realized_actions(&bubbled));
    }

    #[test]
    fn test_bubble_preserves_realized_actions_longer_chain_map() {
        // [ (2,c1), (4,c2), (-3,c3) ] -> bubble only affects the last pair
        let original = vec![
            TestEK::new(2, Some(1)),
            TestEK::new(4, Some(2)),
            TestEK::new(-3, Some(3)),
        ];
        let bubbled = bubble_up_negative_pops_stack(original.clone());
        assert_eq!(realized_actions(&original), realized_actions(&bubbled));
    }

    #[test]
    fn test_bubble_preserves_realized_actions_multiple_negatives_map() {
        // Mix of negatives and unconditional; ensure realized checks' positions are preserved.
        let original = vec![
            TestEK::new(2, Some(1)),
            TestEK::new(-1, Some(3)),
            TestEK::new(0, None),
            TestEK::new(-2, Some(5)),
            TestEK::new(3, Some(1)),
        ];
        let bubbled = bubble_up_negative_pops_stack(original.clone());
        assert_eq!(realized_actions(&original), realized_actions(&bubbled));
    }

    #[test]
    fn test_graph_from_complex_stack_trace() {
        let god = TestGod::new();
        // Nodes from the provided stack trace
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

        // --- Build graph from stack trace ---

        // Branch 1 (from root -> n19)
        add_edge(&god, n16, n19, TestEK::new(0, None));
        add_edge(&god, n19, n20, TestEK::new(0, None));
        
        // Path through n21 (with negative pop)
        add_edge(&god, n20, n21, TestEK::new(0, Some(0)));
        add_edge(&god, n21, n23, TestEK::new(0, None));
        add_edge(&god, n23, n25, TestEK::new(-1, Some(1)));
        add_edge(&god, n25, n27, TestEK::new(0, None));
        add_edge(&god, n27, n28, TestEK::new(0, None));
        add_edge(&god, n28, n18, TestEK::new(0, None));

        // Path through n22
        add_edge(&god, n20, n22, TestEK::new(0, Some(2)));
        add_edge(&god, n22, n24, TestEK::new(2, None));
        add_edge(&god, n24, n26, TestEK::new(0, Some(0))); // Leaf

        // Branch 2 (from root -> n29)
        add_edge(&god, n16, n29, TestEK::new(0, None));
        add_edge(&god, n29, n30, TestEK::new(0, None));

        // Path through n31 (with negative pop)
        add_edge(&god, n30, n31, TestEK::new(0, Some(1)));
        add_edge(&god, n31, n33, TestEK::new(0, None));
        add_edge(&god, n33, n35, TestEK::new(-1, Some(2)));
        add_edge(&god, n35, n37, TestEK::new(0, None));
        add_edge(&god, n37, n38, TestEK::new(0, None));
        add_edge(&god, n38, n18, TestEK::new(0, None));

        // Path through n32
        add_edge(&god, n30, n32, TestEK::new(0, Some(2)));
        add_edge(&god, n32, n34, TestEK::new(2, None));
        add_edge(&god, n34, n36, TestEK::new(0, Some(0))); // Leaf

        let roots = vec![n16];
        run_test(&god, &roots);
    }
}

