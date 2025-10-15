// src/constraint_precompute3_eliminate_negative_pops.rs
use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};
use std::collections::{HashMap, VecDeque};

pub fn eliminate_negative_pops<EK, EV, T, FGet, FReplace, FNeutral, FMerge>(
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
    bubble_up_negative_pops(god, roots, &mut get_pop, &mut replace_pop, &mut neutral_key, &mut merge_ev);
    neutralize_remaining_negative_pops(god, roots, &mut get_pop, &mut replace_pop, &mut neutral_key, &mut merge_ev);
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
    // Single-pass "bubble" transform over a snapshot of the reachable subgraph.
    // For each edge U --(A)--> V where V has some negative-pop edge(s) B,
    // we rewrite only the U->V context:
    //   [A, B]  ==>  [A+B (check of B), -B (check of A), B (check None)]
    // without disturbing other parents of V.
    // Implementation:
    //  - Remove U --(A)--> V
    //  - Add:
    //      U --(A)--> V_nonneg (clone of V containing only non-negative children), if any
    //      For each negative child V --(B)-> W:
    //         let A = get_pop(A), B = get_pop(B) (B < 0)
    //         ek1 = replace_pop(&B_key, A + B)        // takes B's "check"
    //         ek2 = replace_pop(&A_key, -B)           // takes A's "check"
    //         ek3 = replace_pop(&neutral_key(), B)    // takes neutral "check" (= None), pop negative
    //         Create V' and N, then:
    //            U --(ek1)--> V'
    //            V' --(ek2)--> N
    //            N --(ek3)--> W
    //  - Edge values (EV):
    //      U->V_nonneg uses original EV (U->V).
    //      U->V' uses original EV (U->V).
    //      V'->N uses original EV (U->V).
    //      N->W uses EV from (V->W).
    //
    // Note: We collect all transformations first, then apply them. We avoid holding any Arena
    // lock across other Arena operations to prevent deadlocks (RwLock is not re-entrant).

    // Helper structs to carry a plan we can apply after scanning
    struct NegEdge<EK, EV> {
        ek_b: EK,
        b: isize,
        child: Trie2Index,
        ev_bc: EV,
    }
    struct Task<EK, EV> {
        u: Trie2Index,
        ek_a: EK,
        v: Trie2Index,
        ev_ab: EV,
        a: isize,
        negs: Vec<NegEdge<EK, EV>>,
        nonneg: Vec<(EK, Vec<(Trie2Index, EV)>)>,
    }

    // 1) Snapshot of all tasks
    let reachable = Trie::<EK, EV, T>::all_nodes(god, roots);
    let mut tasks: Vec<Task<EK, EV>> = Vec::new();

    for &u_idx in &reachable {
        if let Some(ug) = u_idx.read(god) {
            for (ek_a, dest_map_a) in ug.children() {
                let a = get_pop(ek_a);
                for (v_idx, ev_ab) in dest_map_a.iter() {
                    if let Some(vg) = v_idx.read(god) {
                        let mut negs: Vec<NegEdge<EK, EV>> = Vec::new();
                        let mut nonneg: Vec<(EK, Vec<(Trie2Index, EV)>)> = Vec::new();

                        for (ek_b, dest_map_b) in vg.children() {
                            let b = get_pop(ek_b);
                            if b < 0 {
                                for (w_idx, ev_bc) in dest_map_b.iter() {
                                    negs.push(NegEdge {
                                        ek_b: ek_b.clone(),
                                        b,
                                        child: *w_idx,
                                        ev_bc: ev_bc.clone(),
                                    });
                                }
                            } else {
                                let dests: Vec<(Trie2Index, EV)> = dest_map_b
                                    .iter()
                                    .map(|(ci, e)| (*ci, e.clone()))
                                    .collect();
                                if !dests.is_empty() {
                                    nonneg.push((ek_b.clone(), dests));
                                }
                            }
                        }

                        if !negs.is_empty() {
                            tasks.push(Task {
                                u: u_idx,
                                ek_a: ek_a.clone(),
                                v: *v_idx,
                                ev_ab: ev_ab.clone(),
                                a,
                                negs,
                                nonneg,
                            });
                        }
                    }
                }
            }
        }
    }

    // 2) Apply tasks
    for task in tasks {
        // Read V's value once (no locks held during later writes to U/new nodes)
        let v_value = {
            let vg = task
                .v
                .read(god)
                .expect("Arena read poisoned while cloning V's value for bubble");
            vg.value.clone()
        };

        // Prepare a clone that keeps only non-negative children (if any)
        let mut nonneg_clone_idx: Option<Trie2Index> = None;
        if !task.nonneg.is_empty() {
            let idx = Trie2Index::new(god.insert(Trie::new(v_value.clone())));
            {
                let mut w = idx
                    .write(god)
                    .expect("Arena write poisoned while populating V_nonneg clone");
                for (ek_k, dests) in &task.nonneg {
                    let dm = w.children_mut().entry(ek_k.clone()).or_default();
                    for (child, ev) in dests {
                        dm.insert(*child, ev.clone());
                    }
                }
            }
            nonneg_clone_idx = Some(idx);
        }

        // For each negative child, create a dedicated bubbled path
        let mut neg_insertions: Vec<(EK, Trie2Index)> = Vec::new(); // (ek1, v_prime)
        for neg in &task.negs {
            // ek1 takes B's "check"; ek2 takes A's "check"; ek3 uses neutral "check" (None) with pop B
            let ek1 = replace_pop(&neg.ek_b, task.a + neg.b);
            let ek2 = replace_pop(&task.ek_a, -neg.b);
            let ek3 = {
                let base = neutral_key();
                replace_pop(&base, neg.b)
            };

            // Create V' and intermediate N
            let v_prime_idx = Trie2Index::new(god.insert(Trie::new(v_value.clone())));
            let n_idx = Trie2Index::new(god.insert(Trie::new(v_value.clone())));

            // V' --(ek2)--> N with EV from U->V (ev_ab)
            {
                let mut vprime_w = v_prime_idx
                    .write(god)
                    .expect("Arena write poisoned while wiring V' in bubble");
                vprime_w
                    .children_mut()
                    .entry(ek2)
                    .or_default()
                    .insert(n_idx, task.ev_ab.clone());
            }

            // N --(ek3)--> W with EV from V->W (ev_bc)
            {
                let mut n_w = n_idx
                    .write(god)
                    .expect("Arena write poisoned while wiring intermediate N in bubble");
                n_w
                    .children_mut()
                    .entry(ek3)
                    .or_default()
                    .insert(neg.child, neg.ev_bc.clone());
            }

            // Later we will add U --(ek1)--> V'
            neg_insertions.push((ek1, v_prime_idx));
        }

        // Update U: remove (ek_a -> V), then insert:
        //   U --(ek_a)--> V_nonneg (if any)
        //   U --(ek1)--> V' for each negative child
        {
            let mut uw = task
                .u
                .write(god)
                .expect("Arena write poisoned while updating parent U in bubble");

            // Remove (ek_a -> V), remembering its EV
            let mut ev_ab_to_use = task.ev_ab.clone();
            if let Some(dm) = uw.children_mut().get_mut(&task.ek_a) {
                if let Some(ev_removed) = dm.remove(&task.v) {
                    ev_ab_to_use = ev_removed;
                }
                if dm.is_empty() {
                    uw.children_mut().remove(&task.ek_a);
                }
            }

            // Insert non-negative-preserving branch
            if let Some(v_nonneg_idx) = nonneg_clone_idx {
                let dm = uw.children_mut().entry(task.ek_a.clone()).or_default();
                if let Some(existing_ev) = dm.get_mut(&v_nonneg_idx) {
                    merge_ev(existing_ev, ev_ab_to_use.clone());
                } else {
                    dm.insert(v_nonneg_idx, ev_ab_to_use.clone());
                }
            }

            // Insert bubbled branches for each negative child
            for (ek1, v_prime_idx) in neg_insertions {
                let dm = uw.children_mut().entry(ek1).or_default();
                if let Some(existing_ev) = dm.get_mut(&v_prime_idx) {
                    merge_ev(existing_ev, ev_ab_to_use.clone());
                } else {
                    dm.insert(v_prime_idx, ev_ab_to_use.clone());
                }
            }
        }
    }
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
    let reachable = Trie::<EK, EV, T>::all_nodes(god, roots);
    if reachable.is_empty() {
        return;
    }

    // Plan moves: (from_node, old_key, child, ev, new_key)
    let mut moves: Vec<(Trie2Index, EK, Trie2Index, EV, EK)> = Vec::new();

    for &u_idx in &reachable {
        let ug = u_idx
            .read(god)
            .expect("Arena read poisoned while scanning for neutralization");
        for (ek, dest_map) in ug.children() {
            let pop = get_pop(ek);
            if pop < 0 {
                for (v_idx, ev) in dest_map.iter() {
                    let is_leaf = {
                        let vg = v_idx.read(god).expect("Arena read poisoned");
                        vg.is_leaf()
                    };
                    if is_leaf {
                        let new_ek = neutral_key();
                        moves.push((u_idx, ek.clone(), *v_idx, ev.clone(), new_ek));
                    }
                }
            }
        }
    }

    // Apply moves
    for (u_idx, old_key, v_idx, ev, new_key) in moves {
        let mut uw = u_idx
            .write(god)
            .expect("Arena write poisoned while applying neutralization moves");
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
            // We cannot call merge_ev here (not passed), but we retained EV from removal; if needed, caller
            // supplied bubble-up stage merges. For neutralization phase, preserve existing EV.
            // To be conservative, we prefer the existing EV over ev_to_insert (do nothing).
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
        let mut seen_negative = false;
        for ek in stack {
            if seen_negative {
                assert!(
                    get_pop(ek) <= 0,
                    "Found a positive pop after a negative pop: {:?}",
                    stack
                );
            }
            if get_pop(ek) < 0 {
                seen_negative = true;
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
    use crate::trie_test_framework::TrieTestFramework;
    use std::collections::{BTreeMap, BTreeSet};

    // --- Test Harness Types and Config ---

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

    // --- Stack-based Reference Implementations ---

    /// Canonicalize a stack of edges:
    /// - If an unconditional pop is followed by any other pop, they are merged.
    /// - Remove unconditional no-ops (pop == 0, check == None).
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

    /// Reference implementation for the "bubble up" transformation.
    fn bubble_up_negative_pops_stack(stack: Vec<TestEK>) -> Vec<TestEK> {
        // Single-pass bubbling:
        // For each pair (A, B) where B.pop < 0, rewrite [A, B] as:
        //   [(A.pop + B.pop, B.check), (-B.pop, A.check), (B.pop, None)]
        // This preserves realized actions while moving the negative to the end of the local pair.
        let mut out: Vec<TestEK> = Vec::with_capacity(stack.len() * 2 + 3);
        for cur in stack.into_iter() {
            if let Some(prev) = out.pop() {
                if cur.pop < 0 {
                    let a = prev.pop;
                    let b = cur.pop; // b < 0
                    // (a + b, check of B)
                    out.push(TestEK::new(a + b, cur.check));
                    // (-b, check of A)
                    out.push(TestEK::new(-b, prev.check));
                    // (b, None) -- the residual negative to the pair's end
                    out.push(TestEK::new(b, None));
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
        // Note: This is intentionally a single-pass transform. Any residual trailing negatives
        // can be neutralized by the neutralize_remaining_negative_pops stage in the pipeline.
        out
    }

    /// Reference implementation for the "neutralize" transformation.
    fn neutralize_remaining_negative_pops_stack(mut stack: Vec<TestEK>) -> Vec<TestEK> {
        // If the tail is a negative pop, neutralize it to pop 0, check None.
        if let Some(last) = stack.last_mut() {
            if last.pop < 0 {
                last.pop = 0;
                last.check = None;
            }
        }
        stack
    }

    // --- Semantic Invariant Checkers ---

    /// Compute realized actions as a map from absolute position to check id.
    /// Position is the running sum of pops; we record positions at which a check occurs.
    /// This is the semantic invariant that must be preserved by bubbling.
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

    // --- Trie Construction Helpers (using the framework's harness) ---
    use crate::trie_test_framework::harness;

    fn new_node(god: &TestGod) -> Trie2Index {
        harness::new_node(god, PrecomputedNodeContents::internal())
    }

    fn add_edge(god: &TestGod, from: Trie2Index, to: Trie2Index, key: TestEK) {
        harness::add_edge(god, from, to, key, ());
    }

    // --- Main Test Runner ---

    fn run_test(god: &TestGod, roots: &[Trie2Index]) {
        // --- Stage 1: Test `bubble_up_negative_pops` ---

        // Define the trie transformation for the bubble-up stage.
        let trie_bubble_transform = |god: &TestGod, roots: &[Trie2Index]| {
            bubble_up_negative_pops(
                god,
                roots,
                |ek| ek.pop,
                |ek, new_pop| TestEK::new(new_pop, ek.check),
                || TestEK::new(0, None),
                |ev1, _ev2| *ev1 = (),
            );
        };

        // Define an assertion for the intermediate state after bubbling.
        let bubble_assertion = |stacks: &BTreeSet<Vec<TestEK>>| {
            assert_negative_pops_follow_property_for_stacks(stacks, |ek| ek.pop);
        };

        // Run the test for the bubble-up stage.
        TrieTestFramework::new(god, roots)
            .with_stack_canonicalizer(compress_stack)
            .with_assertion(bubble_assertion)
            .test_transform(bubble_up_negative_pops_stack, trie_bubble_transform);

        // --- Stage 2: Test `neutralize_remaining_negative_pops` ---
        // This stage must run on the output of the bubble-up stage.

        // First, create the post-bubble trie state manually.
        let (god_after_bubble, roots_after_bubble, _) = Trie::deep_copy_subtrees(god, roots);
        bubble_up_negative_pops(
            &god_after_bubble,
            &roots_after_bubble,
            |ek| ek.pop,
            |ek, new_pop| TestEK::new(new_pop, ek.check),
            || TestEK::new(0, None),
            |ev1, _ev2| *ev1 = (),
        );

        // Define the trie transformation for the neutralize stage.
        let trie_neutralize_transform = |god: &TestGod, roots: &[Trie2Index]| {
            neutralize_remaining_negative_pops(
                god,
                roots,
                |ek| ek.pop,
                |ek, new_pop| TestEK::new(new_pop, ek.check),
                || TestEK::new(0, None),
                |ev1, _ev2| *ev1 = (),
            );
        };

        // Define an assertion for the final state.
        let final_assertion = |stacks: &BTreeSet<Vec<TestEK>>| {
            for stack in stacks {
                for ek in stack {
                    assert!(ek.pop >= 0, "Final stack should not have negative pops: {:?}", stack);
                }
            }
        };

        // Run the test for the neutralize stage, starting from the bubbled trie.
        TrieTestFramework::new(&god_after_bubble, &roots_after_bubble)
            .with_stack_canonicalizer(compress_stack)
            .with_assertion(final_assertion)
            .test_transform(neutralize_remaining_negative_pops_stack, trie_neutralize_transform);
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
    fn test_neutralize_handles_conditional_negative_pop_at_end() {
        let input = vec![
            TestEK::new(1, Some(2)),
            TestEK::new(-2, Some(3)),
        ];
        let expected = vec![
            TestEK::new(1, Some(2)),
            TestEK::new(0, None),
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

    // --- Graph-level tests ---

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
