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

    fn compress_stack(mut stack: Vec<TestEK>) -> Vec<TestEK> {
        todo!()
    }

    fn bubble_up_negative_pops_stack(mut stack: Vec<TestEK>) -> Vec<TestEK> {
        todo!()
    }

    fn neutralize_remaining_negative_pops_stack(mut stack: Vec<TestEK>) -> Vec<TestEK> {
        todo!()
    }

    // --- Trie Construction Helpers ---

    fn new_node(god: &TestGod) -> Trie2Index {
        Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())))
    }

    fn add_edge(god: &TestGod, from: Trie2Index, to: Trie2Index, key: TestEK) {
        todo!()
    }

    fn flatten_trie_to_stacks(god: &TestGod, roots: &[Trie2Index]) -> BTreeSet<Vec<TestEK>> {
        todo!()
    }

    fn run_test(god: &TestGod, roots: &[Trie2Index]) {
        let stacks = flatten_trie_to_stacks(god, roots);

        // bubble
        let bubbled_stacks = stacks.iter().map(|s| bubble_up_negative_pops_stack(s.clone())).collect::<BTreeSet<_>>();
        bubble_up_negative_pops(
            god,
            roots,
            |ek| ek.pop,
            |ek, new_pop| TestEK::new(new_pop, ek.check),
            |ev1, ev2| {}
        );
        let bubbled_trie_flattened = flatten_trie_to_stacks(god, roots);
        assert_eq!(bubbled_stacks, bubbled_trie_flattened);

        // neutralize
        let neutralized_stacks = bubbled_stacks.iter().map(|s| neutralize_remaining_negative_pops_stack(s.clone())).collect::<BTreeSet<_>>();
        neutralize_remaining_negative_pops(
            god,
            roots,
            |ek| ek.pop,
            |ek, new_pop| TestEK::new(new_pop, ek.check),
        );
        let neutralized_trie_flattened = flatten_trie_to_stacks(god, roots);
        assert_eq!(neutralized_stacks, neutralized_trie_flattened);

        // final check
        let final_stacks = neutralized_trie_flattened;
        for stack in final_stacks {
            for ek in &stack {
                assert!(ek.pop >= 0);
            }
        }
    }

    #[test]
    fn test_example() {
        // let input = vec![TestEK::new(3, Some(0)), TestEK::new(-2, Some(2))];
        let god = TestGod::new();
        let A = new_node(&god);
        let B = new_node(&god);
        let C = new_node(&god);
        add_edge(&god, A, B, TestEK::new(3, Some(0)));
        add_edge(&god, B, C, TestEK::new(-2, Some(2)));
        let roots = vec![A];
        run_test(&god, &roots)
    }
}
