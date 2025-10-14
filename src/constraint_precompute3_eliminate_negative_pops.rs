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
}
