// src/constraint_precompute3_eliminate_negative_pops.rs
use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};
use std::collections::HashSet;
use std::iter::FromIterator;
use crate::datastructures::EntryApi;

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
    let all_nodes = Trie::all_nodes(god, roots);
    for node_idx in all_nodes {
        let mut edges_to_modify = Vec::new();
        {
            let node_r = node_idx.read(god).unwrap();
            for edge_key in node_r.children().keys() {
                if get_pop(edge_key) < 0 {
                    edges_to_modify.push(edge_key.clone());
                }
            }
        } // read lock dropped

        if !edges_to_modify.is_empty() {
            let mut node_w = node_idx.write(god).unwrap();
            for old_key in edges_to_modify {
                if let Some(dest_map) = node_w.children_mut().remove(&old_key) {
                    let old_pop = get_pop(&old_key);
                    let new_pop = -old_pop;
                    let new_key = make_key(&old_key, new_pop);

                    let target_dest_map = node_w.children_mut().entry(new_key).or_default();
                    for (dest_idx, edge_val) in dest_map {
                        target_dest_map
                            .entry(dest_idx)
                            .and_modify(|existing_val| merge_ev(existing_val, edge_val.clone()))
                            .or_insert(edge_val);
                    }
                }
            }
        }
    }
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
    // No-op for now, as bubble_up handles all cases.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PrecomputedNodeContents;
    use crate::datastructures::hybrid_bitset::HybridBitset;
    use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};
    use std::collections::BTreeSet;

    // --- Test Helpers ---
    type TestEK = (isize, u32);
    type TestEV = HybridBitset;
    type TestT = PrecomputedNodeContents;
    type TestGod = GodWrapper<TestEK, TestEV, TestT>;

    fn new_node(god: &TestGod) -> Trie2Index {
        Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())))
    }

    fn state_bv(indices: &[usize]) -> TestEV {
        let mut bv = TestEV::zeros();
        for &i in indices {
            bv.insert(i);
        }
        bv
    }

    fn add_edge(
        god: &TestGod,
        from: Trie2Index,
        key: TestEK,
        to: Trie2Index,
        val: TestEV,
    ) {
        let mut from_w = from.write(god).unwrap();
        from_w.children_mut().entry(key).or_default().insert(to, val);
    }

    // --- Type Aliases for Readability ---
    type Path = Vec<TestEK>;
    type PathSet = BTreeSet<Path>;

    // --- Tests ---

    #[test]
    fn test_bubble_up_negative_pops_simple() {
        let god = TestGod::new();

        let root = new_node(&god);
        let n1 = new_node(&god);
        let n2 = new_node(&god);
        let n3 = new_node(&god);

        let key1 = (-1, 100);
        let key2 = (-2, 200);
        let key3_pos = (1, 300);
        let key3_neg = (-1, 300);

        add_edge(&god, root, key1, n1, state_bv(&[1]));
        add_edge(&god, n1, key2, n2, state_bv(&[2]));
        add_edge(&god, root, key3_pos, n3, state_bv(&[3]));
        add_edge(&god, root, key3_neg, n3, state_bv(&[4]));

        let actual_paths_before = Trie::get_all_paths(&god, &[root]);
        let expected_paths_before = PathSet::from([
            vec![],
            vec![key1],
            vec![key1, key2],
            vec![key3_pos],
            vec![key3_neg],
        ]);
        assert_eq!(actual_paths_before, expected_paths_before);

        bubble_up_negative_pops(
            &god,
            &[root],
            &mut |k: &TestEK| k.0,
            &mut |k: &TestEK, new_pop| (new_pop, k.1),
            &mut |v1, v2| *v1 |= &v2,
        );

        // 4. ASSERT (After): Verify the set of possible paths is correct after transformation
        let actual_paths_after = Trie::get_all_paths(&god, &[root]);

        let expected_paths_after = PathSet::from([
            // The root path is always present.
            vec![],
            vec![(1, 100)],
            vec![(1, 100), (2, 200)],
            vec![(1, 300)],
        ]);
        assert_eq!(actual_paths_after, expected_paths_after);

        // 5. ASSERT (Values): Verify edge values were merged correctly.
        let root_r = root.read(&god).unwrap();
        let dest_map_for_key3 = root_r.get(&(1, 300)).unwrap();
        let merged_val = dest_map_for_key3.get(&n3).unwrap();
        let expected_merged_val = state_bv(&[3, 4]);
        assert_eq!(*merged_val, expected_merged_val);
    }
}
