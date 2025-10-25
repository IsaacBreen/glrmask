use std::collections::{HashMap, VecDeque};

use crate::constraint::{
    IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper,
    PrecomputeNode3Index, PrecomputedNodeContents,
};
use crate::datastructures::trie::{Trie, Trie2Index};

/// Deeply clone a trie3 graph starting from a Trie2Index root in the intermediate arena.
/// Returns the new root and a mapping from old node indices to new node indices.
pub fn clone_trie3_graph(
    root: &Trie2Index,
    trie3_god: &IntermediateTrie3GodWrapper,
) -> (
    IntermediatePrecomputeNode3Index,
    HashMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index>,
) {
    let mut map: HashMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        HashMap::new();
    let mut q: VecDeque<IntermediatePrecomputeNode3Index> = VecDeque::new();

    let root_ptr = *root;
    let root_value = trie3_god
        .with(root_ptr.into(), |n| n.value.clone())
        .expect("root must exist");
    let new_root = IntermediatePrecomputeNode3Index::new(trie3_god.insert(Trie::new(root_value)));
    map.insert(root_ptr, new_root);
    q.push_back(root_ptr);

    while let Some(old_arc) = q.pop_front() {
        let old_ptr = old_arc;
        let new_arc = map.get(&old_ptr).expect("parent must be created").clone();

        let children_snapshot: Vec<(
            IntermediateTrie3EdgeKey,
            Vec<(IntermediatePrecomputeNode3Index, ())>,
        )> = {
            trie3_god
                .with(old_arc.into(), |g| {
                    g.children()
                        .iter()
                        .map(|(ek, dest_map)| {
                            let entries = dest_map
                                .iter()
                                .map(|(node_ptr, ev)| (node_ptr.clone(), ev.clone()))
                                .collect::<Vec<_>>();
                            (ek.clone(), entries)
                        })
                        .collect()
                })
                .expect("old_arc must exist")
        };

        for (_ek, entries) in &children_snapshot {
            for (node_ptr, _ev) in entries {
                let child_ptr_old = *node_ptr;
                if !map.contains_key(&child_ptr_old) {
                    let child_value = trie3_god
                        .with(child_ptr_old.into(), |n| n.value.clone())
                        .expect("child must exist");
                    let child_arc_new =
                        IntermediatePrecomputeNode3Index::new(trie3_god.insert(Trie::new(child_value)));
                    map.insert(child_ptr_old, child_arc_new);
                    q.push_back(child_ptr_old);
                }
            }
        }

        {
            trie3_god
                .with_mut(new_arc.into(), |new_g| {
                    for (ek, entries) in children_snapshot {
                        for (old_node_ptr, ev) in entries {
                            let new_key = *map.get(&old_node_ptr).expect("must exist");
                            new_g.children_mut().entry(ek.clone()).or_default().insert(new_key, ev);
                        }
                    }
                })
                .expect("new_arc must exist");
        }
    }

    Trie::recompute_all_max_depths(trie3_god, &[new_root]);
    (new_root, map)
}
