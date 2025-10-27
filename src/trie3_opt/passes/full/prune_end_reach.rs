use std::collections::{BTreeMap, HashMap, HashSet, VecDeque, BTreeMap as OrderedBTreeMap};

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::{Trie, Trie2Index};

/// Prune nodes that cannot reach any end node via reverse reachability.
pub fn prune_nodes_not_reaching_end_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(
        2,
        "Pruning Trie3 nodes that cannot reach any end node (reverse reachability)..."
    );
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    if roots_vec.is_empty() {
        return;
    }

    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // Build reverse adjacency: dest -> sources
    let mut incoming: HashMap<PrecomputeNode3Index, Vec<PrecomputeNode3Index>> = HashMap::new();
    for src in &all_nodes {
        let g = src.read(trie3_god).expect("read");
        for (_ek, dm) in g.children() {
            for (dst, _bv) in dm {
                incoming.entry(*dst).or_default().push(*src);
            }
        }
    }

    // Initialize worklist with all end nodes
    let mut productive: HashSet<PrecomputeNode3Index> = HashSet::new();
    let mut q: VecDeque<PrecomputeNode3Index> = VecDeque::new();
    let mut end_nodes_count = 0usize;
    for n in &all_nodes {
        let r = n.read(trie3_god).expect("read");
        if r.value.end {
            end_nodes_count += 1;
            if productive.insert(*n) {
                q.push_back(*n);
            }
        }
    }
    if end_nodes_count == 0 {
        crate::debug!(
            2,
            "No end nodes found in Trie3; skipping end-reachability pruning."
        );
        return;
    }

    // Reverse BFS
    while let Some(d) = q.pop_front() {
        if let Some(srcs) = incoming.get(&d) {
            for s in srcs {
                if productive.insert(*s) {
                    q.push_back(*s);
                }
            }
        }
    }

    let total_nodes = all_nodes.len();
    let productive_nodes = productive.len();
    let prunable = total_nodes.saturating_sub(productive_nodes);
    crate::debug!(
        2,
        "Trie3 end-reachability: total={}, productive={}, prunable={}",
        total_nodes,
        productive_nodes,
        prunable
    );
    if prunable == 0 {
        return;
    }

    // Remove any edge to a non-productive destination
    for n in &all_nodes {
        let mut w = n.write(trie3_god).expect("write");
        let mut new_children: BTreeMap<
            (isize, LLMTokenBV),
            OrderedHashMap<Trie2Index, StateIDBV>,
        > = BTreeMap::new();
        for (ek, dm) in w.children().clone() {
            let mut new_dm: OrderedHashMap<Trie2Index, StateIDBV> = OrderedHashMap::new();
            for (dst, bv) in dm {
                if productive.contains(&dst) {
                    new_dm.insert(dst, bv);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        *w.children_mut() = new_children;
    }

    // GC everything now unreachable from roots
    let roots_vec2: Vec<_> = roots.values().cloned().collect();
    Trie::gc(trie3_god, &roots_vec2);
    Trie::recompute_all_max_depths(trie3_god, &roots_vec2);

    crate::debug!(2, "Finished end-reachability pruning in Trie3.");
}
