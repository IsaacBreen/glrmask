use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Merge nodes by a state-agnostic bisimulation and then aggregate state sets safely.
/// Two nodes are equivalent if they:
///   - have the same `end` flag, and
///   - for every (pop, tokens) key, the set of destination equivalence classes is identical.
///
/// After convergence, we collapse each class to a representative:
///   - For each ((pop, tokens), dest_class) triple across the class, we union all state-sets
///     into a single edge to the representative of dest_class.
///   - Non-representatives are cleared (children removed, end=false), and roots are remapped.
///
/// This preserves all token-labeled languages exactly and remains sound for state constraints
/// (we only take supersets of state bitsets when merging).
pub struct MergeStateAgnosticPass {
    max_iters: usize,
}

impl MergeStateAgnosticPass {
    pub fn new(max_iters: usize) -> Self {
        Self { max_iters }
    }
}

impl OptimizationPass for MergeStateAgnosticPass {
    fn name(&self) -> &'static str {
        "MergeStateAgnostic"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        let node_ids: Vec<_> = trie.node_ids().collect();
        let n = node_ids.len();
        if n == 0 {
            return;
        }
        let id_to_idx: HashMap<NodeId, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();

        // Initial coarse partition by (end flag, out-degree of key groups).
        // We consider "key groups" as unique EdgeKey (pop, tokens) ignoring destinations.
        let mut prev_class: Vec<usize> = vec![0; n];
        let mut init_map: HashMap<(bool, usize), usize> = HashMap::new();
        let mut next_cid = 0usize;
        for (i, id) in node_ids.iter().enumerate() {
            let node = trie.get_node(*id).unwrap();
            let out_keys = node.children().len();
            let key = (node.is_end(), out_keys);
            let cid = *init_map.entry(key).or_insert_with(|| {
                let v = next_cid;
                next_cid += 1;
                v
            });
            prev_class[i] = cid;
        }

        // Iterative refinement of the partition while ignoring state sets.
        for _ in 0..self.max_iters.max(1) {
            // Node signature type:
            //   (is_end, Vec<((pop, tokens), Vec<dest_class>)>)
            // where dest_class vectors are sorted and unique, and the outer vector is sorted
            // by (pop, tokens).
            type NodeSig = (bool, Vec<((isize, SortedSet), Vec<usize>)>);

            let mut sig_to_id: HashMap<NodeSig, usize> = HashMap::new();
            let mut new_class: Vec<usize> = vec![0; n];
            let mut next = 0usize;
            let mut changes = 0usize;

            for (i, id) in node_ids.iter().enumerate() {
                let node = trie.get_node(*id).unwrap();
                // Aggregate for this node: (pop, tokens) -> set of dest classes
                let mut per_key: BTreeMap<(isize, SortedSet), BTreeSet<usize>> = BTreeMap::new();
                for (ek, dm) in node.children() {
                    let entry = per_key.entry((ek.pop, ek.tokens.clone())).or_default();
                    for (dst, _sids) in dm {
                        let d_idx = id_to_idx[dst];
                        let d_class = prev_class[d_idx];
                        entry.insert(d_class);
                    }
                }
                // Canonicalize into sorted vectors
                let mut vec_sig: Vec<((isize, SortedSet), Vec<usize>)> = Vec::with_capacity(per_key.len());
                for (k, class_set) in per_key.into_iter() {
                    let mut dcs: Vec<usize> = class_set.into_iter().collect();
                    // Already sorted because BTreeSet yields in-order; keep explicitness
                    // and ensure deterministic ordering.
                    dcs.sort_unstable();
                    vec_sig.push((k, dcs));
                }
                let sig: NodeSig = (node.is_end(), vec_sig);
                let cid = *sig_to_id.entry(sig).or_insert_with(|| {
                    let v = next;
                    next += 1;
                    v
                });
                new_class[i] = cid;
                if new_class[i] != prev_class[i] {
                    changes += 1;
                }
            }
            prev_class = new_class;
            if changes == 0 {
                break;
            }
        }

        // Build representative mapping.
        let num_classes = prev_class.iter().max().map_or(0, |m| m + 1);
        let mut representatives: Vec<Option<NodeId>> = vec![None; num_classes];
        for (i, &cid) in prev_class.iter().enumerate() {
            if representatives[cid].is_none() {
                representatives[cid] = Some(node_ids[i]);
            }
        }
        let mut node_to_rep: HashMap<NodeId, NodeId> = HashMap::new();
        for (i, id) in node_ids.iter().enumerate() {
            let cid = prev_class[i];
            node_to_rep.insert(*id, representatives[cid].unwrap());
        }

        // Rebuild edges for each representative:
        // Aggregate by (pop, tokens, dest_class) -> union(state sets)
        for class_id in 0..num_classes {
            if let Some(rep_id) = representatives[class_id] {
                let mut aggr: BTreeMap<(isize, SortedSet, usize), SortedSet> = BTreeMap::new();

                for (i, id) in node_ids.iter().enumerate() {
                    if prev_class[i] != class_id {
                        continue;
                    }
                    let node = trie.get_node(*id).unwrap();
                    for (ek, dm) in node.children() {
                        for (dst, sids) in dm {
                            let d_class = prev_class[id_to_idx[dst]];
                            aggr.entry((ek.pop, ek.tokens.clone(), d_class))
                                .or_default()
                                .union_inplace(sids);
                        }
                    }
                }

                let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
                for ((pop, tokens, d_class), sids) in aggr {
                    if let Some(dest_rep) = representatives[d_class] {
                        if !sids.is_empty() {
                            let key = EdgeKey::new(pop, tokens);
                            new_children.entry(key).or_default().insert(dest_rep, sids);
                        }
                    }
                }
                trie.set_children(rep_id, new_children);
            }
        }

        // Clear children of non-representatives and normalize end flags.
        let rep_set: BTreeSet<NodeId> = representatives.iter().filter_map(|&x| x).collect();
        let all_node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in all_node_ids {
            if !rep_set.contains(&node_id) {
                trie.clear_children(node_id);
                trie.set_end(node_id, false);
            }
        }

        // Remap roots to representatives.
        trie.root_ids = trie.root_ids.iter().map(|r| *node_to_rep.get(r).unwrap()).collect();
    }
}
