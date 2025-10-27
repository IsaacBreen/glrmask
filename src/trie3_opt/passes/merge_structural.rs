use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Merge nodes with identical structure via iterative refinement. Signature includes:
/// - End flag
/// - For each edge-key: (pop, tokens) and destination-class aggregated states.
/// This is a simplified variant of the structural merge in the large pipeline, implemented
/// purely over the MiniTrie.
pub struct MergeStructuralPass {
    pub max_iters: usize,
}

impl MergeStructuralPass {
    pub fn new(max_iters: usize) -> Self {
        Self { max_iters }
    }
}

impl OptimizationPass for MergeStructuralPass {
    fn name(&self) -> &'static str {
        "MergeStructural"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        let n = trie.nodes.len();
        if n == 0 {
            return;
        }

        // Map node id -> dense index
        let mut dense_of: HashMap<NodeId, usize> = HashMap::with_capacity(n);
        let mut old_of: Vec<NodeId> = Vec::with_capacity(n);
        for (i, node) in trie.nodes.iter().enumerate() {
            dense_of.insert(node.id, i);
            old_of.push(node.id);
        }

        // Initial classes by end flag and out-degree
        let mut prev_class: Vec<usize> = vec![0; n];
        let mut class_map: HashMap<(bool, usize), usize> = HashMap::new();
        let mut next_class = 0usize;
        for (i, node) in trie.nodes.iter().enumerate() {
            let key = (node.end, node.out_degree());
            let cid = *class_map.entry(key).or_insert_with(|| {
                let v = next_class;
                next_class += 1;
                v
            });
            prev_class[i] = cid;
        }

        for _ in 0..self.max_iters.max(1) {
            // Build signature per node
            #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
            struct EdgeSig {
                pop: isize,
                tokens: SortedSet,
                dest: Vec<(usize, SortedSet)>, // dest_class, states
            }
            type NodeSig = (bool, Vec<EdgeSig>);

            let mut sig_to_id: HashMap<NodeSig, usize> = HashMap::new();
            let mut new_class: Vec<usize> = vec![0; n];
            let mut next = 0usize;
            let mut changes = 0usize;

            for (i, node) in trie.nodes.iter().enumerate() {
                // Aggregate by (pop, tokens) -> dest_class -> union of states
                let mut per_key: BTreeMap<(isize, SortedSet), BTreeMap<usize, SortedSet>> = BTreeMap::new();
                for (ek, dm) in &node.children {
                    let inner = per_key.entry((ek.pop, ek.tokens.clone())).or_default();
                    for (dst, sids) in dm.iter() {
                        let d_dense = *dense_of.get(dst).unwrap();
                        let d_class = prev_class[d_dense];
                        let entry = inner.entry(d_class).or_default();
                        entry.union_inplace(sids);
                    }
                }

                // Build canonical signature vector
                let mut esigs: Vec<EdgeSig> = Vec::with_capacity(per_key.len());
                for ((pop, toks), dm_class) in per_key.into_iter() {
                    let mut dest_vec: Vec<(usize, SortedSet)> = dm_class.into_iter().collect();
                    dest_vec.sort_unstable_by(|a, b| {
                        let c = a.0.cmp(&b.0);
                        if c != std::cmp::Ordering::Equal {
                            return c;
                        }
                        a.1.cmp(&b.1)
                    });
                    esigs.push(EdgeSig {
                        pop,
                        tokens: toks,
                        dest: dest_vec,
                    });
                }
                let sig: NodeSig = (node.end, esigs);

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

        // Build representative mapping
        let num_classes = prev_class.iter().max().map(|x| *x + 1).unwrap_or(0);
        let mut rep_of_class: Vec<Option<NodeId>> = vec![None; num_classes];
        for (i, &cid) in prev_class.iter().enumerate() {
            if rep_of_class[cid].is_none() {
                rep_of_class[cid] = Some(old_of[i]);
            }
        }
        let mut rep_of_node: HashMap<NodeId, NodeId> = HashMap::with_capacity(n);
        for (i, node_id) in old_of.iter().enumerate() {
            let cid = prev_class[i];
            rep_of_node.insert(*node_id, rep_of_class[cid].unwrap());
        }

        let original_nodes = trie.nodes.clone();
        for (cid, rep_id_opt) in rep_of_class.iter().enumerate() {
            if let Some(rep_id) = rep_id_opt {
                let exemplar_idx = prev_class.iter().position(|&c| c == cid).unwrap();
                let exemplar_node = &original_nodes[exemplar_idx];

                let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> =
                    BTreeMap::new();
                for (ek, dm) in &exemplar_node.children {
                    let mut new_dm = BTreeMap::new();
                    for (dst, sids) in dm {
                        let rep = *rep_of_node.get(dst).unwrap();
                        new_dm.entry(rep).or_default().union_inplace(sids);
                    }
                    if !new_dm.is_empty() {
                        new_children.insert(ek.clone(), new_dm);
                    }
                }
                trie.nodes[*rep_id as usize].children = new_children;
            }
        }

        let rep_set: std::collections::HashSet<NodeId> =
            rep_of_class.iter().filter_map(|&x| x).collect();
        for node in &mut trie.nodes {
            if !rep_set.contains(&node.id) {
                node.children.clear();
                node.end = false;
            }
        }

        trie.root_ids = trie
            .root_ids
            .iter()
            .map(|r| *rep_of_node.get(r).unwrap())
            .collect();
    }
}
