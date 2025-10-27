use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

pub struct MergeBisimulationPass {
    max_iters: usize,
}

impl MergeBisimulationPass {
    pub fn new(max_iters: usize) -> Self {
        Self { max_iters }
    }
}

impl OptimizationPass for MergeBisimulationPass {
    fn name(&self) -> &'static str {
        "MergeBisimulation"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        let n = trie.nodes.len();
        if n == 0 {
            return;
        }

        let mut prev_class: Vec<usize> =
            trie.nodes.iter().map(|node| if node.end { 1 } else { 0 }).collect();

        for _ in 0..self.max_iters {
            type AggregatedEdge = ((isize, SortedSet, usize), SortedSet);
            type Signature = (bool, Vec<AggregatedEdge>);
            let mut sig_to_id: HashMap<Signature, usize> = HashMap::new();
            let mut new_class = vec![0; n];
            let mut next_id = 0;
            let mut changes = 0;

            for (u_idx, u_node) in trie.nodes.iter().enumerate() {
                let mut aggr: BTreeMap<(isize, SortedSet, usize), SortedSet> = BTreeMap::new();
                for (ek, dm) in &u_node.children {
                    for (v_id, sids) in dm {
                        let dest_class = prev_class[*v_id as usize];
                        let key = (ek.pop, ek.tokens.clone(), dest_class);
                        aggr.entry(key).or_default().union_inplace(sids);
                    }
                }
                let sig = (u_node.end, aggr.into_iter().collect());
                let cid = *sig_to_id.entry(sig).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                new_class[u_idx] = cid;
                if new_class[u_idx] != prev_class[u_idx] {
                    changes += 1;
                }
            }
            prev_class = new_class;
            if changes == 0 {
                break;
            }
        }

        let num_classes = prev_class.iter().max().map_or(0, |m| m + 1);
        let mut representatives: Vec<Option<NodeId>> = vec![None; num_classes];
        for (u_idx, &class_id) in prev_class.iter().enumerate() {
            if representatives[class_id].is_none() {
                representatives[class_id] = Some(u_idx as NodeId);
            }
        }
        let mut node_to_rep: HashMap<NodeId, NodeId> = HashMap::new();
        for (u_idx, &class_id) in prev_class.iter().enumerate() {
            node_to_rep.insert(u_idx as NodeId, representatives[class_id].unwrap());
        }

        let original_nodes = trie.nodes.clone();
        for (class_id, rep_id_opt) in representatives.iter().enumerate() {
            if let Some(rep_id) = rep_id_opt {
                let exemplar_idx = prev_class.iter().position(|&c| c == class_id).unwrap();
                let exemplar_node = &original_nodes[exemplar_idx];

                let mut aggr: BTreeMap<(isize, SortedSet, usize), SortedSet> = BTreeMap::new();
                for (ek, dm) in &exemplar_node.children {
                    for (v_id, sids) in dm {
                        let v_class = prev_class[*v_id as usize];
                        aggr.entry((ek.pop, ek.tokens.clone(), v_class))
                            .or_default()
                            .union_inplace(sids);
                    }
                }

                let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> =
                    BTreeMap::new();
                for ((pop, tokens, dest_class), sids) in aggr {
                    if let Some(dest_rep) = representatives[dest_class] {
                        let key = EdgeKey::new(pop, tokens);
                        new_children.entry(key).or_default().insert(dest_rep, sids);
                    }
                }
                trie.nodes[*rep_id as usize].children = new_children;
            }
        }

        let rep_set: std::collections::HashSet<NodeId> =
            representatives.iter().filter_map(|&x| x).collect();
        for node in &mut trie.nodes {
            if !rep_set.contains(&node.id) {
                node.children.clear();
                node.end = false;
            }
        }

        trie.root_ids = trie.root_ids.iter().map(|r| *node_to_rep.get(r).unwrap()).collect();
    }
}
