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

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext<'_>) {
        let node_ids: Vec<_> = trie.node_ids().collect();
        let id_to_idx: HashMap<_, _> = node_ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let n = node_ids.len();
        if n == 0 {
            return;
        }

        let mut prev_class: Vec<usize> = vec![0; n];
        for (i, id) in node_ids.iter().enumerate() {
            let node = trie.get_node(*id).unwrap();
            prev_class[i] = if node.is_end() { 1 } else { 0 };
        }

        for _ in 0..self.max_iters {
            type AggregatedEdge = ((isize, SortedSet, usize), SortedSet);
            type Signature = (bool, Vec<AggregatedEdge>);
            let mut sig_to_id: HashMap<Signature, usize> = HashMap::new();
            let mut new_class = vec![0; n];
            let mut next_id = 0;
            let mut changes = 0;

            for (u_idx, u_id) in node_ids.iter().enumerate() {
                let u_node = trie.get_node(*u_id).unwrap();
                let mut aggr: BTreeMap<(isize, SortedSet, usize), SortedSet> = BTreeMap::new();
                for (ek, dm) in u_node.children() {
                    for (v_id, sids) in dm {
                        let dest_class = prev_class[id_to_idx[v_id]];
                        let key = (ek.pop, ek.tokens.clone(), dest_class);
                        aggr.entry(key).or_default().union_inplace(sids);
                    }
                }
                let sig = (u_node.is_end(), aggr.into_iter().collect());
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
                representatives[class_id] = Some(node_ids[u_idx]);
            }
        }
        let mut node_to_rep: HashMap<NodeId, NodeId> = HashMap::new();
        for (u_idx, &class_id) in prev_class.iter().enumerate() {
            node_to_rep.insert(node_ids[u_idx], representatives[class_id].unwrap());
        }

        for (class_id, rep_id_opt) in representatives.iter().enumerate() {
            if let Some(rep_id) = rep_id_opt {
                let exemplar_idx = prev_class.iter().position(|&c| c == class_id).unwrap();
                let exemplar_node = trie.get_node(node_ids[exemplar_idx]).unwrap().clone();

                let mut aggr: BTreeMap<(isize, SortedSet, usize), SortedSet> = BTreeMap::new();
                for (ek, dm) in exemplar_node.children() {
                    for (v_id, sids) in dm {
                        let v_class = prev_class[id_to_idx[v_id]];
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
                trie.set_children(*rep_id, new_children);
            }
        }

        let rep_set: std::collections::HashSet<NodeId> =
            representatives.iter().filter_map(|&x| x).collect();
        let all_node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in all_node_ids {
            if !rep_set.contains(&node_id) {
                trie.clear_children(node_id);
                trie.set_end(node_id, false);
            }
        }

        trie.root_ids = trie.root_ids.iter().map(|r| *node_to_rep.get(r).unwrap()).collect();
    }
}
