use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Remove edges that provably cannot lead to an end node by reverse reachability.
/// Nodes are not deleted in the mini-trie (to keep NodeId stability), but edges to
/// non-productive nodes are pruned.
pub struct PruneDeadPathsPass;

impl OptimizationPass for PruneDeadPathsPass {
    fn name(&self) -> &'static str {
        "PruneDeadPaths"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let mut predecessors: HashMap<NodeId, Vec<(NodeId, EdgeKey)>> = HashMap::new();
        let mut worklist = VecDeque::new();
        let mut live: HashMap<NodeId, SortedSet> = HashMap::new();
        let universe = SortedSet::from_iter(0..=ctx.max_llm_token_id);

        for node in &trie.nodes {
            live.insert(node.id, SortedSet::new());
            if node.end {
                live.insert(node.id, universe.clone());
                worklist.push_back(node.id);
            }
            for (ek, dm) in &node.children {
                for (dst, _) in dm {
                    predecessors
                        .entry(*dst)
                        .or_default()
                        .push((node.id, ek.clone()));
                }
            }
        }

        while let Some(node_id) = worklist.pop_front() {
            let live_at_node = live.get(&node_id).unwrap().clone();
            if let Some(preds) = predecessors.get(&node_id) {
                for (pred_id, edge_key) in preds {
                    let live_from_edge = live_at_node.intersect(&edge_key.tokens);
                    if live_from_edge.is_empty() {
                        continue;
                    }
                    let pred_live = live.get_mut(pred_id).unwrap();
                    let old_len = pred_live.len();
                    pred_live.union_inplace(&live_from_edge);
                    if pred_live.len() > old_len {
                        worklist.push_back(*pred_id);
                    }
                }
            }
        }

        for node in &mut trie.nodes {
            let mut new_children = BTreeMap::new();
            for (ek, dm) in &node.children {
                let mut new_dm_for_key = BTreeMap::new();
                for (dst, sids) in dm {
                    let live_from_child = live.get(dst).unwrap();
                    let live_on_edge = ek.tokens.intersect(live_from_child);

                    if !live_on_edge.is_empty() {
                        let new_ek = EdgeKey::new(ek.pop, live_on_edge);
                        let entry: &mut BTreeMap<NodeId, SortedSet> = new_children.entry(new_ek).or_default();
                        entry
                            .entry(*dst)
                            .and_modify(|e: &mut SortedSet| e.union_inplace(sids))
                            .or_insert(sids.clone());
                    }
                }
            }
            node.children = new_children;
        }
    }
}
