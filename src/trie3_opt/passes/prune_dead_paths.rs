use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Prunes dead paths via backwards token liveness analysis.
/// An edge is live if its token mask has a non-empty intersection with the
/// live token set of its destination. A node's live token set is the union
/// of live tokens propagated from its successors.
pub struct PruneDeadPathsPass;

impl OptimizationPass for PruneDeadPathsPass {
    fn name(&self) -> &'static str {
        "PruneDeadPaths"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        if trie.nodes.is_empty() {
            return;
        }

        let mut predecessors: HashMap<NodeId, Vec<(NodeId, (isize, SortedSet))>> = HashMap::new();
        let mut worklist = VecDeque::new();
        let mut live: HashMap<NodeId, SortedSet> = HashMap::new();

        // 1. Initialize live sets and build predecessor map.
        for node in &trie.nodes {
            live.insert(node.id, SortedSet::new());

            if node.end {
                // Seed end nodes with 'all tokens' to allow backward propagation.
                let all_tokens = SortedSet::from_iter(0..=ctx.max_llm_token_id);
                live.insert(node.id, all_tokens);
                worklist.push_back(node.id);
            }

            for (edge_key, dest_map) in &node.children {
                for child_id in dest_map.keys() {
                    predecessors
                        .entry(*child_id)
                        .or_default()
                        .push((node.id, (edge_key.pop, edge_key.tokens.clone())));
                }
            }
        }

        // 2. Propagate liveness until a fixed point is reached.
        while let Some(node_id) = worklist.pop_front() {
            let live_at_node = live.get(&node_id).unwrap().clone();
            if let Some(preds) = predecessors.get(&node_id) {
                for (pred_id, edge_key) in preds {
                    let live_from_edge = live_at_node.intersect(&edge_key.1);
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

        // 3. Prune the graph based on the computed live sets.
        for node in trie.nodes.iter_mut() {
            let mut new_children: BTreeMap<_, _> = BTreeMap::new();

            for (edge_key, dest_map) in &node.children {
                for (child_id, edge_value_sids) in dest_map {
                    let live_from_child = live.get(child_id).unwrap();
                    let live_on_edge = edge_key.tokens.intersect(live_from_child);

                    if !live_on_edge.is_empty() {
                        let new_edge_key = super::super::core::EdgeKey::new(edge_key.pop, live_on_edge);
                        let new_dest_map_for_key = new_children.entry(new_edge_key).or_default();
                        new_dest_map_for_key.entry(*child_id).and_modify(|v: &mut SortedSet| v.union_inplace(edge_value_sids)).or_insert_with(|| edge_value_sids.clone());
                    }
                }
            }
            node.children = new_children;
        }
    }
}
use std::collections::{BTreeMap, BTreeSet};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId};
use crate::trie3_opt::passes::OptimizationPass;

/// Remove edges that provably cannot lead to an end node by reverse reachability.
/// Nodes are not deleted in the mini-trie (to keep NodeId stability), but edges to
/// non-productive nodes are pruned.
pub struct PruneDeadPathsPass;

impl OptimizationPass for PruneDeadPathsPass {
    fn name(&self) -> &'static str {
        "PruneDeadPaths"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        let productive: BTreeSet<NodeId> = trie.can_reach_end();
        if productive.is_empty() {
            return;
        }
        for n in trie.nodes.iter_mut() {
            let mut new_children: BTreeMap<_, _> = BTreeMap::new();
            for (ek, dm) in n.children.clone() {
                let mut dm2: BTreeMap<_, _> = BTreeMap::new();
                for (dst, sids) in dm {
                    if productive.contains(&dst) {
                        dm2.insert(dst, sids);
                    }
                }
                if !dm2.is_empty() {
                    new_children.insert(ek, dm2);
                }
            }
            n.children = new_children;
        }
    }
}
