use std::collections::{BTreeMap, BTreeSet};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId};
use crate::trie3_opt::passes::OptimizationPass;

/// Remove edges that provably cannot lead to an end node by reverse reachability.
/// Nodes are not deleted in the mini-trie (to keep NodeId stability), but edges to
/// non-productive nodes are pruned.
pub struct PruneUnproductivePathsPass;

impl OptimizationPass for PruneUnproductivePathsPass {
    fn name(&self) -> &'static str {
        "PruneUnproductivePaths"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        // If there are no end nodes in the graph, then no node is "productive".
        // In this case, we should not prune anything, as the graph might be
        // intentionally non-terminating.
        if trie.nodes.iter().all(|n| !n.end) {
            return;
        }

        let productive: BTreeSet<NodeId> = trie.can_reach_end();

        // If no nodes can reach an end node (but end nodes exist), it implies
        // the end nodes are unreachable from any root, which is a valid case for pruning.
        for n in trie.nodes.iter_mut() {
            let mut new_children: BTreeMap<_, _> = BTreeMap::new();
            // We use .clone() here to avoid borrowing issues with n.children.
            // This is acceptable as this pass is not on a hot path.
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