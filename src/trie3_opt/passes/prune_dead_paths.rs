use std::collections::BTreeMap;

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

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        // Identify nodes that can reach an END node.
        let productive_nodes = trie.can_reach_end();

        let node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in node_ids {
            let node = trie.get_node(node_id).unwrap();
            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
            for (ek, dm) in node.children() {
                // Keep the original edge key; only filter out dead destinations.
                let mut dm2: BTreeMap<NodeId, SortedSet> = BTreeMap::new();
                for (dst, sids) in dm {
                    if productive_nodes.contains(dst) {
                        dm2.entry(*dst)
                            .or_default()
                            .union_inplace(sids);
                    }
                }
                if !dm2.is_empty() {
                    new_children.insert(ek.clone(), dm2);
                }
            }
            trie.set_children(node_id, new_children);
        }
    }
}
