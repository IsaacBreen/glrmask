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
        let productive = trie.productive_llm_tokens();

        let node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in node_ids {
            let node = trie.get_node(node_id).unwrap();
            let node_productive_tokens = productive.get(&node_id).unwrap();

            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
            for (ek, dm) in node.children() {
                let new_tokens = ek.tokens.intersect(node_productive_tokens);
                if !new_tokens.is_empty() {
                    let new_ek = EdgeKey::new(ek.pop, new_tokens);
                    let entry = new_children.entry(new_ek).or_default();
                    for (dst, sids) in dm {
                        entry.entry(*dst).or_default().union_inplace(sids);
                    }
                }
            }
            trie.set_children(node_id, new_children);
        }
    }
}
