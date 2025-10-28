use std::collections::BTreeSet;

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId};
use crate::trie3_opt::passes::OptimizationPass;

pub struct PruneUnproductivePathsPass;

impl OptimizationPass for PruneUnproductivePathsPass {
    fn name(&self) -> &'static str {
        "PruneUnproductivePaths"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        let productive: BTreeSet<NodeId> = trie.can_reach_end();
        if productive.len() == trie.num_nodes() {
            return;
        }
        let node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in node_ids {
            if !productive.contains(&node_id) {
                trie.clear_children(node_id);
            } else {
                let node = trie.get_node(node_id).unwrap();
                let mut new_children = node.children().clone();
                new_children.retain(|_, dm| {
                    dm.retain(|dst, _| productive.contains(dst));
                    !dm.is_empty()
                });
                trie.set_children(node_id, new_children);
            }
        }
    }
}
