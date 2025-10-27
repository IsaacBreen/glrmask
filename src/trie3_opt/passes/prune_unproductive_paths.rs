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
        if productive.len() == trie.nodes.len() {
            return;
        }
        for node in &mut trie.nodes {
            if !productive.contains(&node.id) {
                node.children.clear();
            } else {
                node.children.retain(|_, dm| {
                    dm.retain(|dst, _| productive.contains(dst));
                    !dm.is_empty()
                });
            }
        }
    }
}
