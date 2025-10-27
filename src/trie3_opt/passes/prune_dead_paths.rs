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
