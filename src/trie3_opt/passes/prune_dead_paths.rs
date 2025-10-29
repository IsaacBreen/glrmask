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

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let universe = SortedSet::from_iter(0..=ctx.max_llm_token_id);
        let productive_tokens = trie.productive_tokens_at_nodes(&universe);

        let node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in node_ids {
            let node = trie.get_node(node_id).unwrap();
            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
            for (ek, dm) in node.children() {
                for (dst, sids) in dm {
                    let productive_from_child = productive_tokens.get(dst).unwrap();
                    let productive_on_edge = ek.tokens.intersect(productive_from_child);

                    if !productive_on_edge.is_empty() {
                        let new_ek = EdgeKey::new(ek.pop, productive_on_edge);
                        new_children
                            .entry(new_ek)
                            .or_default()
                            .entry(*dst)
                            .and_modify(|e: &mut SortedSet| e.union_inplace(sids))
                            .or_insert(sids.clone());
                    }
                }
            }
            trie.set_children(node_id, new_children);
        }
    }
}
