use std::collections::BTreeMap;

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId};
use crate::trie3_opt::passes::OptimizationPass;
use crate::trie3_opt::SortedSet;

/// Canonicalize all end nodes to a single representative and rewire edges to it.
/// The canonical end node keeps no outgoing edges. This preserves semantics and
/// maximizes sharing downstream.
pub struct CanonicalizeEndNodesPass;

impl OptimizationPass for CanonicalizeEndNodesPass {
    fn name(&self) -> &'static str {
        "CanonicalizeEndNodes"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        // Collect end nodes
        let mut ends: Vec<NodeId> = trie.nodes.iter().filter(|n| n.end).map(|n| n.id).collect();
        if ends.len() <= 1 {
            return;
        }
        ends.sort_unstable();
        let canonical = ends[0];

        // Ensure canonical has no outgoing edges
        {
            let cnode = &mut trie.nodes[canonical as usize];
            cnode.children.clear();
            cnode.end = true;
        }

        // Rewire all edges to target canonical if they target any end
        let end_set: std::collections::BTreeSet<NodeId> = ends.into_iter().collect();

        for node in trie.nodes.iter_mut() {
            let mut new_children: BTreeMap<_, _> = BTreeMap::new();
            for (ek, dm) in node.children.clone() {
                let mut dm2: BTreeMap<NodeId, _> = BTreeMap::new();
                for (dst, sids) in dm {
                    let new_dst = if end_set.contains(&dst) { canonical } else { dst };
                    dm2.entry(new_dst)
                        .and_modify(|e: &mut SortedSet| e.union_inplace(&sids))
                        .or_insert(sids);
                }
                if !dm2.is_empty() {
                    new_children.insert(ek, dm2);
                }
            }
            node.children = new_children;
        }
    }
}
