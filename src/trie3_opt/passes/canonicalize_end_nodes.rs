use std::collections::{BTreeMap, BTreeSet};

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

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext<'_>) {
        // Collect end nodes
        let mut ends: Vec<NodeId> = trie
            .nodes()
            .filter(|n| n.is_end())
            .map(|n| n.id())
            .collect();
        if ends.len() <= 1 {
            return;
        }
        ends.sort_unstable();
        let canonical = ends[0];
        let other_ends: BTreeSet<NodeId> = ends.iter().skip(1).cloned().collect();

        // Ensure canonical has no outgoing edges and is an end node.
        trie.clear_children(canonical);
        trie.set_end(canonical, true);

        // Decommission other end nodes. They will be garbage collected if unreferenced.
        for &end_node_id in &other_ends {
            trie.set_end(end_node_id, false);
            trie.clear_children(end_node_id);
        }

        // Rewire all edges that target any end node to target the canonical one.
        let end_set: BTreeSet<NodeId> = ends.into_iter().collect();
        let node_ids: Vec<_> = trie.node_ids().collect();

        for node_id in node_ids {
            let node = trie.get_node(node_id).unwrap();
            let mut new_children: BTreeMap<_, _> = BTreeMap::new();
            for (ek, dm) in node.children().clone() {
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
            trie.set_children(node_id, new_children);
        }

        // Remap roots if any of them were end nodes that are not the canonical one.
        let mut new_roots = Vec::with_capacity(trie.root_ids.len());
        let mut root_set = BTreeSet::new();
        for root_id in &trie.root_ids {
            let new_root = if other_ends.contains(root_id) {
                canonical
            } else {
                *root_id
            };
            if root_set.insert(new_root) {
                new_roots.push(new_root);
            }
        }
        trie.root_ids = new_roots;
    }
}
