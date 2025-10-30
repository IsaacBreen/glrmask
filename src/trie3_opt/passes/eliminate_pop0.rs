use std::collections::{BTreeMap, BTreeSet};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

pub struct EliminatePop0ExceptRootsPass;

impl OptimizationPass for EliminatePop0ExceptRootsPass {
    fn name(&self) -> &'static str {
        "EliminatePop0ExceptRoots"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        // Eliminate pop=0 edges on non-root nodes by merging them into predecessors.
        // After this loop, only root nodes may have pop=0 edges.
        let mut changed = true;
        while changed {
            changed = false;
            let node_ids: Vec<_> = trie.node_ids().collect();
            let root_set: BTreeSet<_> = trie.root_ids.iter().copied().collect();

            for b_id in node_ids {
                if root_set.contains(&b_id) {
                    continue;
                }

                // Must clone, as we will be modifying the trie.
                let b_node = if let Some(n) = trie.get_node(b_id) {
                    n.clone()
                } else {
                    continue;
                };

                let mut pop0_edges = BTreeMap::new();
                let mut non_pop0_edges = BTreeMap::new();

                for (ek, dm) in b_node.children() {
                    if ek.pop == 0 {
                        pop0_edges.insert(ek.clone(), dm.clone());
                    } else {
                        non_pop0_edges.insert(ek.clone(), dm.clone());
                    }
                }

                if pop0_edges.is_empty() {
                    continue;
                }

                changed = true;
                let parents_of_b = b_node.parents().clone();

                for (pop0_ek, pop0_dm) in &pop0_edges {
                    for (c_id, sids_bc) in pop0_dm {
                        for (a_id, edges_from_a_to_b) in &parents_of_b {
                            for (ek_ab, sids_ab) in edges_from_a_to_b {
                                let new_tokens = ek_ab.tokens.intersect(&pop0_ek.tokens);
                                if new_tokens.is_empty() { continue; }
                                let new_sids = sids_ab.intersect(sids_bc);
                                if new_sids.is_empty() { continue; }
                                let new_ek = EdgeKey::new(ek_ab.pop, new_tokens);
                                trie.add_edge(*a_id, new_ek, *c_id, new_sids);
                            }
                        }
                    }
                }
                trie.set_children(b_id, non_pop0_edges);
            }
        }
    }
}

