use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Eliminate pop=0 edges whose source is not a root by composing predecessors:
/// For each B --(0, L_bc, S_bc)--> C and each A --(p_ab, L_ab, S_ab)--> B,
/// add A --(p_ab, L_ab∧L_bc, S_ab∧S_bc)--> C and remove the B->C entry.
/// Iterate to a fixed point.
pub struct EliminatePop0ExceptRootsPass;

impl OptimizationPass for EliminatePop0ExceptRootsPass {
    fn name(&self) -> &'static str {
        "EliminatePop0ExceptRoots"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        let root_set = trie.root_ids.clone();
        let n = trie.num_nodes();
        if n == 0 {
            return;
        }

        loop {
            // Collect pop=0 edges by cloning to avoid borrow issues.
            let mut zero_edges: Vec<(NodeId, SortedSet, NodeId, SortedSet)> = Vec::new();

            for node in trie.nodes.values() {
                for (ek, dm) in &node.children {
                    if ek.pop == 0 {
                        for (dst, sids) in dm {
                            zero_edges.push((node.id, ek.tokens.clone(), *dst, sids.clone()));
                        }
                    }
                }
            }

            if zero_edges.is_empty() { break; }

            let mut removed_this_iter = 0usize;

            for (b, llm_bc, c, s_bc) in zero_edges {
                if root_set.contains(&b) { continue; }

                if let Some(b_node) = trie.get_node(b) {
                    let preds = b_node.parents().clone(); // clone to avoid borrow issues
                    for (a, edges_from_a) in &preds {
                        for (key_ab, s_ab) in edges_from_a {
                            let p_ab = key_ab.pop;
                            let llm_ab = &key_ab.tokens;
                            let new_tokens = llm_ab.intersect(&llm_bc);
                            if new_tokens.is_empty() { continue; }
                            let new_sids = s_ab.intersect(&s_bc);
                            if new_sids.is_empty() { continue; }

                            let key = crate::trie3_opt::core::EdgeKey::new(p_ab, new_tokens.clone());
                            trie.add_edge(*a, key, c, new_sids);
                        }
                    }
                }

                // Remove B -> C under (0, llm_bc)
                let key = crate::trie3_opt::core::EdgeKey::new(0, llm_bc.clone());
                if trie.remove_edge_dest(b, &key, c).is_some() {
                    removed_this_iter += 1;
                }
            }

            if removed_this_iter == 0 {
                break;
            }
        }
    }
}
