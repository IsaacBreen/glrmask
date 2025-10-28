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
        let n = trie.nodes.len();
        if n == 0 {
            return;
        }

        loop {
            // Collect pop=0 edges
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

                if let Some(b_node) = trie.nodes.get(&b) {
                    let preds = b_node.parents.clone(); // clone to avoid borrow issues
                    for (a, edges_from_a) in &preds {
                        for (key_ab, s_ab) in edges_from_a {
                            let p_ab = key_ab.pop;
                            let llm_ab = &key_ab.tokens;
                            let new_tokens = llm_ab.intersect(&llm_bc);
                            if new_tokens.is_empty() { continue; }
                            let new_sids = s_ab.intersect(&s_bc);
                            if new_sids.is_empty() { continue; }

                            let a_node = trie.nodes.get_mut(a).unwrap();
                            let key = crate::trie3_opt::core::EdgeKey::new(p_ab, new_tokens.clone());
                            let dm = a_node.children.entry(key).or_default();
                            dm.entry(c).or_default().union_inplace(&new_sids);
                        }
                    }
                }

                // Remove B -> C under (0, llm_bc)
                let bnode = trie.nodes.get_mut(&b).unwrap();
                let key = crate::trie3_opt::core::EdgeKey::new(0, llm_bc.clone());
                if let std::collections::btree_map::Entry::Occupied(mut dm_entry) = bnode.children.entry(key) {
                    if dm_entry.get_mut().remove(&c).is_some() {
                        removed_this_iter += 1;
                    }
                    if dm_entry.get().is_empty() {
                        dm_entry.remove();
                    }
                }
            }

            if removed_this_iter == 0 {
                break;
            }
        }
    }
}
