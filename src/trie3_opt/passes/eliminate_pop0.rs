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
            // Build incoming map and collect pop=0 edges
            let mut incoming: HashMap<NodeId, Vec<(NodeId, isize, SortedSet, SortedSet)>> = HashMap::new();
            let mut zero_edges: Vec<(NodeId, SortedSet, NodeId, SortedSet)> = Vec::new();

            for node in &trie.nodes {
                for (ek, dm) in &node.children {
                    for (dst, sids) in dm {
                        incoming.entry(*dst).or_default().push((node.id, ek.pop, ek.tokens.clone(), sids.clone()));
                        if ek.pop == 0 {
                            zero_edges.push((node.id, ek.tokens.clone(), *dst, sids.clone()));
                        }
                    }
                }
            }

            if zero_edges.is_empty() {
                break;
            }

            let mut removed_this_iter = 0usize;

            for (b, llm_bc, c, s_bc) in zero_edges {
                if root_set.contains(&b) {
                    // pop=0 from root is allowed; skip
                    continue;
                }
                if let Some(preds) = incoming.get(&b) {
                    for (a, p_ab, llm_ab, s_ab) in preds {
                        let new_tokens = llm_ab.intersect(&llm_bc);
                        if new_tokens.is_empty() {
                            continue;
                        }
                        let new_sids = s_ab.intersect(&s_bc);
                        if new_sids.is_empty() {
                            continue;
                        }
                        // Add/update A -> C with (p_ab, new_tokens) and SIDs = intersection
                        let a_node = &mut trie.nodes[*a as usize];
                        let key = crate::trie3_opt::core::EdgeKey::new(*p_ab, new_tokens.clone());
                        let dm = a_node.children.entry(key).or_insert_with(BTreeMap::new);
                        dm.entry(c)
                            .and_modify(|e| e.union_inplace(&new_sids))
                            .or_insert(new_sids);
                    }
                }

                // Remove B -> C under (0, llm_bc)
                let bnode = &mut trie.nodes[b as usize];
                let key = crate::trie3_opt::core::EdgeKey::new(0, llm_bc.clone());
                if let Some(dm) = bnode.children.get_mut(&key) {
                    if dm.remove(&c).is_some() {
                        removed_this_iter += 1;
                    }
                    if dm.is_empty() {
                        bnode.children.remove(&key);
                    }
                }
            }

            if removed_this_iter == 0 {
                break;
            }
        }
    }
}
