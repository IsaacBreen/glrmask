use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Compresses unary chains in the trie.
/// A unary chain occurs when a node `u` has exactly one predecessor `p` and one
/// successor `v`, and the edge keys `p->u` and `u->v` are identical.
/// This pass rewires `p` to point directly to `v`, bypassing `u`.
pub struct CompressChainsPass;

impl OptimizationPass for CompressChainsPass {
    fn name(&self) -> &'static str {
        "CompressChains"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        if trie.nodes.is_empty() {
            return;
        }

        for _ in 0..8 {
            // Iterate a few times to handle chains of chains.
            let mut indegree: HashMap<NodeId, usize> = HashMap::new();
            let mut outdegree: HashMap<NodeId, usize> = HashMap::new();
            let mut incoming_edges: HashMap<NodeId, Vec<(NodeId, EdgeKey, SortedSet)>> =
                HashMap::new();

            for node in &trie.nodes {
                *outdegree.entry(node.id).or_default() += node.out_degree();
                for (ek, dm) in &node.children {
                    for (dst, sids) in dm {
                        *indegree.entry(*dst).or_default() += 1;
                        incoming_edges
                            .entry(*dst)
                            .or_default()
                            .push((node.id, ek.clone(), sids.clone()));
                    }
                }
            }

            let mut rewrites: HashMap<NodeId, Vec<(NodeId, EdgeKey, SortedSet)>> = HashMap::new();
            let mut edges_to_remove: HashMap<NodeId, Vec<EdgeKey>> = HashMap::new();
            let mut changed = false;

            for u_node in &trie.nodes {
                let u = u_node.id;
                if trie.root_ids.contains(&u) || u_node.end {
                    continue;
                }

                if indegree.get(&u).cloned().unwrap_or(0) == 1
                    && outdegree.get(&u).cloned().unwrap_or(0) == 1
                {
                    let (p, in_key, s_pu) = incoming_edges.get(&u).unwrap()[0].clone();

                    let (out_key, out_dm) = u_node.children.iter().next().unwrap();
                    let (v, s_uv) = out_dm.iter().next().unwrap();

                    if in_key == *out_key {
                        let s_pv = s_pu.intersect(s_uv);
                        if !s_pv.is_empty() {
                            rewrites.entry(p).or_default().push((*v, in_key.clone(), s_pv));
                            edges_to_remove.entry(p).or_default().push(in_key);
                            changed = true;
                        }
                    }
                }
            }

            if !changed {
                break;
            }

            // Apply changes
            for p_node in trie.nodes.iter_mut() {
                if let Some(keys_to_remove) = edges_to_remove.get(&p_node.id) {
                    p_node.children.retain(|k, _| !keys_to_remove.contains(k));
                }
                if let Some(new_edges) = rewrites.get(&p_node.id) {
                    for (v, key, sids) in new_edges {
                        p_node.children.entry(key.clone()).or_default().entry(*v).and_modify(
                            |s| s.union_inplace(sids),
                        ).or_insert_with(|| sids.clone());
                    }
                }
            }
        }
    }
}
