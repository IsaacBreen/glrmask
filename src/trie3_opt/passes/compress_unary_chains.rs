use std::collections::{BTreeSet, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId};
use crate::trie3_opt::passes::OptimizationPass;

pub struct CompressUnaryChainsPass;

impl OptimizationPass for CompressUnaryChainsPass {
    fn name(&self) -> &'static str {
        "CompressUnaryChains"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        let roots_set: BTreeSet<NodeId> = trie.root_ids.iter().cloned().collect();
        let max_iters = 8;
        for _ in 0..max_iters {
            let mut indegree: HashMap<NodeId, usize> = HashMap::new();
            let mut unique_parent: HashMap<NodeId, (NodeId, EdgeKey)> = HashMap::new();
            let mut out_info: HashMap<NodeId, (EdgeKey, NodeId, crate::trie3_opt::core::SortedSet)> =
                HashMap::new();

            for node in trie.nodes() {
                let node_id = node.id();
                for (ek, dm) in node.children() {
                    for (dst, _) in dm {
                        let count = indegree.entry(*dst).or_default();
                        *count += 1;
                        if *count == 1 {
                            unique_parent.insert(*dst, (node.id, ek.clone()));
                        } else {
                            unique_parent.remove(dst);
                        }
                    }
                }
                if node.is_end() {
                    continue;
                }
                let mut total_dests = 0;
                let mut only_edge = None;
                for (ek, dm) in node.children() {
                    for (dst, sids) in dm {
                        total_dests += 1;
                        if total_dests == 1 {
                            only_edge = Some((ek.clone(), *dst, sids.clone()));
                        }
                    }
                }
                if total_dests == 1 {
                    let (ek, dst, sids) = only_edge.unwrap();
                    out_info.insert(node_id, (ek, dst, sids));
                }
            }

            let mut rewrites = vec![];
            for u_id in trie.node_ids() {
                if roots_set.contains(&u_id) {
                    continue;
                }
                if let (Some((out_key, v_id, s_uv)), Some((p_id, in_key))) =
                    (out_info.get(&u_id), unique_parent.get(&u_id))
                {
                    if p_id != &u_id && in_key.pop == out_key.pop && in_key.tokens == out_key.tokens
                    {
                        rewrites.push((*p_id, u_id, *v_id, in_key.clone(), s_uv.clone()));
                    }
                }
            }

            if rewrites.is_empty() {
                break;
            }

            for (p_id, u_id, v_id, key, s_uv) in rewrites {
                if let Some(s_pu) = trie.remove_edge_dest(p_id, &key, u_id) {
                    let s_comp = s_pu.intersect(&s_uv);
                    if !s_comp.is_empty() {
                        trie.add_edge(p_id, key, v_id, s_comp);
                    }
                }
            }
        }
    }
}
