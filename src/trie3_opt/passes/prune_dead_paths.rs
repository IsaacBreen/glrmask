use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Remove edges that cannot lead to an end node by reverse token-liveness propagation.
/// Compared to a plain reachability prune, this pass uses the edge token masks:
/// - Compute a token-liveness set Live[u] for each node u by starting from all end nodes
///   with Live[end] = ALL_TOKENS and propagating backwards through edge masks:
///   Live[pred] |= Live[child] ∧ edge.tokens.
/// - Rebuild each node's edges: for each original edge (pop, L) to v, keep only the
///   tokens in L ∧ Live[v]. If that intersection is empty, drop the mapping.
/// - State sets (per destination) are preserved and unioned across split keys when needed.
pub struct PruneDeadPathsPass;

impl PruneDeadPathsPass {
    #[inline]
    fn all_tokens(ctx: &OptimizationContext) -> SortedSet {
        SortedSet::from_iter(0..=ctx.max_llm_token_id)
    }

    #[inline]
    fn intersect(a: &SortedSet, b: &SortedSet) -> SortedSet {
        a.intersect(b)
    }
}

impl OptimizationPass for PruneDeadPathsPass {
    fn name(&self) -> &'static str {
        "PruneDeadPaths"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let n = trie.nodes.len();
        if n == 0 {
            return;
        }

        // Build predecessor map: node -> Vec<(pred, edge_key)>
        let mut predecessors: HashMap<NodeId, Vec<(NodeId, EdgeKey)>> = HashMap::new();
        for node in &trie.nodes {
            for (ek, dm) in &node.children {
                for (dst, _sids) in dm {
                    predecessors
                        .entry(*dst)
                        .or_default()
                        .push((node.id, ek.clone()));
                }
            }
        }

        // Initialize live sets: empty, except end nodes seeded with full universe of tokens.
        let mut live: HashMap<NodeId, SortedSet> = HashMap::new();
        let all_tokens = Self::all_tokens(ctx);
        let mut worklist: VecDeque<NodeId> = VecDeque::new();

        for node in &trie.nodes {
            if node.end {
                live.insert(node.id, all_tokens.clone());
                worklist.push_back(node.id);
            } else {
                live.insert(node.id, SortedSet::new());
            }
        }

        // Propagate until fixed point
        while let Some(v) = worklist.pop_front() {
            let live_v = live.get(&v).cloned().unwrap_or_else(SortedSet::new);
            if live_v.is_empty() {
                continue;
            }
            if let Some(preds) = predecessors.get(&v) {
                for (u, ek) in preds {
                    let l_from_edge = Self::intersect(&live_v, &ek.tokens);
                    if l_from_edge.is_empty() {
                        continue;
                    }
                    let entry = live.entry(*u).or_insert_with(SortedSet::new);
                    let old_len = entry.len();
                    entry.union_inplace(&l_from_edge);
                    if entry.len() > old_len {
                        worklist.push_back(*u);
                    }
                }
            }
        }

        // Rebuild each node's children using Live[child] ∧ edge.tokens
        for node in trie.nodes.iter_mut() {
            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();

            for (ek, dm) in node.children.clone() {
                for (dst, sids) in dm {
                    let live_dst = live.get(&dst).cloned().unwrap_or_else(SortedSet::new);
                    let live_on_edge = Self::intersect(&ek.tokens, &live_dst);
                    if live_on_edge.is_empty() {
                        continue;
                    }
                    let new_key = EdgeKey::new(ek.pop, live_on_edge);
                    let dm_entry = new_children.entry(new_key).or_insert_with(BTreeMap::new);
                    dm_entry
                        .entry(dst)
                        .and_modify(|e| e.union_inplace(&sids))
                        .or_insert(sids);
                }
            }

            node.children = new_children;
        }
    }
}
