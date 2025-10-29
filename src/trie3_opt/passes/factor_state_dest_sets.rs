use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Factor by clustering identical destination sets per (state, token) at a node.
/// For each node u and each pop > 0:
///   - For every token t that appears at pop, we aggregate its dest->states map DM_t(t).
///   - For each state s, compute D_u,s(t) = {dest | s ∈ DM_t(t)[dest]}.
///   - Group tokens by identical D_u,s(·). For |D|=1 keep direct edges; for |D|>=2 introduce
///     one intermediate per dest-set D and route u --(0,T)--> I_D with sids, then
///     I_D --(p, union_T_over_states)--> d for all d ∈ D with sids = all_states.
///
/// This reduces per-state fanout at u to:
///   - 1 (via the intermediate) for multi-destination patterns, and
///   - 1 (direct) for single-destination patterns after token merging.
pub struct FactorStateDestSetsPass {
    pub max_intermediates_per_pop: usize,
    pub max_depth_from_roots: usize,
    pub min_out_degree: usize,
}

impl FactorStateDestSetsPass {
    pub fn new(max_intermediates_per_pop: usize, max_depth_from_roots: usize, min_out_degree: usize) -> Self {
        Self {
            max_intermediates_per_pop,
            max_depth_from_roots,
            min_out_degree,
        }
    }

    fn collect_targets(&self, trie: &MiniTrie) -> Vec<NodeId> {
        if self.max_depth_from_roots == 0 {
            return trie.root_ids.iter().cloned().collect();
        }
        let mut targets = Vec::new();
        let mut seen: BTreeSet<NodeId> = BTreeSet::new();
        let mut q: VecDeque<(NodeId, usize)> = trie.root_ids.iter().map(|&r| (r, 0usize)).collect();
        while let Some((u, d)) = q.pop_front() {
            if !seen.insert(u) {
                continue;
            }
            if d <= self.max_depth_from_roots {
                targets.push(u);
            }
            if d < self.max_depth_from_roots {
                if let Some(node) = trie.get_node(u) {
                    for (_ek, dm) in node.children() {
                        for (v, _) in dm {
                            q.push_back((*v, d + 1));
                        }
                    }
                }
            }
        }
        targets
    }

    fn factor_one_node_pop(
        &self,
        trie: &mut MiniTrie,
        node_id: NodeId,
        pop: isize,
        edges: &[(EdgeKey, BTreeMap<NodeId, SortedSet>)],
        all_states: &SortedSet,
    ) -> Option<BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>>> {
        // Aggregate per-token dest->states for this pop
        let mut dm_by_token: HashMap<usize, BTreeMap<NodeId, SortedSet>> = HashMap::new();
        for (ek, dm) in edges {
            debug_assert_eq!(ek.pop, pop);
            if ek.tokens.is_empty() {
                continue;
            }
            for t in ek.tokens.iter() {
                let entry = dm_by_token.entry(t).or_default();
                for (dst, sids) in dm.iter() {
                    entry.entry(*dst).or_default().union_inplace(sids);
                }
            }
        }
        if dm_by_token.is_empty() {
            return None;
        }

        // For each token, build per-state destination sets.
        // per_state_groups: state -> { destset(Vec<NodeId>) -> tokens(SortedSet) }
        let mut per_state_groups: HashMap<usize, HashMap<Vec<NodeId>, SortedSet>> = HashMap::new();
        for (t, dm_t) in dm_by_token.into_iter() {
            // Temporary: state -> Vec<dest>
            let mut dests_by_state: HashMap<usize, Vec<NodeId>> = HashMap::new();
            for (dst, sids) in dm_t.into_iter() {
                for s in sids.iter() {
                    dests_by_state.entry(s).or_default().push(dst);
                }
            }
            for (s, mut v) in dests_by_state {
                v.sort_unstable();
                v.dedup();
                if v.is_empty() {
                    continue;
                }
                per_state_groups
                    .entry(s)
                    .or_default()
                    .entry(v)
                    .or_default()
                    .insert(t);
            }
        }
        if per_state_groups.is_empty() {
            return None;
        }

        // Invert: destset -> { tokens -> states }
        let mut destset_to_tokens_to_states: HashMap<Vec<NodeId>, BTreeMap<SortedSet, SortedSet>> =
            HashMap::new();
        for (s, groups) in per_state_groups.into_iter() {
            for (destset, toks) in groups.into_iter() {
                destset_to_tokens_to_states
                    .entry(destset)
                    .or_default()
                    .entry(toks)
                    .or_default()
                    .insert(s);
            }
        }
        if destset_to_tokens_to_states.is_empty() {
            return None;
        }

        // Cap intermediates if necessary to avoid blow-ups.
        let multi_count = destset_to_tokens_to_states
            .keys()
            .filter(|ds| ds.len() >= 2)
            .count();
        if self.max_intermediates_per_pop > 0 && multi_count > self.max_intermediates_per_pop {
            // Skip factoring for this pop if it would explode.
            return None;
        }

        let mut new_children_for_pop: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
        let mut made_change = false;

        for (destset, tokens_to_states) in destset_to_tokens_to_states.into_iter() {
            if destset.is_empty() {
                continue;
            }
            if destset.len() == 1 {
                // Keep direct edges, merging tokens across identical (state groups).
                let d = destset[0];
                for (toks, states) in tokens_to_states.into_iter() {
                    if toks.is_empty() || states.is_empty() {
                        continue;
                    }
                    let key = EdgeKey::new(pop, toks);
                    new_children_for_pop
                        .entry(key)
                        .or_default()
                        .entry(d)
                        .or_default()
                        .union_inplace(&states);
                    made_change = true;
                }
            } else {
                // Create an intermediate specific to this dest-set.
                // Tokens for mid->dest are the union across all states for this dest-set.
                let mut toks_union = SortedSet::new();
                for (toks, _states) in tokens_to_states.iter() {
                    toks_union.union_inplace(toks);
                }
                if toks_union.is_empty() {
                    continue;
                }
                let mid_id = trie.add_node(false);

                // Source -> mid (pop=0), grouped by (tokens -> states).
                for (toks, states) in tokens_to_states.into_iter() {
                    if toks.is_empty() || states.is_empty() {
                        continue;
                    }
                    let key = EdgeKey::new(0, toks);
                    new_children_for_pop
                        .entry(key)
                        .or_default()
                        .entry(mid_id)
                        .or_default()
                        .union_inplace(&states);
                }

                // Mid -> destinations (pop=p) with all_states; tokens are toks_union.
                for &d in destset.iter() {
                    trie.add_edge(mid_id, EdgeKey::new(pop, toks_union.clone()), d, all_states.clone());
                }
                made_change = true;
            }
        }

        if made_change {
            Some(new_children_for_pop)
        } else {
            None
        }
    }
}

impl OptimizationPass for FactorStateDestSetsPass {
    fn name(&self) -> &'static str {
        "FactorStateDestSets"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext<'_>) {
        let targets = self.collect_targets(trie);
        if targets.is_empty() {
            return;
        }
        let all_states = SortedSet::from_iter(0..=ctx.max_state_id);

        for node_id in targets {
            let node = if let Some(n) = trie.get_node(node_id) { n } else { continue };
            if node.out_degree() < self.min_out_degree {
                continue;
            }
            if node.children().is_empty() {
                continue;
            }

            // Group current edges by pop and clone them to work off-borrow.
            let mut by_pop: BTreeMap<isize, Vec<(EdgeKey, BTreeMap<NodeId, SortedSet>)>> = BTreeMap::new();
            for (ek, dm) in node.children() {
                by_pop.entry(ek.pop).or_default().push((ek.clone(), dm.clone()));
            }

            // Build new children map, keeping pop<=0 as-is and replacing pop>0 where factoring made changes.
            let mut new_children_total: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
            // Keep pop<=0 original edges.
            for (ek, dm) in node.children() {
                if ek.pop <= 0 {
                    let entry = new_children_total.entry(ek.clone()).or_default();
                    for (dst, sids) in dm {
                        entry.entry(*dst).or_default().union_inplace(sids);
                    }
                }
            }

            let mut any_change = false;
            for (pop, edges) in by_pop.into_iter() {
                if pop <= 0 {
                    continue; // already kept
                }
                if let Some(new_for_pop) =
                    self.factor_one_node_pop(trie, node_id, pop, &edges, &all_states)
                {
                    // Use rewritten edges for this pop
                    for (ek, dm) in new_for_pop {
                        let entry = new_children_total.entry(ek).or_default();
                        for (dst, sids) in dm {
                            entry.entry(dst).or_default().union_inplace(&sids);
                        }
                    }
                    any_change = true;
                } else {
                    // Keep original edges for this pop if no change
                    for (ek, dm) in edges {
                        let entry = new_children_total.entry(ek).or_default();
                        for (dst, sids) in dm {
                            entry.entry(dst).or_default().union_inplace(&sids);
                        }
                    }
                }
            }

            if any_change {
                trie.set_children(node_id, new_children_total);
            }
        }
    }
}
