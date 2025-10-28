use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Aggressively collapse per-(state, pop) fanout by introducing:
///   - a per-(state, pop) aggregator A_{s,p} with exactly one u->A_{s,p} edge,
///   - per-unique-destset canonical nodes M_D (shared across states for the same pop),
///   - wiring A_{s,p} --(0, tokens_{s,D})--> M_D,
///   - and M_D --(p, union_tokens_for_D)--> each v in D with state set = all_states.
///
/// This guarantees there is at most one source-level edge for each (state, pop) pair,
/// while preserving exact token/state semantics.
pub struct AggressiveStatePopCollapsePass;

impl OptimizationPass for AggressiveStatePopCollapsePass {
    fn name(&self) -> &'static str {
        "AggressiveStatePopCollapse"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let all_states = SortedSet::from_iter(0..=ctx.max_state_id);
        let node_ids: Vec<_> = trie.node_ids().collect();

        for node_id in node_ids {
            let node = if let Some(n) = trie.get_node(node_id) { n } else { continue };
            if node.children().is_empty() {
                continue;
            }

            // Keep pop <= 0 edges as-is; aggressively refactor pop > 0 edges.
            let mut keep_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
            let mut by_pop: BTreeMap<isize, Vec<(EdgeKey, BTreeMap<NodeId, SortedSet>)>> = BTreeMap::new();

            for (ek, dm) in node.children() {
                if ek.pop <= 0 {
                    let entry = keep_children.entry(ek.clone()).or_default();
                    for (dst, sids) in dm {
                        entry.entry(*dst).or_default().union_inplace(sids);
                    }
                } else {
                    by_pop.entry(ek.pop).or_default().push((ek.clone(), dm.clone()));
                }
            }

            // If there are no pop>0 edges, skip.
            if by_pop.is_empty() {
                continue;
            }

            let mut new_children_total = keep_children;

            // Process each pop>0 independently.
            for (pop, edges_for_pop) in by_pop {
                if edges_for_pop.is_empty() {
                    continue;
                }

                // dm_by_token: token -> (dest -> union(states))
                let mut dm_by_token: HashMap<usize, BTreeMap<NodeId, SortedSet>> = HashMap::new();
                for (ek, dm) in &edges_for_pop {
                    if ek.tokens.is_empty() || dm.is_empty() {
                        continue;
                    }
                    for t in ek.tokens.iter() {
                        let entry = dm_by_token.entry(t).or_default();
                        for (dst, sids) in dm {
                            entry.entry(*dst).or_default().union_inplace(sids);
                        }
                    }
                }
                if dm_by_token.is_empty() {
                    continue;
                }

                // For each token t, compute for each state s the destination set D(s,t).
                // Accumulate per state s: destset -> tokens for which that exact destset appears.
                let mut per_state_destset_tokens: HashMap<usize, BTreeMap<Vec<NodeId>, SortedSet>> = HashMap::new();
                for (t, dm_t) in dm_by_token.into_iter() {
                    // state -> Vec<dest>
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
                        per_state_destset_tokens
                            .entry(s)
                            .or_default()
                            .entry(v)
                            .or_default()
                            .insert(t);
                    }
                }
                if per_state_destset_tokens.is_empty() {
                    continue;
                }

                // Group states by identical behavior (behavior = destset->tokens map).
                // behavior_to_states: (destset->tokens map) -> {states}
                let mut behavior_to_states: BTreeMap<BTreeMap<Vec<NodeId>, SortedSet>, SortedSet> =
                    BTreeMap::new();
                for (s, dmap) in per_state_destset_tokens {
                    behavior_to_states.entry(dmap).or_default().insert(s);
                }

                // Canonicalize unique dest-sets across all states (for this pop) into shared intermediates.
                // Also collect the union of tokens per dest-set across all states.
                let mut tokens_union_by_destset: BTreeMap<Vec<NodeId>, SortedSet> = BTreeMap::new();
                for (dmap, _states) in &behavior_to_states {
                    for (dset, toks) in dmap {
                        tokens_union_by_destset
                            .entry(dset.clone())
                            .or_default()
                            .union_inplace(toks);
                    }
                }

                let mut mid_for_destset: BTreeMap<Vec<NodeId>, NodeId> = BTreeMap::new();
                for (dset, toks_union) in &tokens_union_by_destset {
                    if toks_union.is_empty() || dset.is_empty() {
                        continue;
                    }
                    let mid_id = trie.add_node(false);
                    mid_for_destset.insert(dset.clone(), mid_id);
                    // M_D --(pop, union(toks for D))--> v, sids=all_states
                    for &d in dset {
                        trie.add_edge(
                            mid_id,
                            EdgeKey::new(pop, toks_union.clone()),
                            d,
                            all_states.clone(),
                        );
                    }
                }

                // One aggregator per BEHAVIOR group, not per state.
                for (dmap, states) in behavior_to_states {
                    if states.is_empty() {
                        continue;
                    }
                    // Union over all tokens for this behavior
                    let mut all_toks_for_behavior = SortedSet::new();
                    for toks in dmap.values() {
                        all_toks_for_behavior.union_inplace(toks);
                    }
                    if all_toks_for_behavior.is_empty() {
                        continue;
                    }

                    // Per-behavior aggregator
                    let agg_id = trie.add_node(false);

                    // u --(0, union_all_tokens_for_behavior)--> A_{behavior,p}, sids = {states}
                    new_children_total
                        .entry(EdgeKey::new(0, all_toks_for_behavior))
                        .or_default()
                        .entry(agg_id)
                        .or_default()
                        .union_inplace(&states);

                    // For each dest-set of this behavior: A_{behavior,p} --(0, toks_{behavior,D})--> M_D
                    for (dset, toks_for_dset) in dmap {
                        if toks_for_dset.is_empty() {
                            continue;
                        }
                        if let Some(&mid_id) = mid_for_destset.get(&dset) {
                            let key = EdgeKey::new(0, toks_for_dset.clone());
                            trie.add_edge(agg_id, key, mid_id, all_states.clone());
                        }
                    }
                }
            }

            // Replace u's children with the newly built set (pop <= 0 kept, pop > 0 rewritten).
            trie.set_children(node_id, new_children_total);
        }
    }
}
