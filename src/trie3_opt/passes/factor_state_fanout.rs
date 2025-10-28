use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::EdgeKey;
use crate::trie3_opt::passes::OptimizationPass;

/// State-aware factoring via token atoms and intermediates.
///
/// For each node and each pop > 0:
///   1) Build token-atoms from the outgoing (pop, tokens) sets so that, within an atom T,
///      membership of each original token-set is uniform (either T ⊆ tokens or disjoint).
///   2) For each atom T, compute a destination map dest -> union(states) over original edges
///      that cover T. Then for each state s appearing at T, compute D(s, T) (the set of dests).
///   3) Group states by identical D(s, T). Accumulate token unions across atoms for identical
///      (dest-set, state-set) groups.
///   4) For a dest-set of size:
///      - 0: nothing to emit.
///      - 1: keep a direct edge (p, unioned_tokens) to that single dest for its state-set.
///      - >=2: create (or reuse) a per-dest-set intermediate I; add one source --(0, tokens)--> I
///        for the state-set; and add I --(p, unioned_tokens_over_all_states_for_this_destset)--> v
///        for each v in the dest-set with sids = all_states.
///
/// This strictly reduces per-state fanout at the source from |D(s, T)| to 1 for atoms where
/// |D(s, T)| >= 2 and merges tokens across atoms wherever (D(s, T)) is identical.
pub struct FactorStateFanoutPass;

impl OptimizationPass for FactorStateFanoutPass {
    fn name(&self) -> &'static str {
        "FactorStateFanout"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        const MAX_ATOMS_PER_POP: usize = 4096;
        let node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in node_ids {
            let node = if let Some(n) = trie.get_node(node_id) { n } else { continue };
            if node.children().is_empty() {
                continue;
            }

            // Group existing edges by pop so we can completely rebuild per-pop.
            let mut by_pop: BTreeMap<isize, Vec<(EdgeKey, BTreeMap<NodeId, SortedSet>)>> =
                BTreeMap::new();
            for (ek, dm) in node.children() {
                by_pop.entry(ek.pop).or_default().push((ek.clone(), dm.clone()));
            }

            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
            let all_states = SortedSet::from_iter(0..= _ctx.max_state_id);

            for (pop, edges) in by_pop {
                // We only factor for pop > 0; keep pop<=0 as-is.
                if pop <= 0 {
                    for (ek, dm) in edges {
                        let entry = new_children.entry(ek).or_default();
                        for (dst, sids) in dm {
                            entry.entry(dst).or_default().union_inplace(&sids);
                        }
                    }
                    continue;
                }

                // Build token universe and atoms for these edges.
                let mut universe = SortedSet::new();
                for (ek, _) in &edges {
                    universe.union_inplace(&ek.tokens);
                }
                if universe.is_empty() {
                    // No tokens for this pop; nothing to do.
                    continue;
                }

                let mut atoms: Vec<SortedSet> = vec![universe.clone()];
                let mut aborted = false;
                for (ek, _) in &edges {
                    let toks = &ek.tokens;
                    if toks.is_empty() {
                        continue;
                    }
                    let mut next_atoms = Vec::with_capacity(atoms.len().saturating_mul(2));
                    for b in &atoms {
                        if !b.intersects(toks) {
                            next_atoms.push(b.clone());
                            continue;
                        }
                        let inter = b.intersect(toks);
                        if !inter.is_empty() {
                            next_atoms.push(inter);
                        }
                        let diff = b.difference(toks);
                        if !diff.is_empty() {
                            next_atoms.push(diff);
                        }
                    }
                    if next_atoms.len() > MAX_ATOMS_PER_POP {
                        aborted = true;
                        break;
                    }
                    atoms = next_atoms;
                }
                if aborted {
                    atoms = vec![universe];
                }

                // Accumulators:
                // - For destset size >=2: per-destset intermediate and edges.
                let mut mid_for_destset: HashMap<Vec<NodeId>, NodeId> = HashMap::new();
                let mut tokens_for_destset_union: HashMap<Vec<NodeId>, SortedSet> = HashMap::new();
                let mut destset_states_to_tokens: HashMap<Vec<NodeId>, HashMap<SortedSet, SortedSet>> =
                    HashMap::new();
                // - For destset size == 1: keep direct edges; union tokens across atoms.
                let mut direct_edges: HashMap<NodeId, HashMap<SortedSet, SortedSet>> = HashMap::new();

                for tset in atoms {
                    // Build dest -> union(states) map for this atom tset.
                    let mut dest_map: BTreeMap<NodeId, SortedSet> = BTreeMap::new();
                    for (ek, dm) in &edges {
                        if !tset.intersects(&ek.tokens) {
                            continue;
                        }
                        for (dst, sids) in dm {
                            dest_map.entry(*dst).or_default().union_inplace(sids);
                        }
                    }
                    if dest_map.is_empty() {
                        continue;
                    }

                    // For this atom, compute D(s, atom) per state s.
                    // Use a sparse mapping only for states that appear.
                    let mut state_to_dests: HashMap<usize, Vec<NodeId>> = HashMap::new();
                    for (dst, sids) in &dest_map {
                        for s in sids.iter() {
                            state_to_dests.entry(s).or_default().push(*dst);
                        }
                    }

                    // Group states by identical dest-set for this atom.
                    let mut local_dset_to_states: HashMap<Vec<NodeId>, SortedSet> = HashMap::new();
                    for (s, mut v) in state_to_dests {
                        v.sort_unstable();
                        v.dedup();
                        if v.is_empty() {
                            continue;
                        }
                        local_dset_to_states.entry(v).or_default().insert(s);
                    }

                    // Accumulate across atoms:
                    for (dset, states) in local_dset_to_states {
                        if dset.len() == 1 {
                            // Single destination: keep direct edge (pop, tokens) -> that dest, for these states.
                            let dst = dset[0];
                            direct_edges
                                .entry(dst)
                                .or_default()
                                .entry(states.clone())
                                .or_default()
                                .union_inplace(&tset);
                        } else {
                            // Multi-destination set: go via an intermediate for this dest-set.
                            tokens_for_destset_union
                                .entry(dset.clone())
                                .or_default()
                                .union_inplace(&tset);
                            destset_states_to_tokens
                                .entry(dset)
                                .or_default()
                                .entry(states)
                                .or_default()
                                .union_inplace(&tset);
                        }
                    }
                }

                // Emit direct edges for single-destination groups (merged across atoms).
                for (dst, states_to_tokens) in direct_edges {
                    for (states, toks) in states_to_tokens {
                        if toks.is_empty() || states.is_empty() {
                            continue;
                        }
                        let key = EdgeKey::new(pop, toks);
                        new_children
                            .entry(key)
                            .or_default()
                            .entry(dst)
                            .or_default()
                            .union_inplace(&states);
                    }
                }

                // Emit intermediates and their edges for multi-destination sets.
                for (dset, states_to_tokens) in destset_states_to_tokens {
                    // Create/reuse intermediate for this dest-set
                    let mid_id = *mid_for_destset
                        .entry(dset.clone())
                        .or_insert_with(|| trie.add_node(false));

                    // Source -> intermediate (pop=0) for each state-group; tokens merged across atoms.
                    for (states, toks) in states_to_tokens {
                        if toks.is_empty() || states.is_empty() {
                            continue;
                        }
                        let key = EdgeKey::new(0, toks);
                        new_children
                            .entry(key)
                            .or_default()
                            .entry(mid_id)
                            .or_default()
                            .union_inplace(&states);
                    }

                    // Intermediate -> destinations (pop=p), state-agnostic, tokens = union across all states for this dest-set.
                    let toks_mid = tokens_for_destset_union
                        .get(&dset)
                        .cloned()
                        .unwrap_or_else(SortedSet::new);
                    if !toks_mid.is_empty() {
                        for dst in dset {
                            trie.add_edge(mid_id, EdgeKey::new(pop, toks_mid.clone()), dst, all_states.clone());
                        }
                    }
                }
            }

            trie.set_children(node_id, new_children);
        }
    }
}
