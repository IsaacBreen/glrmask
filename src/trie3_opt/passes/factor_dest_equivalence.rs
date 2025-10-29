use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Destination-equivalence factoring:
/// For a node u and fixed pop p>0:
///   - Build token atoms so every original token mask is a union of atoms.
///   - For each atom a, aggregate dest->states by unioning across all edges covering a.
///   - Define the signature of a destination v as the vector of (atom_idx, states_at_atom).
///     Destinations with identical signatures are equivalent. This is the coarsest partition
///     ensuring every (state, token) destination set is representable as a union of classes.
///   - Replace u's edges at pop p by:
///       u --(0, T_group)--> mid_C [states = grouped by identical tokens]
///       mid_C --(p, union_tokens_for_C)--> v [for all v in class C, all_states]
///   - Only apply if the reduction exceeds a configured threshold.
pub struct FactorStateDestEquivalencePass {
    pub max_depth_from_roots: usize,
    pub max_atoms_per_pop: usize,
    pub min_gain_edges: usize,
    pub min_out_degree: usize,
}

impl FactorStateDestEquivalencePass {
    pub fn new(
        max_depth_from_roots: usize,
        max_atoms_per_pop: usize,
        min_gain_edges: usize,
        min_out_degree: usize,
    ) -> Self {
        Self {
            max_depth_from_roots,
            max_atoms_per_pop,
            min_gain_edges,
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

    fn build_token_atoms_for_pop(
        &self,
        edges: &[(EdgeKey, BTreeMap<NodeId, SortedSet>)],
    ) -> Vec<SortedSet> {
        let mut universe = SortedSet::new();
        for (ek, _) in edges {
            universe.union_inplace(&ek.tokens);
        }
        if universe.is_empty() {
            return Vec::new();
        }
        let mut atoms: Vec<SortedSet> = vec![universe.clone()];
        let mut aborted = false;
        for (ek, _) in edges {
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
            if self.max_atoms_per_pop > 0 && next_atoms.len() > self.max_atoms_per_pop {
                aborted = true;
                break;
            }
            atoms = next_atoms;
        }
        if aborted {
            vec![universe]
        } else {
            atoms
        }
    }

    fn factor_one_node_pop(
        &self,
        trie: &mut MiniTrie,
        node_id: NodeId,
        pop: isize,
        edges_for_pop: &[(EdgeKey, BTreeMap<NodeId, SortedSet>)],
        all_states: &SortedSet,
    ) -> Option<BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>>> {
        // Build token-atoms.
        let atoms = self.build_token_atoms_for_pop(edges_for_pop);
        if atoms.is_empty() {
            return None;
        }

        // Aggregate per-atom dest -> states (union across edges covering the atom).
        let mut per_atom_dm: Vec<BTreeMap<NodeId, SortedSet>> = vec![BTreeMap::new(); atoms.len()];
        // Precompute for each edge the atom indices it covers.
        let mut atom_idxs_per_edge: Vec<Vec<usize>> = Vec::with_capacity(edges_for_pop.len());
        for (ek, _) in edges_for_pop {
            let mut idxs = Vec::new();
            for (i, a) in atoms.iter().enumerate() {
                if ek.tokens.intersects(a) {
                    idxs.push(i);
                }
            }
            atom_idxs_per_edge.push(idxs);
        }
        for ((ek, dm), idxs) in edges_for_pop.iter().zip(atom_idxs_per_edge.iter()) {
            if idxs.is_empty() {
                continue;
            }
            for &i in idxs {
                let out = &mut per_atom_dm[i];
                for (dst, sids) in dm {
                    out.entry(*dst).or_default().union_inplace(sids);
                }
            }
        }

        // Build destination signatures: dest -> Vec<(atom_idx, states)>
        let mut dest_sig: HashMap<NodeId, Vec<(usize, SortedSet)>> = HashMap::new();
        let mut all_dests: BTreeSet<NodeId> = BTreeSet::new();
        for (i, dm) in per_atom_dm.iter().enumerate() {
            for (dst, sids) in dm {
                all_dests.insert(*dst);
                dest_sig.entry(*dst).or_default().push((i, sids.clone()));
            }
        }
        if all_dests.is_empty() {
            return None;
        }
        for sig in dest_sig.values_mut() {
            sig.sort_unstable_by_key(|(i, _)| *i);
        }

        // Group destinations by identical signatures (equivalence classes).
        let mut class_map: HashMap<Vec<(usize, SortedSet)>, Vec<NodeId>> = HashMap::new();
        for (dst, sig) in dest_sig.into_iter() {
            class_map.entry(sig).or_default().push(dst);
        }

        // If nothing merges (all classes size 1), skip.
        let any_merge = class_map.values().any(|v| v.len() >= 2);
        if !any_merge {
            return None;
        }

        // Compute gain threshold: compare old per-state edge count on multi-dest classes
        // vs new per-state edge count after factoring.
        // old_count_multi = sum over atoms (sum_{dst in multi-classes} |states_i(dst)|)
        let mut is_multi_class_dest: BTreeSet<NodeId> = BTreeSet::new();
        for v in class_map.values() {
            if v.len() >= 2 {
                for &d in v {
                    is_multi_class_dest.insert(d);
                }
            }
        }
        let mut old_count_multi: usize = 0;
        for dm in &per_atom_dm {
            for (dst, sids) in dm {
                if is_multi_class_dest.contains(dst) {
                    old_count_multi += sids.len();
                }
            }
        }

        // new_count_multi = sum over multi-classes C of |union over atoms of states(C, atom)|
        let mut new_count_multi: usize = 0;
        // We also cache tokens per state for actual edge emission later.
        // For each class we will compute: tokens_by_state (state -> tokens), tokens_union_for_class.
        let mut class_computed: Vec<(
            Vec<NodeId>,                               // members
            BTreeMap<usize, SortedSet>,                // tokens_by_state
            SortedSet,                                 // tokens_union_for_class
        )> = Vec::new();

        for (sig, members) in class_map.iter() {
            if members.len() < 2 {
                continue;
            }
            let mut states_union = SortedSet::new();
            let mut tokens_by_state: BTreeMap<usize, SortedSet> = BTreeMap::new();
            let mut tokens_union_for_class = SortedSet::new();
            for (atom_idx, sids) in sig {
                if !sids.is_empty() {
                    // Contribute to per-state tokens
                    let atom_tokens = &atoms[*atom_idx];
                    tokens_union_for_class.union_inplace(atom_tokens);
                    for s in sids.iter() {
                        tokens_by_state.entry(s).or_default().union_inplace(atom_tokens);
                    }
                    states_union.union_inplace(sids);
                }
            }
            new_count_multi += states_union.len();
            class_computed.push((members.clone(), tokens_by_state, tokens_union_for_class));
        }

        // If the gain is too small, skip factoring for this pop.
        if old_count_multi <= new_count_multi || (old_count_multi - new_count_multi) < self.min_gain_edges {
            return None;
        }

        // Build the rewritten children for this pop:
        //  - Keep direct edges for single-dest classes by copying from original edges.
        //  - Use intermediates for multi-dest classes.
        let mut new_children_for_pop: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();

        // Keep direct edges for singletons by copying original edges that point to those dests.
        let mut singleton_dests: BTreeSet<NodeId> = BTreeSet::new();
        for v in class_map.values() {
            if v.len() == 1 {
                singleton_dests.insert(v[0]);
            }
        }
        if !singleton_dests.is_empty() {
            for (ek, dm) in edges_for_pop {
                // ek.pop == pop by construction
                let mut dm2: BTreeMap<NodeId, SortedSet> = BTreeMap::new();
                for (dst, sids) in dm {
                    if singleton_dests.contains(dst) && !sids.is_empty() {
                        dm2.entry(*dst).or_default().union_inplace(sids);
                    }
                }
                if !dm2.is_empty() {
                    new_children_for_pop
                        .entry(EdgeKey::new(pop, ek.tokens.clone()))
                        .or_default()
                        .extend(dm2.into_iter());
                }
            }
        }

        // Emit intermediates for multi-dest classes.
        for (members, tokens_by_state, tokens_union_for_class) in class_computed {
            if members.len() < 2 {
                continue;
            }
            if tokens_union_for_class.is_empty() {
                continue;
            }
            let mid_id = trie.add_node(false);

            // Group states by identical token sets to reduce number of source->mid edges.
            let mut tokens_to_states: BTreeMap<SortedSet, SortedSet> = BTreeMap::new();
            for (s, toks) in tokens_by_state {
                if toks.is_empty() {
                    continue;
                }
                tokens_to_states.entry(toks).or_default().insert(s);
            }
            for (toks, states) in tokens_to_states {
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

            // mid -> destinations at pop=p with all_states; tokens are the union across class.
            for &d in &members {
                trie.add_edge(
                    mid_id,
                    EdgeKey::new(pop, tokens_union_for_class.clone()),
                    d,
                    all_states.clone(),
                );
            }
        }

        if new_children_for_pop.is_empty() {
            None
        } else {
            Some(new_children_for_pop)
        }
    }
}

impl OptimizationPass for FactorStateDestEquivalencePass {
    fn name(&self) -> &'static str {
        "FactorStateDestEquivalence"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
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

            // Group current edges by pop and clone to avoid borrow conflicts.
            let mut by_pop: BTreeMap<isize, Vec<(EdgeKey, BTreeMap<NodeId, SortedSet>)>> = BTreeMap::new();
            for (ek, dm) in node.children() {
                by_pop.entry(ek.pop).or_default().push((ek.clone(), dm.clone()));
            }

            // Rebuild children: keep pop<=0 as-is; replace pop>0 where beneficial.
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
            for (pop, edges_for_pop) in by_pop.into_iter() {
                if pop <= 0 {
                    continue;
                }
                if let Some(new_for_pop) = self.factor_one_node_pop(
                    trie,
                    node_id,
                    pop,
                    &edges_for_pop,
                    &all_states,
                ) {
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
                    for (ek, dm) in edges_for_pop {
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
