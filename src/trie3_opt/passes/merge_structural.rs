use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Merge nodes with identical structure via iterative refinement. Signature includes:
/// - End flag
/// - For each edge-key: (pop, tokens) and destination-class aggregated states.
/// This is a simplified variant of the structural merge in the large pipeline, implemented
/// purely over the MiniTrie.
pub struct MergeStructuralPass {
    pub max_iters: usize,
    pub max_atoms_per_pop: usize,
}

impl MergeStructuralPass {
    pub fn new(max_iters: usize, max_atoms_per_pop: usize) -> Self {
        Self {
            max_iters,
            max_atoms_per_pop,
        }
    }
}

impl OptimizationPass for MergeStructuralPass {
    fn name(&self) -> &'static str {
        "MergeStructural"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        // This is an adaptation of the global-atoms merge from the full pipeline.
        let n = trie.nodes.len();
        if n == 0 {
            return;
        }

        // Helper: build global token atoms by pop
        fn build_global_token_atoms_by_pop(
            trie: &MiniTrie,
            max_atoms_per_pop: usize,
        ) -> BTreeMap<isize, Vec<SortedSet>> {
            // Set operation helpers for SortedSet
            fn set_intersection(a: &SortedSet, b: &SortedSet) -> SortedSet {
                let mut intersection = Vec::new();
                let (mut i, mut j) = (0, 0);
                while i < a.0.len() && j < b.0.len() {
                    if a.0[i] < b.0[j] {
                        i += 1;
                    } else if b.0[j] < a.0[i] {
                        j += 1;
                    } else {
                        intersection.push(a.0[i]);
                        i += 1;
                        j += 1;
                    }
                }
                SortedSet(intersection)
            }
            fn set_difference(a: &SortedSet, b: &SortedSet) -> SortedSet {
                let mut difference = Vec::new();
                let (mut i, mut j) = (0, 0);
                while i < a.0.len() {
                    if j < b.0.len() && b.0[j] < a.0[i] {
                        j += 1;
                    } else if j < b.0.len() && b.0[j] == a.0[i] {
                        i += 1;
                        j += 1;
                    } else {
                        difference.push(a.0[i]);
                        i += 1;
                    }
                }
                SortedSet(difference)
            }

            let mut sets_by_pop: BTreeMap<isize, HashSet<SortedSet>> = BTreeMap::new();
            for node in &trie.nodes {
                for ek in node.children.keys() {
                    if !ek.tokens.is_empty() {
                        sets_by_pop.entry(ek.pop).or_default().insert(ek.tokens.clone());
                    }
                }
            }

            let mut atoms_by_pop = BTreeMap::new();
            for (pop, sets) in sets_by_pop {
                let mut universe = SortedSet::new();
                for s in &sets {
                    universe.union_inplace(s);
                }
                if universe.is_empty() {
                    continue;
                }

                let mut atoms = vec![universe];
                for s in &sets {
                    if atoms.len() >= max_atoms_per_pop {
                        break;
                    }
                    let mut next_atoms = Vec::with_capacity(atoms.len() * 2);
                    for a in atoms {
                        let intersection = set_intersection(&a, s);
                        let difference = set_difference(&a, s);
                        if !intersection.is_empty() {
                            next_atoms.push(intersection);
                        }
                        if !difference.is_empty() {
                            next_atoms.push(difference);
                        }
                    }
                    atoms = next_atoms;
                }
                if atoms.len() > max_atoms_per_pop {
                    atoms.sort_by_key(|a| std::cmp::Reverse(a.len()));
                    atoms.truncate(max_atoms_per_pop);
                }
                atoms_by_pop.insert(pop, atoms);
            }
            atoms_by_pop
        }

        let atoms_by_pop = build_global_token_atoms_by_pop(trie, self.max_atoms_per_pop);
        if atoms_by_pop.is_empty() {
            return;
        }

        // Dense ids and raw graph structure
        let mut dense_of: HashMap<NodeId, usize> = HashMap::with_capacity(n);
        let mut old_of: Vec<NodeId> = Vec::with_capacity(n);
        for (i, node) in trie.nodes.iter().enumerate() {
            dense_of.insert(node.id, i);
            old_of.push(node.id);
        }
        type RawEdge = (isize, SortedSet, usize, SortedSet);
        let mut ends: Vec<bool> = vec![false; n];
        let mut raw_edges: Vec<Vec<RawEdge>> = vec![Vec::new(); n];
        for (u_dense, u_id) in old_of.iter().enumerate() {
            let node = trie.get_node(*u_id).unwrap();
            ends[u_dense] = node.end;
            for (ek, dm) in &node.children {
                for (dst_id, sids) in dm {
                    if let Some(v_dense) = dense_of.get(dst_id) {
                        raw_edges[u_dense]
                            .push((ek.pop, ek.tokens.clone(), *v_dense, sids.clone()));
                    }
                }
            }
        }

        // Precompute atom overlaps for each unique token set
        let mut atom_idxs_by_pop_set: BTreeMap<isize, HashMap<SortedSet, Vec<usize>>> =
            BTreeMap::new();
        for (pop, atoms) in &atoms_by_pop {
            let mut pop_sets = HashSet::new();
            for edges in &raw_edges {
                for (p, tokens, _, _) in edges {
                    if *p == *pop {
                        pop_sets.insert(tokens.clone());
                    }
                }
            }
            let mut map = HashMap::new();
            for tokens in pop_sets {
                let mut idxs = Vec::new();
                for (j, atom) in atoms.iter().enumerate() {
                    if !SortedSet::intersection_is_empty(&tokens, atom) {
                        idxs.push(j);
                    }
                }
                map.insert(tokens, idxs);
            }
            atom_idxs_by_pop_set.insert(*pop, map);
        }

        // Longest path distance in pop=0 subgraph to prevent merging nodes in a chain
        let mut pop0_rev_adj: Vec<Vec<usize>> = vec![vec![]; n];
        let mut pop0_out_degree = vec![0; n];
        for u in 0..n {
            for (p, _, v_dense, _) in &raw_edges[u] {
                if *p == 0 {
                    pop0_rev_adj[*v_dense].push(u);
                    pop0_out_degree[u] += 1;
                }
            }
        }
        let mut dist = vec![0; n];
        let mut q: VecDeque<usize> = (0..n).filter(|&i| pop0_out_degree[i] == 0).collect();
        let mut processed_count = 0;
        while let Some(v) = q.pop_front() {
            processed_count += 1;
            for &u in &pop0_rev_adj[v] {
                dist[u] = dist[u].max(1 + dist[v]);
                pop0_out_degree[u] -= 1;
                if pop0_out_degree[u] == 0 {
                    q.push_back(u);
                }
            }
        }
        if processed_count < n {
            let max_dist = n + 1;
            for i in 0..n {
                if pop0_out_degree[i] > 0 {
                    dist[i] = max_dist;
                    q.push_back(i);
                }
            }
            while let Some(v) = q.pop_front() {
                for &u in &pop0_rev_adj[v] {
                    if dist[u] != max_dist {
                        dist[u] = max_dist;
                        q.push_back(u);
                    }
                }
            }
        }

        // Initial partition based on end flag and pop=0 distance.
        let mut prev_class: Vec<usize> = vec![0; n];
        {
            let mut class_map: HashMap<(bool, usize), usize> = HashMap::new();
            let mut next_class_id = 0;
            for i in 0..n {
                let key = (ends[i], dist[i]);
                let class_id = *class_map.entry(key).or_insert_with(|| {
                    let id = next_class_id;
                    next_class_id += 1;
                    id
                });
                prev_class[i] = class_id;
            }
        }

        // Partition refinement loop
        for _ in 0..self.max_iters {
            type SigKey = (bool, Vec<((isize, usize), Vec<(usize, SortedSet)>)>);
            let mut sig_to_id: HashMap<SigKey, usize> = HashMap::new();
            let mut new_class: Vec<usize> = vec![0; n];
            let mut next_id = 0usize;
            let mut changes = 0usize;

            for u in 0..n {
                let mut per_token_set_aggr: HashMap<
                    (isize, &SortedSet),
                    BTreeMap<usize, SortedSet>,
                > = HashMap::new();
                for (p, tokens, v_dense, sids) in &raw_edges[u] {
                    if !atoms_by_pop.contains_key(p) {
                        continue;
                    }
                    let dest_class = prev_class[*v_dense];
                    per_token_set_aggr
                        .entry((*p, tokens))
                        .or_default()
                        .entry(dest_class)
                        .or_default()
                        .union_inplace(sids);
                }

                let mut per_atom_aggr: BTreeMap<(isize, usize), BTreeMap<usize, SortedSet>> =
                    BTreeMap::new();
                for ((pop, tokens), dm) in per_token_set_aggr {
                    if let Some(pop_map) = atom_idxs_by_pop_set.get(&pop) {
                        if let Some(atom_idxs) = pop_map.get(tokens) {
                            for &j in atom_idxs {
                                let entry = per_atom_aggr.entry((pop, j)).or_default();
                                for (dest_class, sids) in &dm {
                                    entry.entry(*dest_class).or_default().union_inplace(sids);
                                }
                            }
                        }
                    }
                }

                let sig_entries: Vec<_> = per_atom_aggr
                    .into_iter()
                    .map(|(k, m)| (k, m.into_iter().collect()))
                    .collect();
                let sig: SigKey = (ends[u], sig_entries);

                let class_id = *sig_to_id.entry(sig).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                new_class[u] = class_id;
                if new_class[u] != prev_class[u] {
                    changes += 1;
                }
            }
            prev_class = new_class;
            if changes == 0 {
                break;
            }
        }

        // Representatives for final classes
        let final_partition = prev_class;
        let num_classes = final_partition.iter().max().map_or(0, |m| m + 1);
        let mut representatives: Vec<Option<NodeId>> = vec![None; num_classes];
        for (u_dense, &class_id) in final_partition.iter().enumerate() {
            if representatives[class_id].is_none() {
                representatives[class_id] = Some(old_of[u_dense]);
            }
        }
        let mut node_to_rep: HashMap<NodeId, NodeId> = HashMap::with_capacity(n);
        for (u_dense, &class_id) in final_partition.iter().enumerate() {
            node_to_rep.insert(old_of[u_dense], representatives[class_id].unwrap());
        }

        // Rewire all edges to class representatives
        for node in trie.nodes.iter_mut() {
            for dm in node.children.values_mut() {
                let old_dm = std::mem::take(dm);
                for (dst, sids) in old_dm {
                    if let Some(rep) = node_to_rep.get(&dst) {
                        dm.entry(*rep).or_default().union_inplace(&sids);
                    }
                }
            }
        }

        // Rebuild representative edges deterministically
        for class_id in 0..num_classes {
            if let Some(rep_id) = representatives[class_id] {
                let u_dense = final_partition.iter().position(|&c| c == class_id).unwrap();
                let mut aggr: BTreeMap<(isize, SortedSet, usize), SortedSet> = BTreeMap::new();
                for (p, tokens, v_dense, sids) in &raw_edges[u_dense] {
                    let dest_class = final_partition[*v_dense];
                    aggr.entry((*p, tokens.clone(), dest_class))
                        .or_default()
                        .union_inplace(sids);
                }
                let mut new_children = BTreeMap::new();
                for ((p, tokens, dest_class), sids) in aggr {
                    if let Some(dst_rep) = representatives[dest_class] {
                        new_children
                            .entry(EdgeKey::new(p, tokens))
                            .or_default()
                            .entry(dst_rep)
                            .or_default()
                            .union_inplace(&sids);
                    }
                }
                let rep_node = trie.get_node_mut(rep_id).unwrap();
                rep_node.children = new_children;
            }
        }

        // Remap roots and clean up
        for r in trie.roots.values_mut() {
            *r = *node_to_rep.get(r).unwrap();
        }
        trie.prune_unreachable();
    }
}
