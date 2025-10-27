use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::trie3_opt::{
    context::OptimizationContext,
    core::{EdgeKey, MiniTrie, NodeId, SortedSet},
    passes::OptimizationPass,
};

pub struct MergeGlobalAtomsPass {
    max_iters: usize,
    max_atoms_per_pop: usize,
}

impl MergeGlobalAtomsPass {
    pub fn new(max_iters: usize, max_atoms_per_pop: usize) -> Self {
        Self { max_iters, max_atoms_per_pop }
    }
}

impl OptimizationPass for MergeGlobalAtomsPass {
    fn name(&self) -> &'static str {
        "MergeGlobalAtoms"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let n = trie.nodes.len();
        if n == 0 {
            return;
        }

        let mut by_pop_masks: BTreeMap<isize, HashSet<SortedSet>> = BTreeMap::new();
        for node in &trie.nodes {
            for (ek, _) in &node.children {
                if !ek.tokens.is_empty() {
                    by_pop_masks.entry(ek.pop).or_default().insert(ek.tokens.clone());
                }
            }
        }

        let universe = SortedSet::from_iter(0..=ctx.max_llm_token_id);
        let mut atoms_by_pop: BTreeMap<isize, Vec<SortedSet>> = BTreeMap::new();
        for (pop, masks) in by_pop_masks {
            let mut blocks = vec![universe.clone()];
            let mut aborted = false;
            for m in masks {
                let mut next_blocks = Vec::new();
                for b in &blocks {
                    let inter = b.intersect(&m);
                    if !inter.is_empty() {
                        next_blocks.push(inter);
                    }
                    let diff_elems: Vec<_> = b.iter().filter(|t| !m.elems.binary_search(t).is_ok()).collect();
                    if !diff_elems.is_empty() {
                        next_blocks.push(SortedSet::from_iter(diff_elems));
                    }
                }
                if self.max_atoms_per_pop > 0 && next_blocks.len() > self.max_atoms_per_pop {
                    aborted = true;
                    break;
                }
                blocks = next_blocks;
            }
            atoms_by_pop.insert(pop, if aborted { vec![universe.clone()] } else { blocks });
        }

        // Precompute atom overlaps for each unique token set.
        let mut atom_idxs_by_pop_mask: BTreeMap<isize, HashMap<SortedSet, Vec<usize>>> =
            BTreeMap::new();
        for (pop, atoms) in &atoms_by_pop {
            let pop_map = atom_idxs_by_pop_mask.entry(*pop).or_default();
            if let Some(masks) = by_pop_masks.get(pop) {
                for mask in masks {
                    let mut idxs = Vec::new();
                    for (i, atom) in atoms.iter().enumerate() {
                        if !mask.intersect(atom).is_empty() {
                            idxs.push(i);
                        }
                    }
                    pop_map.insert(mask.clone(), idxs);
                }
            }
        }

        // Compute pop=0 distance for initial partitioning.
        let mut pop0_adj: Vec<Vec<NodeId>> = vec![vec![]; n];
        let mut pop0_rev_adj: Vec<Vec<NodeId>> = vec![vec![]; n];
        let mut pop0_out_degree = vec![0; n];
        for (u_idx, u_node) in trie.nodes.iter().enumerate() {
            for (ek, dm) in &u_node.children {
                if ek.pop == 0 {
                    for v_id in dm.keys() {
                        pop0_adj[u_idx].push(*v_id);
                        pop0_rev_adj[*v_id as usize].push(u_node.id);
                        pop0_out_degree[u_idx] += 1;
                    }
                }
            }
        }

        let mut dist = vec![0; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if pop0_out_degree[i] == 0 {
                q.push_back(i);
            }
        }

        let mut processed_count = 0;
        while let Some(v_idx) = q.pop_front() {
            processed_count += 1;
            for &u_id in &pop0_rev_adj[v_idx] {
                let u_idx = u_id as usize;
                dist[u_idx] = dist[u_idx].max(1 + dist[v_idx]);
                pop0_out_degree[u_idx] -= 1;
                if pop0_out_degree[u_idx] == 0 {
                    q.push_back(u_idx);
                }
            }
        }

        if processed_count < n {
            let max_dist = n + 1;
            for i in 0..n {
                if pop0_out_degree[i] > 0 {
                    q.push_back(i);
                }
            }
            while let Some(v_idx) = q.pop_front() {
                if dist[v_idx] != max_dist {
                    dist[v_idx] = max_dist;
                    for &u_id in &pop0_rev_adj[v_idx] {
                        q.push_back(u_id as usize);
                    }
                }
            }
        }

        // Partition refinement.
        let mut prev_class: Vec<usize> = vec![0; n];
        let mut class_map: HashMap<(bool, usize), usize> = HashMap::new();
        let mut next_class_id = 0;
        for (i, node) in trie.nodes.iter().enumerate() {
            let key = (node.end, dist[i]);
            let class_id = *class_map.entry(key).or_insert_with(|| {
                let id = next_class_id;
                next_class_id += 1;
                id
            });
            prev_class[i] = class_id;
        }

        for _ in 0..self.max_iters {
            type SigKey = (bool, Vec<((isize, usize), Vec<(usize, SortedSet)>)>);
            let mut sig_to_id: HashMap<SigKey, usize> = HashMap::new();
            let mut next_id = 0;
            let mut new_class = vec![0; n];
            let mut changes = 0;

            for (u_idx, u_node) in trie.nodes.iter().enumerate() {
                let mut per_atom_aggr: BTreeMap<(isize, usize), BTreeMap<usize, SortedSet>> = BTreeMap::new();
                for (ek, dm) in &u_node.children {
                    if let Some(pop_map) = atom_idxs_by_pop_mask.get(&ek.pop) {
                        if let Some(atom_idxs) = pop_map.get(&ek.tokens) {
                            for &atom_idx in atom_idxs {
                                let entry = per_atom_aggr.entry((ek.pop, atom_idx)).or_default();
                                for (v_id, sids) in dm {
                                    let dest_class = prev_class[*v_id as usize];
                                    entry.entry(dest_class).or_default().union_inplace(sids);
                                }
                            }
                        }
                    }
                }
                let sig_entries = per_atom_aggr.into_iter().map(|(k, v)| (k, v.into_iter().collect())).collect();
                let sig = (u_node.end, sig_entries);
                let cid = *sig_to_id.entry(sig).or_insert_with(|| { let id = next_id; next_id += 1; id });
                new_class[u_idx] = cid;
                if new_class[u_idx] != prev_class[u_idx] { changes += 1; }
            }
            prev_class = new_class;
            if changes == 0 { break; }
        }

        // Rewire and rebuild.
        let final_partition = prev_class;
        let num_classes = prev_class.iter().max().map_or(0, |m| m + 1);
        let mut representatives: Vec<Option<NodeId>> = vec![None; num_classes];
        for (u_idx, &class_id) in prev_class.iter().enumerate() {
            if representatives[class_id].is_none() {
                representatives[class_id] = Some(trie.nodes[u_idx].id);
            }
        }
        let mut node_to_rep: HashMap<NodeId, NodeId> = HashMap::new();
        for (u_idx, &class_id) in prev_class.iter().enumerate() {
            node_to_rep.insert(trie.nodes[u_idx].id, representatives[class_id].unwrap());
        }

        // Rebuild representative edges.
        let original_nodes = trie.nodes.clone();
        for class_id in 0..num_classes {
            if let Some(rep_id) = representatives[class_id] {
                let exemplar_idx = final_partition.iter().position(|&c| c == class_id).unwrap();
                let exemplar_node = &original_nodes[exemplar_idx];

                // Aggregate by (pop, tokens, dest_class) -> union of sids
                let mut aggr: BTreeMap<(isize, SortedSet, usize), SortedSet> = BTreeMap::new();
                for (ek, dm) in &exemplar_node.children {
                    for (v_id, sids) in dm {
                        let v_class = final_partition[*v_id as usize];
                        aggr.entry((ek.pop, ek.tokens.clone(), v_class))
                            .or_default()
                            .union_inplace(sids);
                    }
                }

                let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> =
                    BTreeMap::new();
                for ((pop, tokens, dest_class), sids) in aggr {
                    if let Some(dest_rep) = representatives[dest_class] {
                        let key = EdgeKey::new(pop, tokens);
                        new_children.entry(key).or_default().insert(dest_rep, sids);
                    }
                }
                trie.nodes[rep_id as usize].children = new_children;
            }
        }

        // Clear children of non-representatives.
        let rep_set: HashSet<NodeId> = representatives.iter().filter_map(|&x| x).collect();
        for node in &mut trie.nodes {
            if !rep_set.contains(&node.id) {
                node.children.clear();
                node.end = false; // Non-reps are effectively deleted.
            }
        }

        // Remap roots.
        trie.root_ids = trie.root_ids.iter().map(|r| *node_to_rep.get(r).unwrap()).collect();
    }
}
