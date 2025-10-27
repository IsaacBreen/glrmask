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

        let mut by_pop_masks: BTreeMap<isize, Vec<SortedSet>> = BTreeMap::new();
        for node in &trie.nodes {
            for (ek, _) in &node.children {
                if !ek.tokens.is_empty() {
                    by_pop_masks.entry(ek.pop).or_default().push(ek.tokens.clone());
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

        let mut prev_class: Vec<usize> = trie.nodes.iter().map(|n| if n.end { 1 } else { 0 }).collect();

        for _ in 0..self.max_iters {
            type SigKey = (bool, Vec<((isize, usize), Vec<(usize, SortedSet)>)>);
            let mut sig_to_id: HashMap<SigKey, usize> = HashMap::new();
            let mut next_id = 0;
            let mut new_class = vec![0; n];
            let mut changes = 0;

            for (u_idx, u_node) in trie.nodes.iter().enumerate() {
                let mut per_atom_aggr: BTreeMap<(isize, usize), BTreeMap<usize, SortedSet>> = BTreeMap::new();
                for (ek, dm) in &u_node.children {
                    if let Some(atoms) = atoms_by_pop.get(&ek.pop) {
                        for (atom_idx, atom) in atoms.iter().enumerate() {
                            if !ek.tokens.intersect(atom).is_empty() {
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

        let num_classes = prev_class.iter().max().map_or(0, |m| m + 1);
        let mut representatives: Vec<Option<NodeId>> = vec![None; num_classes];
        for (u_idx, &class_id) in prev_class.iter().enumerate() {
            if representatives[class_id].is_none() { representatives[class_id] = Some(u_idx as NodeId); }
        }
        let mut node_to_rep: HashMap<NodeId, NodeId> = HashMap::new();
        for (u_idx, &class_id) in prev_class.iter().enumerate() {
            node_to_rep.insert(u_idx as NodeId, representatives[class_id].unwrap());
        }

        for node in &mut trie.nodes {
            let mut new_children = BTreeMap::new();
            for (ek, dm) in &node.children {
                let mut new_dm: BTreeMap<NodeId, SortedSet> = BTreeMap::new();
                for (dst, sids) in dm {
                    let rep = node_to_rep.get(dst).unwrap();
                    new_dm.entry(*rep).or_default().union_inplace(sids);
                }
                if !new_dm.is_empty() {
                    new_children.insert(ek.clone(), new_dm);
                }
            }
            node.children = new_children;
        }
        trie.root_ids = trie.root_ids.iter().map(|r| *node_to_rep.get(r).unwrap()).collect();
    }
}
