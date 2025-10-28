use std::collections::BTreeMap;

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

pub struct FactorRootFanoutPass {
    max_atoms_per_pop: usize,
}

impl FactorRootFanoutPass {
    pub fn new(max_atoms_per_pop: usize) -> Self {
        Self { max_atoms_per_pop }
    }
}

impl OptimizationPass for FactorRootFanoutPass {
    fn name(&self) -> &'static str {
        "FactorRootFanout"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let all_states = SortedSet::from_iter(0..=ctx.max_state_id);
        let root_ids: Vec<_> = trie.root_ids.iter().cloned().collect();

        for root_id in root_ids {
            let root_node = trie.nodes.get(&root_id).unwrap().clone();
            if root_node.children.is_empty() {
                continue;
            }

            let mut keep_children = BTreeMap::new();
            let mut by_pop: BTreeMap<isize, Vec<(SortedSet, BTreeMap<NodeId, SortedSet>)>> =
                BTreeMap::new();
            let mut any_p_gt_0 = false;

            for (ek, dm) in root_node.children {
                if ek.pop <= 0 {
                    keep_children.insert(ek, dm);
                } else {
                    any_p_gt_0 = true;
                    by_pop.entry(ek.pop).or_default().push((ek.tokens, dm));
                }
            }

            if !any_p_gt_0 {
                continue;
            }

            let mut new_children = keep_children;
            let mut made_progress = false;

            for (pop, entries) in by_pop {
                let mut universe = SortedSet::new();
                for (tokens, _) in &entries {
                    universe.union_inplace(tokens);
                }
                if universe.is_empty() {
                    continue;
                }

                let mut blocks = vec![universe.clone()];
                let mut aborted = false;
                for (tokens, _) in &entries {
                    let mut next_blocks = Vec::new();
                    for b in &blocks {
                        let inter = b.intersect(tokens);
                        if !inter.is_empty() {
                            next_blocks.push(inter);
                        }
                        let diff_elems: Vec<_> =
                            b.iter().filter(|t| !tokens.elems.binary_search(t).is_ok()).collect();
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
                if aborted || blocks.is_empty() {
                    blocks = vec![universe];
                }

                for b in blocks {
                    let mut dest_agg: BTreeMap<NodeId, SortedSet> = BTreeMap::new();
                    for (tokens, dm) in &entries {
                        if b.intersect(tokens).is_empty() {
                            continue;
                        }
                        for (dst, sids) in dm {
                            dest_agg.entry(*dst).or_default().union_inplace(sids);
                        }
                    }
                    if dest_agg.is_empty() {
                        continue;
                    }

                    let mid_id = trie.add_node(false);
                    let mid_node = trie.nodes.get_mut(&mid_id).unwrap();
                    mid_node.children.insert(EdgeKey::new(pop, b.clone()), dest_agg);

                    new_children
                        .entry(EdgeKey::new(0, b))
                        .or_default()
                        .insert(mid_id, all_states.clone());
                    made_progress = true;
                }
            }

            if made_progress {
                trie.nodes.get_mut(&root_id).unwrap().children = new_children;
            }
        }
    }
}
