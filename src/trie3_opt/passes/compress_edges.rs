use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Group identical destination-maps under the same pop by unioning token masks.
/// This reduces the number of edge-keys by ensuring that, for a fixed pop,
/// edges with identical destination semantics share a single token-set.
pub struct CompressEdgesPass;

impl OptimizationPass for CompressEdgesPass {
    fn name(&self) -> &'static str {
        "CompressEdges"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        for n in trie.nodes.iter_mut() {
            if n.children.is_empty() {
                continue;
            }

            let old_children = n.children.clone();
            let old_cost: usize = old_children
                .iter()
                .map(|(ek, dm)| ek.tokens.len() + dm.len())
                .sum();

            let mut by_pop: BTreeMap<isize, HashMap<Vec<(u32, SortedSet)>, SortedSet>> =
                BTreeMap::new();
            for (ek, dm) in &old_children {
                let mut dest_vec: Vec<(u32, SortedSet)> =
                    dm.iter().map(|(&dst, sids)| (dst, sids.clone())).collect();
                dest_vec.sort_unstable_by_key(|k| k.0);
                by_pop
                    .entry(ek.pop)
                    .or_default()
                    .entry(dest_vec)
                    .or_default()
                    .union_inplace(&ek.tokens);
            }

            let mut new_children: BTreeMap<EdgeKey, BTreeMap<u32, SortedSet>> = BTreeMap::new();
            for (pop, groups) in by_pop {
                for (dest_vec, tokens) in groups {
                    if tokens.is_empty() {
                        continue;
                    }
                    let mut dm = BTreeMap::new();
                    for (dst, sids) in dest_vec {
                        if !sids.is_empty() {
                            dm.insert(dst, sids);
                        }
                    }
                    if !dm.is_empty() {
                        new_children.insert(EdgeKey::new(pop, tokens), dm);
                    }
                }
            }

            let new_cost: usize = new_children
                .iter()
                .map(|(ek, dm)| ek.tokens.len() + dm.len())
                .sum();

            if new_cost < old_cost {
                n.children = new_children;
            }
        }
    }
}
