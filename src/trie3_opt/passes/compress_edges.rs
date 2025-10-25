use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, SortedSet};
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
            // First group by pop and canonical destination map
            let mut by_pop: BTreeMap<isize, HashMap<Vec<(u32, SortedSet)>, SortedSet>> = BTreeMap::new();
            for (ek, dm) in n.children.iter() {
                // Canonical dest vector sorted by node id with unioned states if multiple entries
                let mut dest_vec: Vec<(u32, SortedSet)> = Vec::with_capacity(dm.len());
                for (dst, sids) in dm.iter() {
                    dest_vec.push((*dst, sids.clone()));
                }
                // Already in BTreeMap => sorted ordering by key, but ensure canonical anyway
                // by sorting on (dst, lexicographic sids)
                dest_vec.sort_unstable_by(|a, b| {
                    let c = a.0.cmp(&b.0);
                    if c != std::cmp::Ordering::Equal {
                        return c;
                    }
                    a.1.cmp(&b.1)
                });
                let entry = by_pop.entry(ek.pop).or_default();
                entry
                    .entry(dest_vec)
                    .and_modify(|tok| tok.union_inplace(&ek.tokens))
                    .or_insert(ek.tokens.clone());
            }

            // Rebuild children
            let mut new_children: BTreeMap<_, _> = BTreeMap::new();
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
                        new_children.insert(super::super::core::EdgeKey::new(pop, tokens), dm);
                    }
                }
            }
            n.children = new_children;
        }
    }
}
