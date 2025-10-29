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

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext<'_>) {
        let node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in node_ids {
            let n = trie.get_node(node_id).unwrap();
            if n.children().is_empty() {
                continue;
            }

            let old_children = n.children().clone();
            // Cost metric: sum of token set sizes and destination map sizes.
            let old_cost: usize = old_children
                .iter()
                .map(|(ek, dm)| ek.tokens.len() + dm.len())
                .sum();

            // Stage 1: Group by (pop, canonicalized dest-map), unioning token sets.
            let mut by_pop: BTreeMap<isize, HashMap<Vec<(u32, SortedSet)>, SortedSet>> =
                BTreeMap::new();
            for (ek, dm) in &old_children {
                if ek.tokens.is_empty() || dm.is_empty() {
                    continue;
                }
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

            // Stage 2: For each pop, combine groups that yield identical final token sets
            // by unioning their destination maps.
            let mut new_children: BTreeMap<EdgeKey, BTreeMap<u32, SortedSet>> = BTreeMap::new();
            for (pop, groups) in by_pop {
                // tokens -> aggregated destination map
                let mut out_by_tokens: BTreeMap<SortedSet, BTreeMap<u32, SortedSet>> =
                    BTreeMap::new();

                for (dest_vec, tokens) in groups {
                    if tokens.is_empty() {
                        continue;
                    }
                    let dest_out = out_by_tokens.entry(tokens).or_default();
                    for (dst, sids) in dest_vec {
                        if !sids.is_empty() {
                            dest_out.entry(dst).or_default().union_inplace(&sids);
                        }
                    }
                }

                // Emit final edges for this pop
                for (tokens, dm) in out_by_tokens {
                    if !tokens.is_empty() && !dm.is_empty() {
                        new_children.insert(EdgeKey::new(pop, tokens), dm);
                    }
                }
            }

            let new_cost: usize = new_children
                .iter()
                .map(|(ek, dm)| ek.tokens.len() + dm.len())
                .sum();

            if new_cost < old_cost {
                trie.set_children(node_id, new_children);
            }
        }
    }
}
