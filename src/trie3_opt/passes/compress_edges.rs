use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Compress edges by grouping identical destination maps (for a given pop) and unioning token masks.
/// Adopt the rewrite only if a local cost metric improves. The local cost metric for a node is:
///   cost = sum over edges of (token_ranges_count(tokens) + number_of_dest_entries)
///
/// Where token_ranges_count counts the number of contiguous runs in the sorted token set.
/// This approximates the "range" cost used in the full precompute graph (RangeSetBlaze) while
/// operating over the MiniTrie (which uses a compact SortedSet).
pub struct CompressEdgesPass;

impl CompressEdgesPass {
    #[inline]
    fn token_ranges_count(tokens: &SortedSet) -> usize {
        if tokens.elems.is_empty() {
            return 0;
        }
        let mut ranges = 1usize;
        let mut prev = tokens.elems[0];
        for &x in tokens.elems.iter().skip(1) {
            if x != prev + 1 {
                ranges += 1;
            }
            prev = x;
        }
        ranges
    }

    #[inline]
    fn children_cost(children: &BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>>) -> usize {
        let mut ranges_sum = 0usize;
        let mut dests_sum = 0usize;
        for (ek, dm) in children {
            ranges_sum += Self::token_ranges_count(&ek.tokens);
            dests_sum += dm.len();
        }
        ranges_sum + dests_sum
    }
}

impl OptimizationPass for CompressEdgesPass {
    fn name(&self) -> &'static str {
        "CompressEdges"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        for node in trie.nodes.iter_mut() {
            if node.children.is_empty() {
                continue;
            }

            // Snapshot current children for cost calculation and to build candidates.
            let old_snapshot = node.children.clone();
            let old_cost = Self::children_cost(&old_snapshot);

            // First stage: group by (pop, canonicalized dest-map) and union tokens.
            // canonicalized dest-map is a sorted Vec<(dst, states)>
            let mut by_pop: BTreeMap<isize, HashMap<Vec<(NodeId, SortedSet)>, SortedSet>> =
                BTreeMap::new();
            for (ek, dm) in &old_snapshot {
                // Canonical dest vector sorted by node id with unioned states if multiple entries
                let mut dest_vec: Vec<(NodeId, SortedSet)> = Vec::with_capacity(dm.len());
                for (dst, sids) in dm.iter() {
                    dest_vec.push((*dst, sids.clone()));
                }
                // Ensure canonical sorting by (dst, states)
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

            // Second stage: for each pop, combine groups that yield identical final token sets
            // by unioning their destination maps (results in one entry per (pop, tokens)).
            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
            for (pop, groups) in by_pop {
                // tokens -> aggregated destination map
                let mut out_by_tokens: BTreeMap<SortedSet, BTreeMap<NodeId, SortedSet>> =
                    BTreeMap::new();

                for (dest_vec, tokens) in groups {
                    if tokens.is_empty() {
                        continue;
                    }
                    let dest_out = out_by_tokens.entry(tokens).or_insert_with(BTreeMap::new);
                    for (dst, sids) in dest_vec {
                        if !sids.is_empty() {
                            dest_out
                                .entry(dst)
                                .and_modify(|e| e.union_inplace(&sids))
                                .or_insert(sids);
                        }
                    }
                }

                // Emit final edges: (pop, tokens) -> dest map
                for (tokens, dest_btree) in out_by_tokens {
                    if tokens.is_empty() {
                        continue;
                    }
                    if dest_btree.is_empty() {
                        continue;
                    }
                    let key = EdgeKey::new(pop, tokens);
                    let entry = new_children.entry(key).or_insert_with(BTreeMap::new);
                    for (dst, sids) in dest_btree {
                        entry
                            .entry(dst)
                            .and_modify(|e| e.union_inplace(&sids))
                            .or_insert(sids);
                    }
                }
            }

            // Compare costs and adopt only if strictly better.
            let new_cost = Self::children_cost(&new_children);
            if new_cost < old_cost {
                node.children = new_children;
            } else {
                // Keep old structure (no change)
            }
        }
    }
}
