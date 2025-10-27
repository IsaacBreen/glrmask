use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Reduces state fanout by factoring edges. For each node, it groups outgoing edges
/// by (pop, destination). For each group, it merges the edges by unioning their
/// token sets and state ID sets. This can reduce the number of distinct edges
/// a single (pop, state_id) transition can take, thus lowering state fanout.
pub struct FactorStateFanoutPass;

impl OptimizationPass for FactorStateFanoutPass {
    fn name(&self) -> &'static str {
        "FactorStateFanout"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        for node in trie.nodes.iter_mut() {
            if node.children.is_empty() {
                continue;
            }

            // Group by (pop, canonical destination map) and union tokens. This is sound.
            let mut by_pop: BTreeMap<isize, HashMap<Vec<(NodeId, SortedSet)>, SortedSet>> = BTreeMap::new();

            for (ek, dm) in node.children.iter() {
                let mut dest_vec: Vec<(NodeId, SortedSet)> = dm.iter().map(|(d, s)| (*d, s.clone())).collect();
                dest_vec.sort_unstable(); // NodeId and SortedSet are Ord

                by_pop.entry(ek.pop).or_default()
                    .entry(dest_vec)
                    .or_default()
                    .union_inplace(&ek.tokens);
            }

            // Rebuild children from factored edges.
            let mut new_children = BTreeMap::new();
            for (pop, groups) in by_pop {
                for (dest_vec, tokens) in groups {
                    if tokens.is_empty() { continue; }
                    let dm: BTreeMap<_, _> = dest_vec.into_iter().collect();
                    if !dm.is_empty() {
                        new_children.insert(
                            crate::trie3_opt::core::EdgeKey::new(pop, tokens),
                            dm,
                        );
                    }
                }
            }
            node.children = new_children;
        }
    }
}
