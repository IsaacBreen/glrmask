use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::EdgeKey;
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

            // Step 1: Deconstruct children into a per-destination view.
            // Map from (pop, dest) -> list of (tokens, states)
            let mut per_dest: BTreeMap<(isize, NodeId), Vec<(SortedSet, SortedSet)>> =
                BTreeMap::new();
            for (ek, dm) in &node.children {
                for (dest, states) in dm {
                    per_dest
                        .entry((ek.pop, *dest))
                        .or_default()
                        .push((ek.tokens.clone(), states.clone()));
                }
            }

            // Step 2: For each (pop, dest), merge all edges into one by unioning their token and state sets.
            let mut merged_per_dest: BTreeMap<(isize, NodeId), (SortedSet, SortedSet)> = BTreeMap::new();
            for ((pop, dest), edges) in per_dest {
                let mut merged_tokens = SortedSet::new();
                let mut merged_states = SortedSet::new();
                for (tokens, states) in edges {
                    merged_tokens.union_inplace(&tokens);
                    merged_states.union_inplace(&states);
                }
                if !merged_tokens.is_empty() && !merged_states.is_empty() {
                    merged_per_dest.insert((pop, dest), (merged_tokens, merged_states));
                }
            }

            // Step 3: Reconstruct node.children from the per-destination merged edges.
            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();
            for ((pop, dest), (tokens, states)) in merged_per_dest {
                let key = crate::trie3_opt::core::EdgeKey::new(pop, tokens);
                new_children.entry(key).or_default().insert(dest, states);
            }
            node.children = new_children;
        }
    }
}
