use std::collections::BTreeMap;

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

            // Group by (pop, destination)
            let mut grouped_edges: BTreeMap<(isize, NodeId), (SortedSet, SortedSet)> = BTreeMap::new();

            for (ek, dm) in node.children.iter() {
                for (dest, sids) in dm.iter() {
                    let key = (ek.pop, *dest);
                    let entry = grouped_edges.entry(key).or_insert_with(|| (SortedSet::new(), SortedSet::new()));
                    entry.0.union_inplace(&ek.tokens); // union tokens
                    entry.1.union_inplace(sids);      // union sids
                }
            }

            // Rebuild children from factored edges.
            // We need to group again by (pop, tokens) to form valid edges.
            let mut new_children_intermediate: BTreeMap<(isize, SortedSet), BTreeMap<NodeId, SortedSet>> = BTreeMap::new();

            for ((pop, dest), (tokens, sids)) in grouped_edges {
                let key = (pop, tokens);
                let dm = new_children_intermediate.entry(key).or_insert_with(BTreeMap::new);
                dm.insert(dest, sids);
            }

            // Finalize into BTreeMap<EdgeKey, ...>
            let mut new_children = BTreeMap::new();
            for ((pop, tokens), dm) in new_children_intermediate {
                if !tokens.is_empty() && !dm.is_empty() {
                    new_children.insert(
                        crate::trie3_opt::core::EdgeKey::new(pop, tokens),
                        dm,
                    );
                }
            }

            node.children = new_children;
        }
    }
}
