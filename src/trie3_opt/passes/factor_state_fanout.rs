use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::EdgeKey;
use crate::trie3_opt::passes::OptimizationPass;

/// Aggressively reduces state fanout by factoring per-state token coverage.
/// For each node and for each pop:
///   - For every state s, compute its "coverage": a map from destination D to the
///     union of token sets on all edges from the node to D that include s.
///   - States that share the same coverage map are grouped together.
///   - For each group, the coverage map is inverted to group destinations by token set.
///   - A minimal set of new edges is emitted, one for each unique token set,
///     pointing to all corresponding destinations, for the grouped states.
/// This pass is much more aggressive than simply merging edges with identical
/// destinations, as it considers a state's entire routing table for a given pop
/// to find equivalences.
pub struct FactorStateFanoutPass;

impl OptimizationPass for FactorStateFanoutPass {
    fn name(&self) -> &'static str {
        "FactorStateFanout"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        let node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in node_ids {
            let node = trie.get_node(node_id).unwrap();
            if node.children().is_empty() {
                continue;
            }

            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();

            // Group edges by pop to process them together.
            let mut by_pop: BTreeMap<isize, Vec<(&EdgeKey, &BTreeMap<NodeId, SortedSet>)>> =
                BTreeMap::new();
            for (ek, dm) in node.children() {
                by_pop.entry(ek.pop).or_default().push((ek, dm));
            }

            for (pop, edges) in by_pop {
                // 1. Build state_to_coverage: map from a state to its full routing info for this pop.
                // The routing info is a map from destination node to the union of tokens that can lead there.
                let mut state_to_coverage: HashMap<usize, BTreeMap<NodeId, SortedSet>> =
                    HashMap::new();
                for (ek, dm) in edges {
                    for (dest, sids) in dm {
                        for s in sids.iter() {
                            state_to_coverage
                                .entry(s)
                                .or_default()
                                .entry(*dest)
                                .or_default()
                                .union_inplace(&ek.tokens);
                        }
                    }
                }

                // 2. Group states by identical coverage. States with the same coverage are behaviorally
                // equivalent from this node for this pop, and can be grouped.
                let mut coverage_to_states: HashMap<BTreeMap<NodeId, SortedSet>, SortedSet> =
                    HashMap::new();
                for (state, coverage) in state_to_coverage {
                    coverage_to_states
                        .entry(coverage)
                        .or_default()
                        .insert(state);
                }

                // 3. For each group of states, reconstruct the minimal set of edges.
                for (coverage, states) in coverage_to_states {
                    if states.is_empty() {
                        continue;
                    }

                    // Invert the coverage map to group destinations by required token set.
                    let mut tokens_to_dests: BTreeMap<SortedSet, BTreeSet<NodeId>> =
                        BTreeMap::new();
                    for (dest, tokens) in coverage {
                        if !tokens.is_empty() {
                            tokens_to_dests.entry(tokens).or_default().insert(dest);
                        }
                    }

                    // Emit one edge for each distinct token set.
                    for (tokens, dests) in tokens_to_dests {
                        let key = EdgeKey::new(pop, tokens);
                        let dm = new_children.entry(key).or_default();
                        for dest in dests {
                            dm.entry(dest).or_default().union_inplace(&states);
                        }
                    }
                }
            }

            trie.set_children(node_id, new_children);
        }
    }
}
