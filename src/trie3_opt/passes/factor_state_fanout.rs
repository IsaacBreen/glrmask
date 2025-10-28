use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::EdgeKey;
use crate::trie3_opt::passes::OptimizationPass;

/// Aggressively reduces state fanout by grouping states with identical transition behaviors.
/// For each node and for each pop:
///   - For every state `s`, compute its "behavior": a map from each destination `d` it can
///     transition to, to the union of token sets on edges that permit the `s->d` transition.
///   - States that share an identical behavior map are grouped together.
///   - New edges are created for each group. The behavior map is inverted to group destinations
///     by their required token sets, forming new `(tokens -> dests)` edges. The grouped states
///     are assigned to these new edges.
/// This is a powerful optimization for reducing the number of distinct edges and state fanout,
/// especially when many states have similar transition patterns.
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

            // Build: pop -> state -> (dest -> unioned tokens)
            let mut by_pop_state: HashMap<isize, HashMap<usize, BTreeMap<NodeId, SortedSet>>> =
                HashMap::new();

            for (ek, dm) in node.children() {
                let pop = ek.pop;
                let tokens = &ek.tokens;
                let state_map = by_pop_state.entry(pop).or_default();
                for (dest, states) in dm {
                    for s in states.iter() {
                        state_map
                            .entry(s)
                            .or_default()
                            .entry(*dest)
                            .or_default()
                            .union_inplace(tokens);
                    }
                }
            }

            // Reconstruct children by grouping states with identical behavior.
            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> =
                BTreeMap::new();

            for (pop, state_map) in by_pop_state {
                // Invert: state -> behavior  to  behavior -> states
                // Behavior is a map from destination to the union of tokens required to get there.
                let mut behavior_to_states: HashMap<BTreeMap<NodeId, SortedSet>, SortedSet> =
                    HashMap::new();
                for (state, behavior) in state_map {
                    if !behavior.is_empty() {
                        behavior_to_states.entry(behavior).or_default().insert(state);
                    }
                }

                for (behavior, states) in behavior_to_states {
                    if states.is_empty() {
                        continue;
                    }

                    // Invert behavior: dest -> tokens  to  tokens -> dests
                    let mut tokens_to_dests: BTreeMap<SortedSet, BTreeSet<NodeId>> =
                        BTreeMap::new();
                    for (dest, tokens) in behavior {
                        if !tokens.is_empty() {
                            tokens_to_dests.entry(tokens).or_default().insert(dest);
                        }
                    }

                    for (tokens, dests) in tokens_to_dests {
                        let key = crate::trie3_opt::core::EdgeKey::new(pop, tokens);
                        let dm = new_children.entry(key).or_default();
                        for dest in dests {
                            dm.insert(dest, states.clone());
                        }
                    }
                }
            }

            trie.set_children(node_id, new_children);
        }
    }
}
