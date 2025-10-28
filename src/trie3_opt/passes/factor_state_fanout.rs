use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::EdgeKey;
use crate::trie3_opt::passes::OptimizationPass;

/// Aggressively reduces state fanout by factoring per-state token coverage.
/// For each node and for each (pop, dest):
///   - For every state s that appears on any outgoing edge to `dest`, compute U_s,
///     the union of token sets across all such edges.
///   - States that share the same U_s are grouped together and emitted as a single
///     edge with key (pop, U_s) and dest -> {grouped states}.
/// This guarantees that, for a fixed (node, pop, dest, state), the state appears
/// in at most one outgoing transition from the node, often cutting state fanout
/// dramatically while preserving exact semantics.
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

            // Build: pop -> dest -> (state -> unioned tokens)
            let mut by_pop_dest_state: HashMap<isize, HashMap<NodeId, HashMap<usize, SortedSet>>> =
                HashMap::new();

            for (ek, dm) in node.children() {
                let pop = ek.pop;
                let tokens = &ek.tokens;
                let dest_map = by_pop_dest_state.entry(pop).or_default();
                for (dest, states) in dm {
                    let s2t = dest_map.entry(*dest).or_default();
                    for s in states.iter() {
                        s2t.entry(s)
                            .or_insert_with(SortedSet::new)
                            .union_inplace(tokens);
                    }
                }
            }

            // Reconstruct children from the per-state coverage, grouping states by identical token sets.
            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();

            for (pop, dest_map) in by_pop_dest_state {
                for (dest, s2toks) in dest_map {
                    if s2toks.is_empty() {
                        continue;
                    }

                    // tokens -> states
                    let mut tokens_to_states: HashMap<SortedSet, SortedSet> = HashMap::new();
                    for (state, toks) in s2toks {
                        if toks.is_empty() {
                            continue;
                        }
                        tokens_to_states
                            .entry(toks)
                            .or_insert_with(SortedSet::new)
                            .insert(state);
                    }

                    for (tokens, states) in tokens_to_states {
                        if tokens.is_empty() || states.is_empty() {
                            continue;
                        }
                        let key = crate::trie3_opt::core::EdgeKey::new(pop, tokens);
                        new_children.entry(key).or_default().insert(dest, states);
                    }
                }
            }

            trie.set_children(node_id, new_children);
        }
    }
}
