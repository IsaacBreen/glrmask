use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

pub struct FactorCommonDestinationsPass {
    min_incoming: usize,
}

impl FactorCommonDestinationsPass {
    pub fn new(min_incoming: usize) -> Self {
        Self { min_incoming }
    }
}

impl OptimizationPass for FactorCommonDestinationsPass {
    fn name(&self) -> &'static str {
        "FactorCommonDestinations"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        if self.min_incoming == 0 {
            return;
        }

        let all_states = SortedSet::from_iter(0..=ctx.max_state_id);
        let all_tokens = SortedSet::from_iter(0..=ctx.max_llm_token_id);

        // Map: dest -> { (pop, tokens) -> { states -> [sources] } }
        let mut incoming_map: HashMap<NodeId, HashMap<EdgeKey, HashMap<SortedSet, Vec<NodeId>>>> =
            HashMap::new();

        for src_node in trie.nodes() {
            for (edge_key, dest_map) in src_node.children() {
                for (dest_id, sids) in dest_map {
                    incoming_map
                        .entry(*dest_id)
                        .or_default()
                        .entry(edge_key.clone())
                        .or_default()
                        .entry(sids.clone())
                        .or_default()
                        .push(src_node.id());
                }
            }
        }

        for (dest_id, edges_by_key) in incoming_map {
            for (edge_key, sources_by_sids) in edges_by_key {
                for (sids, sources) in sources_by_sids {
                    if sources.len() >= self.min_incoming {
                        // Create intermediate node
                        let intermediate_id = trie.add_node(false);

                        // Add edge from intermediate to original destination
                        let inter_to_dest_key = EdgeKey::new(edge_key.pop, all_tokens.clone());
                        trie.add_edge(intermediate_id, inter_to_dest_key, dest_id, sids.clone());

                        // Reroute sources to point to intermediate node
                        for src_id in &sources {
                            trie.remove_edge_dest(*src_id, &edge_key, dest_id);

                            // Add new edge to intermediate node. Use the same token-set as the factored key
                            // instead of "all tokens" to avoid massive token-universe expansions in the MiniTrie.
                            // This preserves exact semantics: the original (pop, tokens)->dest is now
                            // source --(0, tokens)--> intermediate --(pop, tokens)--> dest.
                            let none_like_edge_key = EdgeKey::new(0, edge_key.tokens.clone());
                            trie.add_edge(
                                *src_id,
                                none_like_edge_key,
                                intermediate_id,
                                all_states.clone(),
                            );
                        }
                    }
                }
            }
        }
    }
}
