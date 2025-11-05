// src/precompute4/conversion.rs
use crate::constraint::{PrecomputeNode3, PrecomputeNode3Index, PrecomputedNodeContents, StateIDBV, Trie3GodWrapper, LLMTokenBV};
use crate::datastructures::trie::Trie;
use crate::precompute4::weighted_automata::{DWA, StateID, Weight};
use std::collections::BTreeMap;

pub fn dwa_to_precompute3(
    dwa: &DWA,
    trie3_god: &Trie3GodWrapper,
    internal_max_llm_token: usize,
    max_parser_state_id: usize,
) -> PrecomputeNode3Index {
    let mut dwa_state_to_trie_node: BTreeMap<StateID, PrecomputeNode3Index> = BTreeMap::new();
    let all_parser_states = StateIDBV::ones(max_parser_state_id + 1);

    // Create all nodes first
    for (dwa_id, dwa_state) in dwa.states.0.iter().enumerate() {
        let live_tokens = LLMTokenBV::ones(internal_max_llm_token + 1);
        // All nodes from DWA states are internal; a shared 'end_node' will represent final states.
        let contents = PrecomputedNodeContents { end: false, live_tokens };
        let trie_node = PrecomputeNode3Index::new(trie3_god.insert(Trie::new(contents)));
        dwa_state_to_trie_node.insert(dwa_id, trie_node);
    }

    let end_node = PrecomputeNode3Index::new(trie3_god.insert(Trie::new(PrecomputedNodeContents::leaf())));

    // Create edges
    for (dwa_id, dwa_state) in dwa.states.0.iter().enumerate() {
        let src_trie_node = *dwa_state_to_trie_node.get(&dwa_id).unwrap();
        let is_root = dwa_id == dwa.body.start_state;
        let pop_len = if is_root { 0 } else { 1 };

        // Edge to the shared end node for final states
        if let Some(final_weight) = &dwa_state.final_weight {
            let final_weight_bv = LLMTokenBV::from(final_weight.rsb.clone());
            if !final_weight_bv.is_empty() {
                let edge_key = (0, final_weight_bv); // pop 0
                let edge_val = all_parser_states.clone();
                trie3_god.insert_edge_simple(src_trie_node, end_node, edge_key, edge_val);
            }
        }

        let mut handled_exceptions = StateIDBV::zeros();

        // Exception transitions
        for (&char_code, &target_dwa_id) in &dwa_state.transitions.exceptions {
            if char_code < 0 {
                eprintln!("Encountered negative transition code {} during conversion. Please run negative-resolution pass before conversion.", char_code);
                continue;
            }
            let target_trie_node = *dwa_state_to_trie_node.get(&target_dwa_id).unwrap();
            let parser_state_id = char_code as usize;
            handled_exceptions.insert(parser_state_id);

            let trans_weight = dwa_state.trans_weights_exceptions.get(&char_code).cloned().unwrap_or_else(Weight::zeros);
            let trans_weight_bv = LLMTokenBV::from(trans_weight.rsb.clone());

            if !trans_weight_bv.is_empty() {
                let edge_key = (pop_len as isize, trans_weight_bv); // pop 1
                let mut edge_val = StateIDBV::zeros();
                edge_val.insert(parser_state_id);
                trie3_god.insert_edge_simple(src_trie_node, target_trie_node, edge_key, edge_val);
            }
        }

        // Default transition
        if let Some(target_dwa_id) = dwa_state.transitions.default {
            let target_trie_node = *dwa_state_to_trie_node.get(&target_dwa_id).unwrap();
            let trans_weight = dwa_state.trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::zeros);
            let trans_weight_bv = LLMTokenBV::from(trans_weight.rsb.clone());

            if !trans_weight_bv.is_empty() {
                let edge_key = (pop_len as isize, trans_weight_bv); // pop 1
                let edge_val = &all_parser_states - &handled_exceptions;
                if !edge_val.is_empty() {
                    trie3_god.insert_edge_simple(src_trie_node, target_trie_node, edge_key, edge_val);
                }
            }
        }
    }

    *dwa_state_to_trie_node.get(&dwa.body.start_state).unwrap()
}
