// src/precompute4/conversion.rs
use crate::constraint::{PrecomputeNode3, PrecomputeNode3Index, PrecomputedNodeContents, StateIDBV, Trie3GodWrapper, LLMTokenBV};
use crate::datastructures::trie::Trie;
use crate::precompute4::weighted_automata::{DWA, StateID, Weight};
use crate::tokenizer::TokenizerStateID;
use std::collections::BTreeMap;
use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;

pub fn dwa_to_precompute3(
    dwa: &DWA,
    internal_max_llm_token: usize,
    max_parser_state_id: usize,
) -> (BTreeMap<TokenizerStateID, PrecomputeNode3Index>, Trie3GodWrapper) {
    let trie3_god = Trie3GodWrapper::new();
    let mut precomputed3 = BTreeMap::new();

    // The root of the DWA has transitions on tokenizer state IDs.
    // For each, we start a new Trie3 conversion.
    let start_dwa_state = &dwa.states[dwa.body.start_state];

    for (&char_code, &target_dwa_id) in &start_dwa_state.transitions {
        let mut weight = start_dwa_state.get_weight(char_code).cloned().unwrap();
        if let Some(state_weight) = &start_dwa_state.state_weight {
            weight &= state_weight;
        }
        let tokenizer_state_id = TokenizerStateID(char_code as usize);
        let trie3_root = convert_dwa_subgraph(
            dwa,
            target_dwa_id,
            &trie3_god,
            internal_max_llm_token,
            max_parser_state_id,
            weight,
        );
        precomputed3.insert(tokenizer_state_id, trie3_root);
    }

    (precomputed3, trie3_god)
}

fn convert_dwa_subgraph(
    dwa: &DWA,
    start_dwa_id: StateID,
    trie3_god: &Trie3GodWrapper,
    internal_max_llm_token: usize,
    max_parser_state_id: usize,
    start_weight: Weight,
) -> PrecomputeNode3Index {
    crate::debug!(5, "Converting DWA: {}", dwa);
    let mut dwa_state_to_trie_node: BTreeMap<StateID, PrecomputeNode3Index> = BTreeMap::new();
    let all_parser_states = StateIDBV::ones(max_parser_state_id + 1);

    let end_node = PrecomputeNode3Index::new(trie3_god.insert(Trie::new(PrecomputedNodeContents::leaf())));

    let mut q = vec![start_dwa_id];
    let root_trie_node = PrecomputeNode3Index::new(trie3_god.insert(Trie::new(PrecomputedNodeContents::root(internal_max_llm_token))));
    dwa_state_to_trie_node.insert(start_dwa_id, root_trie_node);

    let mut processed_dwa_states = BTreeMap::new();
    processed_dwa_states.insert(start_dwa_id, root_trie_node);

    while let Some(dwa_id) = q.pop() {
        let src_trie_node = *dwa_state_to_trie_node.get(&dwa_id).unwrap();
        let dwa_state = &dwa.states[dwa_id];
        let pop_len = if dwa_id == start_dwa_id { 0 } else { 1 };

        // Edge to the shared end node for final states
        if let Some(final_weight) = &dwa_state.final_weight {
            let mut weight = final_weight.clone();
            if dwa_id == start_dwa_id {
                weight &= &start_weight;
            }
            weight.clip_max(internal_max_llm_token);
            let weight_bv = LLMTokenBV::from(weight.rsb);
            if !weight_bv.is_empty() {
                let edge_key = (0, weight_bv); // pop 0
                let edge_val = all_parser_states.clone();
                trie3_god.insert_edge_simple(src_trie_node, end_node, edge_key, edge_val);
            }
        }

        let mut handled_exceptions = StateIDBV::zeros();

        for (&char_code, &target_dwa_id) in &dwa_state.transitions {
            if char_code == DEFAULT_TRANSITION_SYMBOL { continue; }
            if char_code < 0 {
                eprint!("All exceptions: {:?}", dwa_state.transitions.keys());
                panic!("Encountered negative transition code {} during conversion. Please run negative-resolution pass before conversion.", char_code);
            }
            let parser_state_id = char_code as usize;
            handled_exceptions.insert(parser_state_id);

            let mut weight = dwa_state.trans_weights.get(&char_code).cloned().unwrap_or_else(Weight::zeros);
            if dwa_id == start_dwa_id {
                weight &= &start_weight;
            }
            weight.clip_max(internal_max_llm_token);
            let weight_bv = LLMTokenBV::from(weight.rsb);

            if !weight_bv.is_empty() {
                let target_trie_node = get_or_create_trie_node(target_dwa_id, &mut q, &mut dwa_state_to_trie_node, trie3_god);
                let edge_key = (pop_len as isize, weight_bv); // pop 1
                let mut edge_val = StateIDBV::zeros();
                edge_val.insert(parser_state_id);
                trie3_god.insert_edge_simple(src_trie_node, target_trie_node, edge_key, edge_val);
            }
        }

        if let Some(&target_dwa_id) = dwa_state.transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            let mut weight = dwa_state.trans_weights.get(&DEFAULT_TRANSITION_SYMBOL).cloned().unwrap_or_else(Weight::zeros);
            if dwa_id == start_dwa_id {
                weight &= &start_weight;
            }
            weight.clip_max(internal_max_llm_token);
            let weight_bv = LLMTokenBV::from(weight.rsb);

            if !weight_bv.is_empty() {
                let target_trie_node = get_or_create_trie_node(target_dwa_id, &mut q, &mut dwa_state_to_trie_node, trie3_god);
                let edge_key = (pop_len as isize, weight_bv); // pop 1
                let edge_val = &all_parser_states - &handled_exceptions;
                if !edge_val.is_empty() {
                    trie3_god.insert_edge_simple(src_trie_node, target_trie_node, edge_key, edge_val);
                }
            }
        }
    }

    root_trie_node
}

fn get_or_create_trie_node(
    dwa_id: StateID,
    q: &mut Vec<StateID>,
    dwa_state_to_trie_node: &mut BTreeMap<StateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) -> PrecomputeNode3Index {
    if let Some(node) = dwa_state_to_trie_node.get(&dwa_id) {
        return *node;
    }
    let contents = PrecomputedNodeContents::internal();
    let trie_node = PrecomputeNode3Index::new(trie3_god.insert(Trie::new(contents)));
    dwa_state_to_trie_node.insert(dwa_id, trie_node);
    q.push(dwa_id);
    trie_node
}
