use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::tokenizer::TokenizerStateID;
use crate::precompute4::weighted_automata::{DWABody, DWAState, DWAStates, StateID, Weight, DWA};
use std::collections::{BTreeMap, BTreeSet};
use crate::datastructures::trie::Trie;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{compute_all_characterizations, BelowBottomCharacterization};
use crate::precompute4::resolve_negatives::resolve_negative_codes_for_all;
use std::cell::RefCell;
use range_set_blaze::RangeSetBlaze;
use crate::precompute4::utils;
use crate::precompute4::weighted_automata::{NWA, NWAStates, NWABody};

pub type Precomputed4 = BTreeMap<TokenizerStateID, DWA>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullDWABuildError {
    ParserStateIdOutOfRange { state_id: ParserStateID },
}

fn build_template_dwa_from_characterization(
    bb: &BelowBottomCharacterization,
) -> Result<DWA, FullDWABuildError> {
    let mut dwa = DWA::new();
    let w_all = Weight::all();

    // Create a node for each non-terminal, similar to the NWA construction.
    let mut nt_nodes: BTreeMap<NonTerminalID, StateID> = BTreeMap::new();
    for &nt in &bb.all_nts {
        let id = dwa.add_state();
        nt_nodes.insert(nt, id);
    }

    let start = dwa.body.start_state;

    // --- Initial Actions from Start State ---

    for &(initial_state, shift_state) in &bb.initial_shifts {
        let pos_initial = utils::encode_symbol_i16(initial_state)?;
        let neg_initial = utils::encode_negative_i16(initial_state)?;
        let neg_shift = utils::encode_negative_i16(shift_state)?;

        let s1 = dwa.add_state();
        let s2 = dwa.add_state();
        let s3 = dwa.add_state();

        // start --(+initial)--> s1 --(-initial)--> s2 --(-shift)--> s3 (final)
        dwa.states[start].transitions.exceptions.insert(pos_initial, s1);
        dwa.states[start].trans_weights_exceptions.insert(pos_initial, w_all.clone());
        dwa.states[s1].transitions.exceptions.insert(neg_initial, s2);
        dwa.states[s1].trans_weights_exceptions.insert(neg_initial, w_all.clone());
        dwa.states[s2].transitions.exceptions.insert(neg_shift, s3);
        dwa.states[s2].trans_weights_exceptions.insert(neg_shift, w_all.clone());
        dwa.states[s3].final_weight = Some(w_all.clone());
    }

    for &(initial_state, len, nt) in &bb.initial_reduces {
        let pos_initial = utils::encode_symbol_i16(initial_state)?;
        let target_nt_state = *nt_nodes.get(&nt).expect("nt_node must exist for initial_reduce");

        // Create a chain of default transitions for the pops.
        // start --(+initial)--> s1 --(default)*len--> target_nt_state
        let mut from = start;
        let mut next_state = if len == 0 { target_nt_state } else { dwa.add_state() };
        dwa.states[from].transitions.exceptions.insert(pos_initial, next_state);
        dwa.states[from].trans_weights_exceptions.insert(pos_initial, w_all.clone());
        from = next_state;

        for i in 0..len {
            let to = if i == len - 1 { target_nt_state } else { dwa.add_state() };
            dwa.states[from].transitions.default = Some(to);
            dwa.states[from].trans_weight_default = Some(w_all.clone());
            from = to;
        }
    }

    // --- Actions from Non-Terminal States ---

    for (nt, rc) in &bb.reduce_characterizations {
        let src_nt_state = *nt_nodes.get(nt).expect("nt_node must exist for reduce_char");

        for &(revealed_state, len, reduce_nt) in &rc.reveal_and_rereduces {
            let pos_revealed = utils::encode_symbol_i16(revealed_state)?;
            let dst_nt_state = *nt_nodes.get(&reduce_nt).expect("dst nt_node must exist");

            // src --(+revealed)--> s1 --(default)*len--> dst
            let mut from = src_nt_state;
            let mut next_state = if len == 0 { dst_nt_state } else { dwa.add_state() };
            dwa.states[from].transitions.exceptions.insert(pos_revealed, next_state);
            dwa.states[from].trans_weights_exceptions.insert(pos_revealed, w_all.clone());
            from = next_state;

            for i in 0..len {
                let to = if i == len - 1 { dst_nt_state } else { dwa.add_state() };
                dwa.states[from].transitions.default = Some(to);
                dwa.states[from].trans_weight_default = Some(w_all.clone());
                from = to;
            }
        }

        for &(revealed_state, goto_state, shift_state) in &rc.reveal_goto_shift_escapes {
            let pos_revealed = utils::encode_symbol_i16(revealed_state)?;
            let neg_revealed = utils::encode_negative_i16(revealed_state)?;
            let neg_goto = utils::encode_negative_i16(goto_state)?;
            let neg_shift = utils::encode_negative_i16(shift_state)?;

            let s1 = dwa.add_state();
            let s2 = dwa.add_state();
            let s3 = dwa.add_state();
            let s4 = dwa.add_state();

            // src --(+revealed)--> s1 --(-revealed)--> s2 --(-goto)--> s3 --(-shift)--> s4 (final)
            dwa.states[src_nt_state].transitions.exceptions.insert(pos_revealed, s1);
            dwa.states[src_nt_state].trans_weights_exceptions.insert(pos_revealed, w_all.clone());
            dwa.states[s1].transitions.exceptions.insert(neg_revealed, s2);
            dwa.states[s1].trans_weights_exceptions.insert(neg_revealed, w_all.clone());
            dwa.states[s2].transitions.exceptions.insert(neg_goto, s3);
            dwa.states[s2].trans_weights_exceptions.insert(neg_goto, w_all.clone());
            dwa.states[s3].transitions.exceptions.insert(neg_shift, s4);
            dwa.states[s3].trans_weights_exceptions.insert(neg_shift, w_all.clone());
            dwa.states[s4].final_weight = Some(w_all.clone());
        }
    }

    dwa.simplify();

    Ok(dwa)
}

fn build_template_dwas(
    parser: &GLRParser,
) -> Result<BTreeMap<TerminalID, DWA>, FullDWABuildError> {
    let all = compute_all_characterizations(parser);
    let mut out = BTreeMap::new();
    for (term, bb) in all {
        let dwa = build_template_dwa_from_characterization(&bb)?;
        crate::debug!(6, "Built template DWA for terminal {:?}:", term);
        crate::debug!(6, "{}", dwa);
        out.insert(term, dwa);
    }
    Ok(out)
}

fn build_ignore_terminal_dwa() -> DWA {
    // Identity DWA: start is final, no transitions.
    let mut dwa = DWA::new();
    dwa.states[dwa.body.start_state].final_weight = Some(Weight::all());
    dwa
}

// Helper: collect final states of a DWA
fn collect_final_states(dwa: &DWA) -> BTreeSet<usize> {
    let mut finals = BTreeSet::new();
    for (i, st) in dwa.states.0.iter().enumerate() {
        if st.final_weight.is_some() {
            finals.insert(i);
        }
    }
    finals
}

// Helper: join_map for concatenation: map each left final to the right's start.
fn join_map_final_to_start(left: &DWA, right: &DWA) -> BTreeMap<usize, BTreeSet<usize>> {
    let left_final_states = collect_final_states(left);
    let right_start = right.body.start_state;
    let mut join_map: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for lf in left_final_states {
        join_map.insert(lf, BTreeSet::from([right_start]));
    }
    join_map
}

// Public API: precompute4 using NWA-first approach, determinize at the end.
pub fn precompute4(parser: &GLRParser, precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>, trie1_god: &Trie1GodWrapper) -> Precomputed4 {
    use std::cell::RefCell;
    crate::debug!(5, "Starting precompute4...");
    // 1. Build template DWAs for all terminals.
    let template_dwas = match build_template_dwas(parser) {
        Ok(m) => m,
        Err(e) => panic!("Failed to build template DWAs: {:?}", e),
    };
    let ignore_dwa = build_ignore_terminal_dwa();

    // 2. Set up shared NWA state arena.
    let states_arena = RefCell::new(NWAStates::default());

    // 3. Reverse the precompute1 trie.
    let trie1_roots: Vec<_> = precomputed1.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &trie1_roots);

    let leaf_node = all_nodes.iter().find_map(|&idx| {
        idx.read(trie1_god).and_then(|g| if g.value.end { Some(idx) } else { None })
    }).expect("Precompute1 trie must have a single leaf node.");

    let reversed_trie1_god = Trie::reverse(trie1_god, &trie1_roots);
    let reversed_trie_root = leaf_node;

    // 4. Traverse the reversed trie with NWA bodies.
    let initial_nwa_body = {
        let mut states = states_arena.borrow_mut();
        let start = states.add_state();
        states[start].final_weight = Some(Weight::all());
        NWABody { start_state: start }
    };
    let initial_values = vec![(reversed_trie_root, initial_nwa_body)];
    let traversal_data = Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root]).expect("Failed to compute traversal data for reversed trie1");
    let original_trie1_roots_map: BTreeMap<_,_> = precomputed1.iter().map(|(k,v)|(v.clone(), *k)).collect();

    let mut final_bodies: BTreeMap<TokenizerStateID, NWABody> = BTreeMap::new();

    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // step function
        |current_nwa_body: &NWABody, edge_terminal_opt, dest_map| {
            let template_dwa: &DWA = if edge_terminal_opt.is_some() && *edge_terminal_opt != parser.ignore_terminal_id {
                let terminal_id = edge_terminal_opt.unwrap();
                template_dwas.get(&terminal_id).expect_else(|| format!("No template DWA for terminal {:?}", terminal_id))
            } else {
                &ignore_dwa
            };

            let mut results: Vec<(PrecomputeNode1Index, NWABody)> = Vec::new();
            for (dest_idx, llm_token_bv) in dest_map.iter() {
                let mut states = states_arena.borrow_mut();

                // Convert template DWA to NWA and copy it into the arena
                let template_nwa = NWA::from_dwa(template_dwa);
                crate::debug!(5, "Applying template NWA for terminal {:?} with epsilon gate weight {:?}...", edge_terminal_opt, llm_token_bv);
                let (template_start_in_arena, _) = states.copy_subgraph_from(&template_nwa.states, template_nwa.body.start_state);
                crate::debug!(5, "Template NWA copied into arena. Current arena size: {} states.", states.0.len());
                let left_body = NWABody { start_state: template_start_in_arena };

                // Concatenate: left then current (right) via epsilon with weight = llm_token_bv
                crate::debug!(5, "Starting NWA::concatenate_components: left_start={} right_start={}...", left_body.start_state, current_nwa_body.start_state);
                let eps_weight = Weight::from_rsb(llm_token_bv.inner.as_ref().clone());
                let composed_body = NWA::concatenate_components(&mut states, &left_body, current_nwa_body, &eps_weight);
                crate::debug!(5, "NWA::concatenate_components finished. New start state: {}.", composed_body.start_state);

                results.push((*dest_idx, composed_body));
            }
            results
        },
        // merge function: union them via epsilon
        |body1, body2| {
            let mut states = states_arena.borrow_mut();
            crate::debug!(5, "Starting NWA::union_components: body1_start={} body2_start={}...", body1.start_state, body2.start_state);
            *body1 = NWA::union_components(&mut states, body1, &body2);
            crate::debug!(5, "NWA::union_components finished. New start state: {}.", body1.start_state);
        },
        // process function: capture at original roots
        |_node_data, node_idx, nwa_body| {
            if let Some(tokenizer_state_id) = original_trie1_roots_map.get(&node_idx) {
                final_bodies.insert(*tokenizer_state_id, nwa_body.clone());
            }
            Some(nwa_body) // continue traversal
        },
    );

    // Determinize each final NWA subgraph into a DWA
    let final_nwa_states = states_arena.into_inner();
    let mut final_dwas = BTreeMap::new();
    for (tok_id, body) in final_bodies {
        crate::debug!(5, "Determinizing final NWA subgraph for tokenizer state {:?} (start: {})...", tok_id, body.start_state);
        let mut new_dwa = NWA::determinize_components(&final_nwa_states, &body);
        crate::debug!(5, "Determinization produced DWA with {} states. Starting simplification...", new_dwa.states.len());
        new_dwa.simplify();
        crate::debug!(5, "Simplification finished ({} states).", new_dwa.states.len());
        final_dwas.insert(tok_id, new_dwa);
    }

    crate::debug!(5, "Starting resolve_negative_codes_for_all...");
    resolve_negative_codes_for_all(&mut final_dwas);
    crate::debug!(5, "resolve_negative_codes_for_all finished.");

    final_dwas
}
