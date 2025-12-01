use std::collections::BTreeMap;

use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{compute_all_characterizations, BelowBottomCharacterization};
use crate::precompute4::utils;
use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::precompute4::weighted_automata::{DWA, NWA, NWABuildError, StateID, Weight};

/// Error type for building the full DWA structures used in precompute4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullDWABuildError {
    ParserStateIdOutOfRange { state_id: ParserStateID },
    AutomatonBuild(NWABuildError),
}

impl From<NWABuildError> for FullDWABuildError {
    fn from(e: NWABuildError) -> Self { FullDWABuildError::AutomatonBuild(e) }
}

/// Build a template NWA corresponding to the characterization of a single terminal.
pub fn build_template_nwa_from_characterization(bb: &BelowBottomCharacterization) -> Result<NWA, FullDWABuildError> {
    let mut nwa = NWA::new();
    let w_all = Weight::all();

    // Node for each non-terminal.
    let mut nt_nodes: BTreeMap<NonTerminalID, StateID> = BTreeMap::new();
    for &nt in &bb.all_nts {
        let id = nwa.states.add_state();
        nt_nodes.insert(nt, id);
    }

    // NWA::new() initializes a single start state.
    let start = nwa.body.start_states[0];

    // Initial shifts from start.
    for &(initial_state, shift_state) in &bb.initial_shifts {
        let pos_initial = utils::encode_symbol_i16(initial_state)?;
        let neg_initial = utils::encode_negative_i16(initial_state)?;
        let neg_shift = utils::encode_negative_i16(shift_state)?;

        let s0 = nwa.states.add_state();
        let s1 = nwa.states.add_state();
        let s2 = nwa.states.add_state();
        let s3 = nwa.states.add_state();

        // start --eps--> s0 --(+initial)--> s1 --(-initial)--> s2 --(-shift)--> s3 (final)
        nwa.add_epsilon(start, s0, w_all.clone());
        nwa.add_transition(s0, pos_initial, s1, w_all.clone())?;
        nwa.add_transition(s1, neg_initial, s2, w_all.clone())?;
        nwa.add_transition(s2, neg_shift, s3, w_all.clone())?;
        nwa.states[s3].final_weight = Some(w_all.clone());
    }

    // Initial reduces from start.
    for &(initial_state, len, nt) in &bb.initial_reduces {
        let pos_initial = utils::encode_symbol_i16(initial_state)?;
        let target_nt_state = *nt_nodes.get(&nt).expect("nt_node must exist for initial_reduce");

        // start --eps--> s0 --(+initial)--> s1 --(default)*len--> target_nt_state
        let s0 = nwa.states.add_state();
        nwa.add_epsilon(start, s0, w_all.clone());
        let mut from = s0;
        let next_state = if len == 0 { target_nt_state } else { nwa.states.add_state() };
        nwa.add_transition(from, pos_initial, next_state, w_all.clone())?;
        from = next_state;

        for i in 0..len {
            let to = if i == len - 1 { target_nt_state } else { nwa.states.add_state() };
            nwa.states.add_transition(from, DEFAULT_TRANSITION_SYMBOL, to, w_all.clone())?;
            from = to;
        }
    }

    // Actions from non-terminal states.
    for (nt, rc) in &bb.reduce_characterizations {
        let src_nt_state = *nt_nodes.get(nt).expect("nt_node must exist for reduce_char");

        for &(revealed_state, len, reduce_nt) in &rc.reveal_and_rereduces {
            let pos_revealed = utils::encode_symbol_i16(revealed_state)?;
            let dst_nt_state = *nt_nodes.get(&reduce_nt).expect("dst nt_node must exist");

            // src --eps--> s0 --(+revealed)--> s1 --(default)*len--> dst
            let s0 = nwa.states.add_state();
            nwa.add_epsilon(src_nt_state, s0, w_all.clone());
            let mut from = s0;
            let next_state = if len == 0 { dst_nt_state } else { nwa.states.add_state() };
            nwa.add_transition(from, pos_revealed, next_state, w_all.clone())?;
            from = next_state;

            for i in 0..len {
                let to = if i == len - 1 { dst_nt_state } else { nwa.states.add_state() };
                nwa.states.add_transition(from, DEFAULT_TRANSITION_SYMBOL, to, w_all.clone())?;
                from = to;
            }
        }

        for &(revealed_state, goto_state, shift_state) in &rc.reveal_goto_shift_escapes {
            let pos_revealed = utils::encode_symbol_i16(revealed_state)?;
            let neg_revealed = utils::encode_negative_i16(revealed_state)?;
            let neg_goto = utils::encode_negative_i16(goto_state)?;
            let neg_shift = utils::encode_negative_i16(shift_state)?;

            let s0 = nwa.states.add_state();
            let s1 = nwa.states.add_state();
            let s2 = nwa.states.add_state();
            let s3 = nwa.states.add_state();
            let s4 = nwa.states.add_state();

            // src --eps--> s0 --(+revealed)--> s1 --(-revealed)--> s2 --(-goto)--> s3 --(-shift)--> s4 (final)
            nwa.add_epsilon(src_nt_state, s0, w_all.clone());
            nwa.add_transition(s0, pos_revealed, s1, w_all.clone())?;
            nwa.add_transition(s1, neg_revealed, s2, w_all.clone())?;
            nwa.add_transition(s2, neg_goto, s3, w_all.clone())?;
            nwa.add_transition(s3, neg_shift, s4, w_all.clone())?;
            nwa.states[s4].final_weight = Some(w_all.clone());
        }
    }

    Ok(nwa)
}

/// Build template DWAs for all terminals in the parser.
pub fn build_template_dwas(parser: &GLRParser) -> Result<BTreeMap<TerminalID, DWA>, FullDWABuildError> {
    // NOTE: Removed rayon parallelism - benchmarks showed single-threaded is 3x faster
    // (317ms vs 951ms) due to memory contention in dwa.simplify()/minimize operations.
    let all = compute_all_characterizations(parser);
    crate::debug!(5, "Computed characterizations.");

    // Build templates sequentially - actually faster than parallel due to memory contention
    let results: Result<Vec<_>, _> = all.into_iter().map(|(term, bb)| {
        let nwa = build_template_nwa_from_characterization(&bb)?;
        // Skip nwa.simplify() - let determinize handle it
        let mut dwa = nwa.determinize();
        dwa.simplify();
        crate::debug!(7, "Built template DWA for terminal {:?}:", term);
        Ok((term, dwa))
    }).collect();

    results.map(|vec| {
        let map: BTreeMap<TerminalID, DWA> = vec.into_iter().collect();
        
        // Validation: Ensure at least one Template DFA has a merge (two incoming edges from different sources).
        // This is a critical structural property for the complexity argument.
        let mut found_merge = false;
        for dwa in map.values() {
            let mut incoming: BTreeMap<StateID, std::collections::HashSet<StateID>> = BTreeMap::new();
            for (src, state) in dwa.states.0.iter().enumerate() {
                for (_, &dst) in &state.transitions {
                    incoming.entry(dst).or_default().insert(src);
                }
            }
            
            for sources in incoming.values() {
                if sources.len() >= 2 {
                    found_merge = true;
                    break;
                }
            }
            if found_merge { break; }
        }
        
        if !found_merge {
            println!("Validation Warning: No Template DFA exhibits the 'two incoming edges' rule (merge from different sources). This is expected for simple grammars but might be an issue for complex ones.");
        }

        map
    })
}

/// Identity DWA used for the "ignore" terminal: start is final and there are no transitions.
pub fn build_ignore_terminal_dwa() -> DWA {
    let mut dwa = DWA::new();
    dwa.states[dwa.body.start_state].final_weight = Some(Weight::all());
    dwa
}

/// DWA that accepts the empty string with the given weight.
pub fn build_epsilon_dwa(weight: Weight) -> DWA {
    let mut dwa = DWA::new();
    dwa.states[dwa.body.start_state].final_weight = Some(weight);
    dwa
}
