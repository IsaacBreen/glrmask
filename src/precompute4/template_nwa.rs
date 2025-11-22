use std::collections::BTreeMap;
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, TerminalID};
use crate::precompute4::characterize::{compute_all_characterizations, BelowBottomCharacterization};
use crate::precompute4::utils::{encode_negative_i16, encode_symbol_i16, DEFAULT_TRANSITION_SYMBOL};
use crate::precompute4::weighted_automata::{DWA, NWA, NWABuildError, StateID, Weight};

pub type FullDWABuildError = NWABuildError;

pub(crate) fn build_template_nwa_from_characterization(bb: &BelowBottomCharacterization) -> Result<NWA, FullDWABuildError> {
    let mut nwa = NWA::new(); // One start state at index 0
    let start = nwa.body.start_states[0];
    let w_all = Weight::all();

    let mut nt_nodes: BTreeMap<NonTerminalID, StateID> = BTreeMap::new();
    for &nt in &bb.all_nts { nt_nodes.insert(nt, nwa.add_state()); }

    // Initial Shifts
    for &(ini, shift) in &bb.initial_shifts {
        let (p_ini, n_ini, n_shift) = (encode_symbol_i16(ini).unwrap(), encode_negative_i16(ini).unwrap(), encode_negative_i16(shift).unwrap());
        let (s0, s1, s2, s3) = (nwa.add_state(), nwa.add_state(), nwa.add_state(), nwa.add_state());
        nwa.add_epsilon(start, s0, w_all.clone());
        nwa.add_transition(s0, p_ini, s1, w_all.clone())?;
        nwa.add_transition(s1, n_ini, s2, w_all.clone())?;
        nwa.add_transition(s2, n_shift, s3, w_all.clone())?;
        nwa.states[s3].final_weight = Some(w_all.clone());
    }

    // Initial Reduces
    for &(ini, len, nt) in &bb.initial_reduces {
        let p_ini = encode_symbol_i16(ini).unwrap();
        let target = nt_nodes[&nt];
        let s0 = nwa.add_state();
        nwa.add_epsilon(start, s0, w_all.clone());
        let mut curr = s0;
        let next = if len == 0 { target } else { nwa.add_state() };
        nwa.add_transition(curr, p_ini, next, w_all.clone())?;
        curr = next;
        for i in 0..len {
            let to = if i == len - 1 { target } else { nwa.add_state() };
            nwa.add_transition(curr, DEFAULT_TRANSITION_SYMBOL, to, w_all.clone())?;
            curr = to;
        }
    }

    // Non-terminal characterizations
    for (nt, rc) in &bb.reduce_characterizations {
        let src = nt_nodes[nt];
        // Reveal & Rereduce
        for &(revealed, len, reduce_nt) in &rc.reveal_and_rereduces {
            let p_rev = encode_symbol_i16(revealed).unwrap();
            let dst = nt_nodes[&reduce_nt];
            let s0 = nwa.add_state();
            nwa.add_epsilon(src, s0, w_all.clone());
            let mut curr = s0;
            let next = if len == 0 { dst } else { nwa.add_state() };
            nwa.add_transition(curr, p_rev, next, w_all.clone())?;
            curr = next;
            for i in 0..len {
                let to = if i == len - 1 { dst } else { nwa.add_state() };
                nwa.add_transition(curr, DEFAULT_TRANSITION_SYMBOL, to, w_all.clone())?;
                curr = to;
            }
        }
        // Reveal-Goto-Shift Escapes
        for &(revealed, goto, shift) in &rc.reveal_goto_shift_escapes {
            let (p_rev, n_rev, n_goto, n_shift) = (encode_symbol_i16(revealed).unwrap(), encode_negative_i16(revealed).unwrap(), encode_negative_i16(goto).unwrap(), encode_negative_i16(shift).unwrap());
            let (s0, s1, s2, s3, s4) = (nwa.add_state(), nwa.add_state(), nwa.add_state(), nwa.add_state(), nwa.add_state());
            nwa.add_epsilon(src, s0, w_all.clone());
            nwa.add_transition(s0, p_rev, s1, w_all.clone())?;
            nwa.add_transition(s1, n_rev, s2, w_all.clone())?;
            nwa.add_transition(s2, n_goto, s3, w_all.clone())?;
            nwa.add_transition(s3, n_shift, s4, w_all.clone())?;
            nwa.states[s4].final_weight = Some(w_all.clone());
        }
    }

    Ok(nwa)
}

pub(crate) fn build_template_dwas(parser: &GLRParser) -> Result<BTreeMap<TerminalID, DWA>, FullDWABuildError> {
    let mut out = BTreeMap::new();
    for (term, bb) in compute_all_characterizations(parser) {
        let mut nwa = build_template_nwa_from_characterization(&bb)?;
        nwa.simplify();
        let mut dwa = nwa.determinize_to_dwa_with_rustfst();
        dwa.simplify();
        out.insert(term, dwa);
    }
    Ok(out)
}

pub(crate) fn build_ignore_terminal_dwa() -> DWA {
    let mut dwa = DWA::new();
    dwa.states[dwa.body.start_state].final_weight = Some(Weight::all());
    dwa
}

pub(crate) fn build_epsilon_dwa(weight: Weight) -> DWA {
    let mut dwa = DWA::new();
    dwa.states[dwa.body.start_state].final_weight = Some(weight);
    dwa
}
