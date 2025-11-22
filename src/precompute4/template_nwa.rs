// src/precompute4/template_nwa.rs

use std::collections::BTreeMap;
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{compute_all_characterizations, BelowBottomCharacterization};
use crate::precompute4::utils::{self, DEFAULT_TRANSITION_SYMBOL};
use crate::precompute4::weighted_automata::{DWA, NWA, NWABuildError, StateID, Weight};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullDWABuildError {
    ParserStateIdOutOfRange { state_id: ParserStateID },
    AutomatonBuild(NWABuildError),
}

impl From<NWABuildError> for FullDWABuildError { fn from(e: NWABuildError) -> Self { FullDWABuildError::AutomatonBuild(e) } }

pub(crate) fn build_template_nwa_from_characterization(bb: &BelowBottomCharacterization) -> Result<NWA, FullDWABuildError> {
    let mut nwa = NWA::new();
    let w_all = Weight::all();
    let start = nwa.body.start_states[0];

    let mut nt_nodes: BTreeMap<NonTerminalID, StateID> = BTreeMap::new();
    for &nt in &bb.all_nts { nt_nodes.insert(nt, nwa.states.add_state()); }

    for &(init, shift) in &bb.initial_shifts {
        let (s0, s1, s2, s3) = (nwa.add_state(), nwa.add_state(), nwa.add_state(), nwa.add_state());
        nwa.add_epsilon(start, s0, w_all.clone());
        nwa.add_transition(s0, utils::encode_symbol_i16(init)?, s1, w_all.clone())?;
        nwa.add_transition(s1, utils::encode_negative_i16(init)?, s2, w_all.clone())?;
        nwa.add_transition(s2, utils::encode_negative_i16(shift)?, s3, w_all.clone())?;
        nwa.states[s3].final_weight = Some(w_all.clone());
    }

    for &(init, len, nt) in &bb.initial_reduces {
        let s0 = nwa.add_state();
        let target = *nt_nodes.get(&nt).unwrap();
        nwa.add_epsilon(start, s0, w_all.clone());
        let mut curr = s0;
        let next = if len == 0 { target } else { nwa.add_state() };
        nwa.add_transition(curr, utils::encode_symbol_i16(init)?, next, w_all.clone())?;
        curr = next;
        for i in 0..len {
            let to = if i == len - 1 { target } else { nwa.add_state() };
            nwa.add_transition(curr, DEFAULT_TRANSITION_SYMBOL, to, w_all.clone())?;
            curr = to;
        }
    }

    for (nt, rc) in &bb.reduce_characterizations {
        let src = *nt_nodes.get(nt).unwrap();
        for &(revealed, len, reduce_nt) in &rc.reveal_and_rereduces {
            let s0 = nwa.add_state();
            let target = *nt_nodes.get(&reduce_nt).unwrap();
            nwa.add_epsilon(src, s0, w_all.clone());
            let mut curr = s0;
            let next = if len == 0 { target } else { nwa.add_state() };
            nwa.add_transition(curr, utils::encode_symbol_i16(revealed)?, next, w_all.clone())?;
            curr = next;
            for i in 0..len {
                let to = if i == len - 1 { target } else { nwa.add_state() };
                nwa.add_transition(curr, DEFAULT_TRANSITION_SYMBOL, to, w_all.clone())?;
                curr = to;
            }
        }
        for &(revealed, goto, shift) in &rc.reveal_goto_shift_escapes {
            let (s0, s1, s2, s3, s4) = (nwa.add_state(), nwa.add_state(), nwa.add_state(), nwa.add_state(), nwa.add_state());
            nwa.add_epsilon(src, s0, w_all.clone());
            nwa.add_transition(s0, utils::encode_symbol_i16(revealed)?, s1, w_all.clone())?;
            nwa.add_transition(s1, utils::encode_negative_i16(revealed)?, s2, w_all.clone())?;
            nwa.add_transition(s2, utils::encode_negative_i16(goto)?, s3, w_all.clone())?;
            nwa.add_transition(s3, utils::encode_negative_i16(shift)?, s4, w_all.clone())?;
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
