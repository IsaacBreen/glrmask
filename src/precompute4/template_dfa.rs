use std::collections::BTreeMap;

use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{compute_all_characterizations, TerminalCharacterization};
use crate::precompute4::utils;
use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::dwa_i32::{DWA, NWA, NWABuildError, StateID, Weight};
use crate::dfa_i32::{DFA, NFA};
use crate::dfa_i32::nfa::NFABuildError;

/// Error type for building the Parser DWA structures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullDWABuildError {
    ParserStateIdOutOfRange { state_id: ParserStateID },
    AutomatonBuild(NWABuildError),
    NFABuild(NFABuildError),
}

impl From<NWABuildError> for FullDWABuildError {
    fn from(e: NWABuildError) -> Self { FullDWABuildError::AutomatonBuild(e) }
}

impl From<NFABuildError> for FullDWABuildError {
    fn from(e: NFABuildError) -> Self { FullDWABuildError::NFABuild(e) }
}

/// Build a weighted NWA from a terminal characterization.
/// 
/// The resulting NWA encodes how the terminal interacts with the parse stack:
/// - Initial shifts become labeled transitions
/// - Initial reduces become chains with "pop" transitions (DEFAULT_TRANSITION_SYMBOL)
/// - Reduction cascades are represented by nonterminal state nodes
/// 
/// Note: The "pop" transitions (DEFAULT_TRANSITION_SYMBOL) are what make this
/// conceptually a Weighted Pushdown System - they consume stack symbols.
/// After determinization, these are resolved into a true DWA.
pub fn build_nwa_from_terminal_characterization(tc: &TerminalCharacterization) -> Result<NWA, FullDWABuildError> {
    let mut nwa = NWA::new();
    let w_all = Weight::all();

    // Node for each non-terminal.
    let mut nt_nodes: BTreeMap<NonTerminalID, StateID> = BTreeMap::new();
    for &nt in &tc.all_nts {
        let id = nwa.states.add_state();
        nt_nodes.insert(nt, id);
    }

    // NWA::new() initializes a single start state.
    let start = nwa.body.start_states[0];

    // Initial shifts from start.
    for &(initial_state, shift_state) in &tc.initial_shifts {
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
    for &(initial_state, len, nt) in &tc.initial_reduces {
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
    for (nt, rc) in &tc.reduce_characterizations {
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

/// Build an unweighted NFA from a terminal characterization.
/// 
/// This is the unweighted version of build_nwa_from_terminal_characterization.
/// Since template DFAs don't actually need weights during construction (they get
/// Weight::all() everywhere), we can use simpler/faster unweighted automata.
pub fn build_nfa_from_terminal_characterization(tc: &TerminalCharacterization) -> Result<NFA, FullDWABuildError> {
    let mut nfa = NFA::new();

    // Node for each non-terminal.
    let mut nt_nodes: BTreeMap<NonTerminalID, usize> = BTreeMap::new();
    for &nt in &tc.all_nts {
        let id = nfa.add_state();
        nt_nodes.insert(nt, id);
    }

    // NFA::new() initializes a single start state.
    let start = nfa.body.start_states[0];

    // Initial shifts from start.
    for &(initial_state, shift_state) in &tc.initial_shifts {
        let pos_initial = utils::encode_symbol_i16(initial_state)?;
        let neg_initial = utils::encode_negative_i16(initial_state)?;
        let neg_shift = utils::encode_negative_i16(shift_state)?;

        let s0 = nfa.add_state();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        let s3 = nfa.add_state();

        // start --eps--> s0 --(+initial)--> s1 --(-initial)--> s2 --(-shift)--> s3 (final)
        nfa.add_epsilon(start, s0);
        nfa.add_transition(s0, pos_initial, s1)?;
        nfa.add_transition(s1, neg_initial, s2)?;
        nfa.add_transition(s2, neg_shift, s3)?;
        nfa.set_final(s3);
    }

    // Initial reduces from start.
    for &(initial_state, len, nt) in &tc.initial_reduces {
        let pos_initial = utils::encode_symbol_i16(initial_state)?;
        let target_nt_state = *nt_nodes.get(&nt).expect("nt_node must exist for initial_reduce");

        // start --eps--> s0 --(+initial)--> s1 --(default)*len--> target_nt_state
        let s0 = nfa.add_state();
        nfa.add_epsilon(start, s0);
        let mut from = s0;
        let next_state = if len == 0 { target_nt_state } else { nfa.add_state() };
        nfa.add_transition(from, pos_initial, next_state)?;
        from = next_state;

        for i in 0..len {
            let to = if i == len - 1 { target_nt_state } else { nfa.add_state() };
            nfa.states.add_transition(from, DEFAULT_TRANSITION_SYMBOL, to)?;
            from = to;
        }
    }

    // Actions from non-terminal states.
    for (nt, rc) in &tc.reduce_characterizations {
        let src_nt_state = *nt_nodes.get(nt).expect("nt_node must exist for reduce_char");

        for &(revealed_state, len, reduce_nt) in &rc.reveal_and_rereduces {
            let pos_revealed = utils::encode_symbol_i16(revealed_state)?;
            let dst_nt_state = *nt_nodes.get(&reduce_nt).expect("dst nt_node must exist");

            // src --eps--> s0 --(+revealed)--> s1 --(default)*len--> dst
            let s0 = nfa.add_state();
            nfa.add_epsilon(src_nt_state, s0);
            let mut from = s0;
            let next_state = if len == 0 { dst_nt_state } else { nfa.add_state() };
            nfa.add_transition(from, pos_revealed, next_state)?;
            from = next_state;

            for i in 0..len {
                let to = if i == len - 1 { dst_nt_state } else { nfa.add_state() };
                nfa.states.add_transition(from, DEFAULT_TRANSITION_SYMBOL, to)?;
                from = to;
            }
        }

        for &(revealed_state, goto_state, shift_state) in &rc.reveal_goto_shift_escapes {
            let pos_revealed = utils::encode_symbol_i16(revealed_state)?;
            let neg_revealed = utils::encode_negative_i16(revealed_state)?;
            let neg_goto = utils::encode_negative_i16(goto_state)?;
            let neg_shift = utils::encode_negative_i16(shift_state)?;

            let s0 = nfa.add_state();
            let s1 = nfa.add_state();
            let s2 = nfa.add_state();
            let s3 = nfa.add_state();
            let s4 = nfa.add_state();

            // src --eps--> s0 --(+revealed)--> s1 --(-revealed)--> s2 --(-goto)--> s3 --(-shift)--> s4 (final)
            nfa.add_epsilon(src_nt_state, s0);
            nfa.add_transition(s0, pos_revealed, s1)?;
            nfa.add_transition(s1, neg_revealed, s2)?;
            nfa.add_transition(s2, neg_goto, s3)?;
            nfa.add_transition(s3, neg_shift, s4)?;
            nfa.set_final(s4);
        }
    }

    Ok(nfa)
}

/// Deprecated alias for build_nwa_from_terminal_characterization
#[deprecated(since = "0.3.0", note = "Use build_nwa_from_terminal_characterization instead")]
pub fn build_template_nwa_from_characterization(tc: &TerminalCharacterization) -> Result<NWA, FullDWABuildError> {
    build_nwa_from_terminal_characterization(tc)
}

/// Build template DFAs for all terminals in the parser.
/// 
/// This builds unweighted DFAs which can later be converted to DWAs.
/// Since template automata don't need weights during construction (they get
/// Weight::all() everywhere), building unweighted DFAs is simpler and faster.
/// 
/// A characterization key that excludes the terminal ID, allowing us to share DFAs
/// between terminals with identical grammatical behavior.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CharacterizationKey {
    initial_shifts: std::collections::BTreeSet<(ParserStateID, ParserStateID)>,
    initial_reduces: std::collections::BTreeSet<(ParserStateID, usize, NonTerminalID)>,
    reduce_characterizations: BTreeMap<NonTerminalID, (
        std::collections::BTreeSet<(ParserStateID, usize, NonTerminalID)>,
        std::collections::BTreeSet<(ParserStateID, ParserStateID, ParserStateID)>,
    )>,
    all_nts: std::collections::BTreeSet<NonTerminalID>,
}

impl CharacterizationKey {
    fn from_characterization(tc: &TerminalCharacterization) -> Self {
        CharacterizationKey {
            initial_shifts: tc.initial_shifts.clone(),
            initial_reduces: tc.initial_reduces.clone(),
            reduce_characterizations: tc.reduce_characterizations.iter()
                .map(|(nt, rc)| (*nt, (rc.reveal_and_rereduces.clone(), rc.reveal_goto_shift_escapes.clone())))
                .collect(),
            all_nts: tc.all_nts.clone(),
        }
    }
}

/// Build template DFAs for all terminals in the parser.
/// 
/// Each terminal gets its own DFA that encodes how it interacts with the parse stack.
/// These are later converted to DWAs and composed into the final Parser DWA.
pub fn build_template_dfas(parser: &GLRParser) -> Result<BTreeMap<TerminalID, DFA>, FullDWABuildError> {
    use rayon::prelude::*;
    
    let all = compute_all_characterizations(parser);
    crate::debug!(5, "Computed terminal characterizations for {} terminals", all.len());

    // OPTIMIZATION: Group terminals by their characterization key (excluding terminal ID).
    // Terminals with identical grammatical behavior can share the same DFA.
    let mut key_to_terms: std::collections::HashMap<CharacterizationKey, Vec<(TerminalID, TerminalCharacterization)>> = 
        std::collections::HashMap::new();
    
    for (term, tc) in all {
        let key = CharacterizationKey::from_characterization(&tc);
        key_to_terms.entry(key).or_default().push((term, tc));
    }
    
    let unique_chars = key_to_terms.len();
    crate::debug!(5, "Found {} unique characterizations (sharing DFAs for {} groups)", 
        unique_chars, key_to_terms.values().filter(|v| v.len() > 1).count());

    // Build NFAs in parallel (pure computation, no shared state)
    let key_term_list: Vec<_> = key_to_terms.into_iter().collect();
    let nfas_and_terms: Vec<_> = key_term_list
        .par_iter()
        .map(|(_key, terms)| {
            let (first_term, first_tc) = &terms[0];
            let nfa = build_nfa_from_terminal_characterization(first_tc).unwrap();
            (*first_term, terms.clone(), nfa)
        })
        .collect();
    
    // Determinize and minimize in parallel (thread-local weight caches eliminate contention)
    let results: Vec<_> = nfas_and_terms
        .into_par_iter()
        .map(|(first_term, terms, nfa)| {
            let dfa = nfa.determinize_and_minimize();
            (first_term, terms, dfa)
        })
        .collect();
    
    let mut result = BTreeMap::new();
    for (first_term, terms, dfa) in results {
        crate::debug!(6, "Terminal {:?}: {} states after minimize", first_term, dfa.states.len());
        
        // Debug stats at level 6
        if crate::r#macro::is_debug_level_enabled(6) {
            for (term, tc) in &terms {
                let num_rc_nonempty = tc.reduce_characterizations.values().filter(|r| !r.reveal_and_rereduces.is_empty() || !r.reveal_goto_shift_escapes.is_empty()).count();
                crate::debug!(6, "Terminal {:?}: {} shifts, {} reduces, {} non-trivial reduce chars, {} DFA states, {} transitions", 
                    term, 
                    tc.initial_shifts.len(), 
                    tc.initial_reduces.len(), 
                    num_rc_nonempty, 
                    dfa.states.len(),
                    dfa.states.num_transitions()
                );
                crate::debug!(7, "{}", tc);
            }
        }
        
        // Clone the DFA for all terminals with this characterization
        for (term, _) in terms {
            result.insert(term, dfa.clone());
        }
    }

    Ok(result)
}

/// Build template DWAs for all terminals in the parser.
/// 
/// Each terminal gets its own DWA that encodes how it interacts with the parse stack.
/// These are later composed into the final Parser DWA.
/// 
/// This function builds unweighted DFAs first (via build_template_dfas), then
/// converts them to DWAs with Weight::all() on all transitions and finals.
pub fn build_template_dwas(parser: &GLRParser) -> Result<BTreeMap<TerminalID, DWA>, FullDWABuildError> {
    let dfas = build_template_dfas(parser)?;
    
    // Convert DFAs to DWAs
    let dwas: BTreeMap<TerminalID, DWA> = dfas
        .into_iter()
        .map(|(term_id, dfa)| (term_id, dfa.to_dwa()))
        .collect();
    
    Ok(dwas)
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
