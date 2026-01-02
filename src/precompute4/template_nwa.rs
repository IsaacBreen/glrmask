use std::collections::{BTreeMap, BTreeSet};


use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{compute_all_characterizations, TerminalCharacterization};
use crate::precompute4::utils;
use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::precompute4::weighted_automata::{DWA, NWA, NWABuildError, StateID, Weight};

/// Error type for building the Parser DWA structures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullDWABuildError {
    ParserStateIdOutOfRange { state_id: ParserStateID },
    AutomatonBuild(NWABuildError),
}

impl From<NWABuildError> for FullDWABuildError {
    fn from(e: NWABuildError) -> Self { FullDWABuildError::AutomatonBuild(e) }
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

    // OPTIMIZATION: Compute which NTs can actually reach a final state via reveal_goto_shift_escapes.
    // If an NT has no escape edges and only chains to other NTs without escapes, it's dead code.
    let reachable_nts = compute_reachable_nts(tc);
    
    // Only create nodes for reachable NTs (those that can eventually lead to a shift)
    let mut nt_nodes: BTreeMap<NonTerminalID, StateID> = BTreeMap::new();
    for &nt in &reachable_nts {
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
    // OPTIMIZATION: Group by (len, nt) to share the default-transition chain.
    // Before: O(n) paths where n = number of (state, len, nt) tuples
    // After: O(groups) chains where groups = unique (len, nt) pairs
    let mut reduces_by_signature: BTreeMap<(usize, NonTerminalID), Vec<ParserStateID>> = BTreeMap::new();

    for &(initial_state, len, nt) in &tc.initial_reduces {
        reduces_by_signature.entry((len, nt)).or_default().push(initial_state);
    }
    
    for ((len, nt), initial_states) in reduces_by_signature {
        // Skip reduces targeting unreachable NTs (NTs with no path to escape edges)
        let Some(&target_nt_state) = nt_nodes.get(&nt) else { continue };

        
        // Create shared chain: collector_state --(default)*len--> target_nt_state
        // First, create the collector state that all initial states will feed into
        let collector_state = if len == 0 { target_nt_state } else { nwa.states.add_state() };
        
        // Build the default transition chain from collector to target
        let mut from = collector_state;
        for i in 0..len {
            let to = if i == len - 1 { target_nt_state } else { nwa.states.add_state() };
            nwa.states.add_transition(from, DEFAULT_TRANSITION_SYMBOL, to, w_all.clone())?;
            from = to;
        }
        
        // Connect each initial_state to the shared collector
        for initial_state in initial_states {
            let pos_initial = utils::encode_symbol_i16(initial_state)?;
            let s0 = nwa.states.add_state();
            nwa.add_epsilon(start, s0, w_all.clone());
            nwa.add_transition(s0, pos_initial, collector_state, w_all.clone())?;
        }
    }


    // Actions from non-terminal states (only for reachable NTs).
    for (nt, rc) in &tc.reduce_characterizations {
        // Skip NTs that can't reach any escape edges
        let Some(&src_nt_state) = nt_nodes.get(nt) else { continue };


        for &(revealed_state, len, reduce_nt) in &rc.reveal_and_rereduces {
            // Skip edges to unreachable NTs
            let Some(&dst_nt_state) = nt_nodes.get(&reduce_nt) else { continue };
            let pos_revealed = utils::encode_symbol_i16(revealed_state)?;


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

/// Compute which NTs can actually reach a final state (shift escape).
/// 
/// Works backwards: starts with NTs that have reveal_goto_shift_escapes,
/// then propagates to NTs whose reveal_and_rereduces can reach those.
/// 
/// Returns empty set if no NTs have escape edges (all reduce characterizations are dead ends).
fn compute_reachable_nts(tc: &TerminalCharacterization) -> BTreeSet<NonTerminalID> {
    // Step 1: Find all NTs with escape edges (these can directly reach final states)
    let mut reachable: BTreeSet<NonTerminalID> = tc.reduce_characterizations
        .iter()
        .filter(|(_, rc)| !rc.reveal_goto_shift_escapes.is_empty())
        .map(|(nt, _)| *nt)
        .collect();
    
    if reachable.is_empty() {
        // No escape edges anywhere - the entire reduce characterization machinery is dead code
        crate::debug!(5, "  No escape edges found - skipping all {} reduce characterization NT nodes", tc.reduce_characterizations.len());
        return reachable;
    }
    
    // Step 2: Build reverse adjacency: target_nt -> set of source NTs that can reach it
    let mut reverse_adj: BTreeMap<NonTerminalID, BTreeSet<NonTerminalID>> = BTreeMap::new();
    for (src_nt, rc) in &tc.reduce_characterizations {
        for &(_revealed, _len, target_nt) in &rc.reveal_and_rereduces {
            reverse_adj.entry(target_nt).or_default().insert(*src_nt);
        }
    }
    
    // Step 3: Propagate backwards - any NT that can reach a reachable NT is also reachable
    let mut worklist: Vec<NonTerminalID> = reachable.iter().cloned().collect();
    while let Some(nt) = worklist.pop() {
        if let Some(sources) = reverse_adj.get(&nt) {
            for &src in sources {
                if reachable.insert(src) {
                    worklist.push(src);
                }
            }
        }
    }
    
    crate::debug!(5, "  Reachable NTs: {} of {} total", reachable.len(), tc.all_nts.len());
    reachable
}


/// Deprecated alias for build_nwa_from_terminal_characterization
#[deprecated(since = "0.3.0", note = "Use build_nwa_from_terminal_characterization instead")]
pub fn build_template_nwa_from_characterization(tc: &TerminalCharacterization) -> Result<NWA, FullDWABuildError> {
    build_nwa_from_terminal_characterization(tc)
}

/// Build terminal DWAs for all terminals in the parser.
/// 
/// A characterization key that excludes the terminal ID, allowing us to share DWAs
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

/// Each terminal gets its own DWA that encodes how it interacts with the parse stack.
/// These are later composed into the final Parser DWA.
pub fn build_template_dwas(parser: &GLRParser) -> Result<BTreeMap<TerminalID, DWA>, FullDWABuildError> {
    use rayon::prelude::*;
    
    let all = compute_all_characterizations(parser);
    crate::debug!(5, "Computed terminal characterizations for {} terminals", all.len());
    
    // At level 6, print the full parser table
    crate::debug!(6, "{}", parser);


    // OPTIMIZATION: Group terminals by their characterization key (excluding terminal ID).
    // Terminals with identical grammatical behavior can share the same DWA.
    let mut key_to_terms: std::collections::HashMap<CharacterizationKey, Vec<(TerminalID, TerminalCharacterization)>> = 
        std::collections::HashMap::new();
    
    for (term, tc) in all {
        let key = CharacterizationKey::from_characterization(&tc);
        key_to_terms.entry(key).or_default().push((term, tc));
    }
    
    let unique_chars = key_to_terms.len();
    crate::debug!(5, "Found {} unique characterizations (sharing DWAs for {} groups)", 
        unique_chars, key_to_terms.values().filter(|v| v.len() > 1).count());

    // Build NWAs in parallel (pure computation, no shared state)
    let key_term_list: Vec<_> = key_to_terms.into_iter().collect();
    let nwas_and_terms: Vec<_> = key_term_list
        .par_iter()
        .map(|(_key, terms)| {
            let (first_term, first_tc) = &terms[0];
            let nwa = build_nwa_from_terminal_characterization(first_tc).unwrap();
            (*first_term, terms.clone(), nwa)
        })
        .collect();
    
    // Determinize and simplify serially (memory contention in parallel slows things down)
    // Tested parallel 2025: 467-494ms vs serial 380-400ms. Serial wins for many small DWAs.
    let mut result = BTreeMap::new();
    for (first_term, terms, nwa) in nwas_and_terms {
        let mut dwa = nwa.determinize();
        crate::debug!(6, "Terminal {:?} (and {} others): {} states before simplify", 
            first_term, terms.len() - 1, dwa.states.len());
        dwa.simplify_single_pass();
        crate::debug!(6, "Terminal {:?}: {} states after simplify", first_term, dwa.states.len());
        
        // Debug stats at level 5: print one line per terminal with characterization and DFA stats
        if crate::r#macro::is_debug_level_enabled(5) {
            for (term, tc) in &terms {
                let num_rc_nonempty = tc.reduce_characterizations.values().filter(|r| !r.reveal_and_rereduces.is_empty() || !r.reveal_goto_shift_escapes.is_empty()).count();
                
                // Analyze collapse: how many unique (len, nt) pairs vs total initial_reduces?
                let unique_len_nt: std::collections::BTreeSet<_> = tc.initial_reduces.iter()
                    .map(|&(_state, len, nt)| (len, nt))
                    .collect();
                
                // How many unique shift_states?
                let unique_shift_states: std::collections::BTreeSet<_> = tc.initial_shifts.iter()
                    .map(|&(_initial, shift)| shift)
                    .collect();
                
                // Count total reveal_goto_shift_escapes (the terminal escape points)
                let total_rgs: usize = tc.reduce_characterizations.values()
                    .map(|r| r.reveal_goto_shift_escapes.len())
                    .sum();
                
                // How many unique (goto_state, shift_state) pairs in reveal_goto_shift_escapes?
                let unique_goto_shift: std::collections::BTreeSet<_> = tc.reduce_characterizations.values()
                    .flat_map(|r| r.reveal_goto_shift_escapes.iter())
                    .map(|&(_revealed, goto, shift)| (goto, shift))
                    .collect();
                
                crate::debug!(5, "Terminal {:?}: {} shifts, {} reduces, {} non-trivial reduce chars, {} DFA states, {} transitions", 
                    term, 
                    tc.initial_shifts.len(), 
                    tc.initial_reduces.len(), 
                    num_rc_nonempty, 
                    dwa.states.len(),
                    dwa.states.num_transitions()
                );
                crate::debug!(5, "  Collapse analysis: {} unique (len,nt) from {} reduces, {} unique shift_states from {} shifts",
                    unique_len_nt.len(),
                    tc.initial_reduces.len(),
                    unique_shift_states.len(),
                    tc.initial_shifts.len()
                );
                crate::debug!(5, "  Escape analysis: {} total rgs entries, {} unique (goto,shift) pairs",
                    total_rgs,
                    unique_goto_shift.len()
                );
                
                // At level 6, print the full characterization
                crate::debug!(6, "{}", tc);
            }
        }
        
        // Clone the DWA for all terminals with this characterization
        for (term, _) in terms {
            result.insert(term, dwa.clone());
        }
    }

    Ok(result)
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
