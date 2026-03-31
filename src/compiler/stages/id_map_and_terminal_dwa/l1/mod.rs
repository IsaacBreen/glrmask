//! L1 terminal DWA: fast direct construction for terminals with max path length ≤ 1.
//!
//! Since L1 terminals never co-occur with another terminal in a single token,
//! the DWA can be built by walking each token from each state and checking
//! which terminal matches at the end. No full NWA trie-walk pipeline needed.

use std::collections::HashMap;
use std::sync::Arc;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::minimize_fast;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::ds::weight::Weight;
use crate::Vocab;

use super::types::{TerminalColoring, debug_profile_enabled};

/// Build an L1 id_map and terminal DWA for the given vocab and terminal set.
///
/// 1. Build id_map via `InternalIdMap::build_l1` (fast fingerprint-based equiv).
/// 2. Build L1 terminal DWA via the direct walk path (no trie-walk NWA).
///
/// Returns `None` if the vocab is empty or no terminal matches exist.
pub(crate) fn build_l1_id_map_and_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    _terminal_coloring: &TerminalColoring,
    _use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
) -> Option<(InternalIdMap, DWA)> {
    if vocab.is_empty() {
        return None;
    }

    // 1. Build L1 id_map (fast fingerprint-based equivalence, no DFA walk).
    let id_map = InternalIdMap::build_l1(tokenizer, vocab);

    // 2. Build L1 terminal DWA via direct walk.
    let num_terminals = grammar.num_terminals as u32;
    let dwa = build_l1_terminal_dwa(
        tokenizer,
        vocab,
        &id_map,
        ignore_terminal,
        num_terminals,
        Some(active_terminals),
    )?;

    Some((id_map, dwa))
}

/// Build an L1 terminal DWA directly — returns just the DWA.
///
/// Walks all (token, representative_state) pairs, builds a 2-state NWA,
/// then determinizes + minimizes.
fn build_l1_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
    num_terminals: u32,
    active_terminals: Option<&[bool]>,
) -> Option<DWA> {
    let partition_total_start = std::time::Instant::now();

    let internal_vocab: Vec<(u32, Vec<u8>)> = id_map
        .vocab_tokens
        .iter_representative_ids()
        .enumerate()
        .filter_map(|(internal_token_id, representative)| {
            vocab
                .entries
                .get(&representative)
                .map(|bytes| (internal_token_id as u32, bytes.clone()))
        })
        .collect();

    if internal_vocab.is_empty() {
        return None;
    }

    let num_tsids = id_map.num_tsids();
    let num_term = num_terminals as usize;

    // Pre-populate flat DFA transition table for fast byte-by-byte walking.
    let dfa = tokenizer.dfa.states();
    let num_dfa_states = dfa.len();
    let flat_dfa: Vec<[u32; 256]> = (0..num_dfa_states)
        .map(|s| {
            let mut flat = [u32::MAX; 256];
            for (b, &target) in dfa[s].transitions.iter() {
                flat[b as usize] = target;
            }
            flat
        })
        .collect();

    // Build map: representative_state → list of internal TSIDs with that representative.
    let mut rep_to_tsids: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (tsid_idx, representative_state) in
        id_map.tokenizer_states.iter_representative_ids().enumerate()
    {
        rep_to_tsids
            .entry(representative_state)
            .or_default()
            .push(tsid_idx as u32);
    }
    let unique_reps: Vec<u32> = rep_to_tsids.keys().copied().collect();

    // Phase 1: Walk only unique (token, representative_state) pairs.
    let mut tokens_direct_by_rep: Vec<FxHashMap<u32, RangeSetBlaze<u32>>> = vec![FxHashMap::default(); num_term];
    let mut future_groups_by_rep: HashMap<(u32, u32), Vec<u32>> = HashMap::new();

    let walk_start = std::time::Instant::now();
    let mut total_pairs: u64 = 0;
    let mut total_alive: u64 = 0;

    for &(internal_token_id, ref bytes) in &internal_vocab {
        for &rep_state in &unique_reps {
            total_pairs += 1;

            let mut scan_state = rep_state;
            let mut alive = true;
            for &byte in <[u8]>::iter(bytes) {
                let next = flat_dfa[scan_state as usize][byte as usize];
                if next == u32::MAX {
                    alive = false;
                    break;
                }
                scan_state = next;
            }
            if !alive {
                continue;
            }
            total_alive += 1;

            for t in tokenizer.dfa.finalizers(scan_state).iter() {
                let terminal = t as TerminalID;
                tokens_direct_by_rep[terminal as usize]
                    .entry(rep_state)
                    .or_default()
                    .insert(internal_token_id);
            }

            future_groups_by_rep
                .entry((rep_state, scan_state))
                .or_default()
                .push(internal_token_id);
        }
    }
    let phase1_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    // Phase 2: Process future groups, collecting per (terminal, rep_state).
    let phase2_start = std::time::Instant::now();
    let mut future_cache: FxHashMap<u32, Vec<TerminalID>> = FxHashMap::default();
    let num_groups = future_groups_by_rep.len();

    let mut tokens_future_by_rep: Vec<FxHashMap<u32, RangeSetBlaze<u32>>> = vec![FxHashMap::default(); num_term];

    for ((rep_state, ending_state), token_ids) in &future_groups_by_rep {
        let futures = future_cache
            .entry(*ending_state)
            .or_insert_with(|| {
                tokenizer
                    .possible_future_terminals_iter(*ending_state)
                    .collect()
            });
        let group_set: RangeSetBlaze<u32> =
            RangeSetBlaze::from_iter(token_ids.iter().copied().map(|id| id..=id));
        for &future_terminal in futures.iter() {
            let entry = tokens_future_by_rep[future_terminal as usize]
                .entry(*rep_state)
                .or_default();
            *entry |= &group_set;
        }
    }
    let phase2_ms = phase2_start.elapsed().as_secs_f64() * 1000.0;

    // Build the 2-state NWA directly.
    let nwa_start = std::time::Instant::now();
    let mut nwa = NWA::new(num_tsids, id_map.max_internal_token_id());
    let start = nwa.add_state();
    let accept = nwa.add_state();
    nwa.start_states = vec![start];
    nwa.set_final_weight(accept, Weight::all());

    let rep_ids: Vec<u32> = id_map.tokenizer_states.iter_representative_ids().collect();

    let mut num_transitions = 0u32;
    for terminal_id in 0..num_term {
        if let Some(active) = active_terminals {
            if !active.get(terminal_id).copied().unwrap_or(false) {
                continue;
            }
        }
        let is_ignored = ignore_terminal == Some(terminal_id as TerminalID);
        let direct_map = &mut tokens_direct_by_rep[terminal_id];
        let future_map = &mut tokens_future_by_rep[terminal_id];

        let build_weight = |direct: &mut FxHashMap<u32, RangeSetBlaze<u32>>,
                            future: &mut FxHashMap<u32, RangeSetBlaze<u32>>,
                            include_future: bool|
         -> Option<Weight> {
            let mut rep_token_sets: FxHashMap<u32, Arc<RangeSetBlaze<u32>>> = FxHashMap::default();
            for (&rep_state, direct_set) in direct.iter() {
                let mut token_set = direct_set.clone();
                if include_future {
                    if let Some(future_set) = future.get(&rep_state) {
                        token_set |= future_set.clone();
                    }
                }
                if !token_set.is_empty() {
                    rep_token_sets.insert(rep_state, Arc::new(token_set));
                }
            }
            if include_future {
                for (&rep_state, future_set) in future.iter() {
                    if rep_token_sets.contains_key(&rep_state) {
                        continue;
                    }
                    if !future_set.is_empty() {
                        rep_token_sets.insert(rep_state, Arc::new(future_set.clone()));
                    }
                }
            }
            if rep_token_sets.is_empty() {
                return None;
            }

            let weight = Weight::from_per_tsid_shared(
                rep_ids.iter().enumerate().filter_map(|(tsid, &rep_state)| {
                    rep_token_sets
                        .get(&rep_state)
                        .map(|ts| (tsid as u32, Arc::clone(ts)))
                }),
            );
            if weight.is_empty() {
                None
            } else {
                Some(weight)
            }
        };

        if is_ignored {
            let mut empty_future: FxHashMap<u32, RangeSetBlaze<u32>> = FxHashMap::default();
            if let Some(weight) = build_weight(direct_map, &mut empty_future, false) {
                nwa.add_transition(start, terminal_id as i32, accept, weight);
                num_transitions += 1;
            }
            let mut empty_direct: FxHashMap<u32, RangeSetBlaze<u32>> = FxHashMap::default();
            if let Some(weight) = build_weight(&mut empty_direct, future_map, true) {
                nwa.add_epsilon(start, accept, weight);
            }
        } else {
            if let Some(weight) = build_weight(direct_map, future_map, true) {
                nwa.add_transition(start, terminal_id as i32, accept, weight);
                num_transitions += 1;
            }
        }
    }

    // Determinize + minimize.
    let det_start = std::time::Instant::now();
    let dwa_result = determinize(&nwa).expect("L1 direct NWA determinize failed");
    let det_ms = det_start.elapsed().as_secs_f64() * 1000.0;
    let det_states = dwa_result.num_states();
    let min_start_t = std::time::Instant::now();
    let dwa_min = minimize_fast(&dwa_result);
    let min_ms = min_start_t.elapsed().as_secs_f64() * 1000.0;
    let min_states = dwa_min.num_states();

    let nwa_ms = nwa_start.elapsed().as_secs_f64() * 1000.0;
    let partition_total_ms = partition_total_start.elapsed().as_secs_f64() * 1000.0;

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][terminal_dwa] partition_build_l1_direct vocab={} tsids={} pairs={} alive={} groups={} transitions={} phase1_ms={:.1} phase2_ms={:.1} nwa_ms={:.1} det_states={} det_ms={:.1} min_states={} min_ms={:.1} total_ms={:.1}",
            internal_vocab.len(), num_tsids, total_pairs, total_alive,
            num_groups, num_transitions, phase1_ms, phase2_ms, nwa_ms, det_states, det_ms,
            min_states, min_ms, partition_total_ms,
        );
    }

    Some(dwa_min)
}
