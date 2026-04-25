//! L2+ terminal DWA: full NWA-based construction for terminals with path length ≥ 2.
//!
//! Uses the same structure as the pre-partition/path-length code (commit 67146d8):
//! build vocab trie → compute possible_matches → seed root nodes → trie-walk
//! NWA build → postprocess (always_allowed → collapse → disallowed → prune →
//! canonicalize) → determinize → minimize.
//!
//! The only structural difference from the old code is `active_terminals`
//! filtering: terminals not in the L2+ set are skipped during the trie walk.

pub(crate) mod equivalence_analysis;
pub(crate) mod nwa_builder;
pub(crate) mod postprocess;

use std::collections::BTreeMap;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::{minimize_from_env, minimize_with_threshold};
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::grammar::flat::TerminalID;
use crate::compiler::possible_matches::{
    PossibleMatchesComputer,
};
use crate::compiler::stages::id_map_and_terminal_dwa::merge::{DroppedOriginalStateTsidFallback, LocalIdMapTerminalDwa};
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::Weight;
use crate::Vocab;

use super::grammar_helpers::compute_always_allowed_follows;
use super::types::{TerminalColoring, TerminalDwaPhaseProfile, compile_profile_enabled, debug_profile_enabled};
use equivalence_analysis::compat::TokenizerView;
use equivalence_analysis::disallowed_follows::normalize_disallowed_follows;
use equivalence_analysis::state::fast as fast_state_equivalence;
use nwa_builder::{build_nwa_via_trie_walk, internal_vocab_entries, seed_root_nodes};
use postprocess::{
    apply_disallowed_follow_constraints, canonicalize_acyclic_nwa, collapse_always_allowed,
    prune_non_coreachable_states,
};

fn build_partition_pruned_tokenizer(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    relevant_bytes: &[bool; 256],
) -> Tokenizer {
    tokenizer.clone_filtered_for_terminals(active_terminals, relevant_bytes)
}

fn build_dropped_original_state_tsid_fallback(
    original_tokenizer: &Tokenizer,
    simplified_tokenizer: &Tokenizer,
    simplified_id_map: &crate::compiler::stages::equiv_types::InternalIdMap,
    original_to_local_state: &[u32],
    vocab: &Vocab,
    active_terminals: &[bool],
    relevant_bytes: &[bool; 256],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> DroppedOriginalStateTsidFallback {
    let mut original_to_local_tsid = vec![u32::MAX; original_to_local_state.len()];
    let mut simplified_state_to_original = vec![u32::MAX; simplified_tokenizer.num_states() as usize];

    for (original_state, &local_state) in original_to_local_state.iter().enumerate() {
        if local_state == u32::MAX {
            continue;
        }
        original_to_local_tsid[original_state] =
            simplified_id_map.tokenizer_states.original_to_internal[local_state as usize];
        if simplified_state_to_original[local_state as usize] == u32::MAX {
            simplified_state_to_original[local_state as usize] = original_state as u32;
        }
    }

    let representative_local_tsids_and_states: Vec<(u32, usize)> = simplified_id_map
        .tokenizer_states
        .iter_representative_ids()
        .enumerate()
        .filter_map(|local_state| {
            let (local_tsid, local_state) = local_state;
            let original_state = simplified_state_to_original[local_state as usize];
            (original_state != u32::MAX).then_some((local_tsid as u32, original_state as usize))
        })
        .collect();
    if representative_local_tsids_and_states.is_empty() {
        return DroppedOriginalStateTsidFallback::new(original_to_local_tsid);
    }

    // Fast path: missing states that can never produce an active terminal are
    // equivalent to an already-mapped dead representative state.
    let active_bitset = bitset_from_active_terminals(active_terminals, original_tokenizer.num_terminals as usize);
    let mut dead_representative_tsid = None;
    for &(local_tsid, original_state) in &representative_local_tsids_and_states {
        let has_active_final = original_tokenizer
            .matched_terminals_iter(original_state as u32)
            .any(|tid| active_terminals.get(tid as usize).copied().unwrap_or(false));
        let has_active_future = !original_tokenizer
            .possible_future_terminals(original_state as u32)
            .is_disjoint(&active_bitset);
        if !has_active_final && !has_active_future {
            dead_representative_tsid = Some(local_tsid);
            break;
        }
    }

    let mut unresolved_missing_states = Vec::new();
    if let Some(dead_tsid) = dead_representative_tsid {
        for (state, tsid) in original_to_local_tsid.iter_mut().enumerate() {
            if *tsid != u32::MAX {
                continue;
            }
            let has_active_final = original_tokenizer
                .matched_terminals_iter(state as u32)
                .any(|tid| active_terminals.get(tid as usize).copied().unwrap_or(false));
            let has_active_future = !original_tokenizer
                .possible_future_terminals(state as u32)
                .is_disjoint(&active_bitset);
            if !has_active_final && !has_active_future {
                *tsid = dead_tsid;
            } else {
                unresolved_missing_states.push(state);
            }
        }
    } else {
        unresolved_missing_states = original_to_local_tsid
            .iter()
            .enumerate()
            .filter_map(|(state, &tsid)| (tsid == u32::MAX).then_some(state))
            .collect();
    }

    if unresolved_missing_states.is_empty() {
        return DroppedOriginalStateTsidFallback::new(original_to_local_tsid);
    }

    let representative_tokens: Vec<&[u8]> = simplified_id_map
        .vocab_tokens
        .iter_representative_ids()
        .filter_map(|token_id| vocab.entries.get(&token_id).map(|bytes| bytes.as_slice()))
        .collect();
    if representative_tokens.is_empty() {
        return DroppedOriginalStateTsidFallback::new(original_to_local_tsid);
    }

    let mut states: Vec<usize> = representative_local_tsids_and_states
        .iter()
        .map(|(_, original_state)| *original_state)
        .collect();
    states.extend(unresolved_missing_states.iter().copied());

    let pruned_tokenizer = build_partition_pruned_tokenizer(original_tokenizer, active_terminals, relevant_bytes);
    let tokenizer_view = TokenizerView::new(&pruned_tokenizer);
    let normalized_disallowed = normalize_disallowed_follows(pruned_tokenizer.num_terminals as usize, disallowed_follows);
    let representative_mapping = fast_state_equivalence::find_state_equivalence_classes_with_disallowed(
        &tokenizer_view,
        &representative_tokens,
        &states,
        &normalized_disallowed,
    );

    let mut representative_state_to_tsid = FxHashMap::default();
    for (idx, &(local_tsid, _)) in representative_local_tsids_and_states.iter().enumerate() {
        let mapped_rep = representative_mapping[idx];
        representative_state_to_tsid.entry(mapped_rep).or_insert(local_tsid);
    }

    for (idx, &original_state) in unresolved_missing_states.iter().enumerate() {
        let rep = representative_mapping[representative_local_tsids_and_states.len() + idx];
        if let Some(&tsid) = representative_state_to_tsid.get(&rep) {
            original_to_local_tsid[original_state] = tsid;
        }
    }

    DroppedOriginalStateTsidFallback::new(original_to_local_tsid)
}

fn bitset_from_active_terminals(active_terminals: &[bool], num_terminals: usize) -> BitSet {
    let mut bits = BitSet::new(num_terminals);
    for (terminal_id, &active) in active_terminals.iter().enumerate() {
        if active {
            bits.set(terminal_id);
        }
    }
    bits
}

/// Build an L2+ id_map and terminal DWA for the given vocab and terminal set.
///
/// Builds its own id_map via `InternalIdMap::build_with_group_filter` (full DFA-
/// based equivalence analysis restricted to L2+ terminal groups). Then builds
/// the terminal DWA using the old-shaped trie-walk NWA pipeline matching the
/// 67146d8 code shape:
///
/// 1. Build internal vocab entries
/// 2. Build vocab prefix trie
/// 3. Compute possible_matches (for root node seeding)
/// 4. Create NWA, seed root nodes
/// 5. Trie-walk NWA build
/// 6. Postprocess: always_allowed → collapse → disallowed → prune → canonicalize
/// 7. Determinize → minimize
///
/// `disallowed_follows` is threaded explicitly for id_map building.
///
/// Returns `None` if the vocab is empty.
pub(crate) fn build_l2p_id_map_and_terminal_dwa(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    shared_vocab_dfa_cache: Option<&equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
) -> Option<LocalIdMapTerminalDwa> {
    if vocab.is_empty() {
        return None;
    }

    let original_tokenizer = tokenizer;

    let total_started_at = Instant::now();
    let num_original_states = tokenizer.num_states() as usize;
    let num_active_terminals = active_terminals.iter().filter(|&&active| active).count();
    let mut relevant_bytes = [false; 256];
    for bytes in vocab.entries.values() {
        for &byte in bytes {
            relevant_bytes[byte as usize] = true;
        }
    }

    // ---- Step 0: Simplify tokenizer for active terminals ----
    // Strip non-active terminal bits from finalizers, drop transitions on
    // bytes absent from this partition's vocab, and minimize. This merges
    // states that only differed by non-active terminal info or irrelevant-byte
    // transitions, reducing the state count for equivalence analysis and NWA
    // building.
    let simplify_started_at = Instant::now();
    let (simplified_tok, orig_to_simplified) =
        tokenizer.simplify_for_terminals(active_terminals, Some(&relevant_bytes));
    let simplify_ms = simplify_started_at.elapsed().as_secs_f64() * 1000.0;

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l2p_simplify] partition={} original_states={} simplified_states={}",
            partition_label, num_original_states, simplified_tok.num_states(),
        );
    }

    // DIAGNOSTIC: compare simplified state count against a tokenizer built
    // from scratch using only the active terminals. These should match
    // (they describe the same language); any overshoot indicates a bug in
    // the simplify pipeline.
    if std::env::var_os("GLRMASK_DEBUG_SIMPLIFY_COMPARE_FROM_SCRATCH").is_some() {
        // Re-minimize the simplified DFA. If state count drops, minimize is
        // not reaching a fixed point in the pipeline.
        let dfa_clone = simplified_tok.dfa.clone();
        let t0 = Instant::now();
        let (remin, _) = dfa_clone.minimize_with_state_mapping();
        let remin_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[glrmask/debug][l2p_remin] partition={} simplified_states={} remin_states={} remin_ms={:.1}",
            partition_label,
            simplified_tok.num_states(),
            remin.num_states(),
            remin_ms,
        );
    }

    // From here on, use the simplified tokenizer for all operations.
    let tokenizer = &simplified_tok;

    // ---- Step 1: Equivalence analysis (on simplified tokenizer) ----
    let id_map_started_at = Instant::now();
    let simplified_id_map = equivalence_analysis::combined::analyze_equivalences_with_group_filter(
        tokenizer,
        vocab,
        disallowed_follows,
        ignore_terminal,
        None,
        shared_vocab_dfa_cache,
        flat_trans,
    );
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    // tsid_fallback is independent of the NWA build / postprocess /
    // determinize / minimize pipeline: it only feeds into the final
    // `LocalIdMapTerminalDwa` struct. Run it in parallel with steps 2-8
    // via rayon::join. On the critical partition this cuts ~150-200ms
    // off the per-partition sequential critical path.
    let tokenizer = &simplified_tok;
    let (
        (dropped_original_state_tsid_fallback, tsid_fallback_ms),
        (
            dwa,
            vocab_tree_ms,
            possible_matches_ms,
            seed_ms,
            trie_build_ms,
            always_allowed_ms,
            collapse_ms,
            disallowed_ms,
            prune_ms,
            canonicalize_ms,
            determinize_ms,
            minimize_ms,
            nwa_states_after_build,
            nwa_states_after_collapse,
            nwa_states_after_disallowed,
            nwa_states_after_prune,
            nwa_states_after_canonicalize,
            dwa_stats_before_compact,
            dwa_stats_after_compact,
            early_none,
        ),
    ) = rayon::join(
        || {
            let t0 = Instant::now();
            let fb = build_dropped_original_state_tsid_fallback(
                original_tokenizer,
                &simplified_tok,
                &simplified_id_map,
                &orig_to_simplified,
                vocab,
                active_terminals,
                &relevant_bytes,
                disallowed_follows,
            );
            (fb, t0.elapsed().as_secs_f64() * 1000.0)
        },
        || {
            // ---- Step 2-3: Internal vocab + prefix tree ----
            let vocab_tree_started_at = Instant::now();
            let internal_vocab = internal_vocab_entries(vocab, &simplified_id_map);
            if internal_vocab.is_empty() {
                // Signal early-None via a sentinel. Build a dummy DWA;
                // outer code will observe `early_none=true` and return.
                return (
                    crate::automata::weighted_u32::dwa::DWA::new(0, 0),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0usize, 0usize, 0usize, 0usize, 0usize,
                    crate::automata::weighted_u32::dwa::DWA::new(0, 0).stats(),
                    crate::automata::weighted_u32::dwa::DWA::new(0, 0).stats(),
                    true,
                );
            }
            let full_tree = VocabPrefixTree::build_owned(
                internal_vocab
                    .iter()
                    .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
                    .collect(),
            );
            let vocab_tree_ms = vocab_tree_started_at.elapsed().as_secs_f64() * 1000.0;

            // ---- Step 4: Possible matches (lazy via computer) ----
            let mut pm_computer = PossibleMatchesComputer::new(tokenizer);
            let possible_matches_ms = 0.0;

            // ---- Step 5: Create NWA and seed root nodes ----
            let seed_started_at = Instant::now();
            let mut nwa = NWA::new(simplified_id_map.num_tsids(), simplified_id_map.max_internal_token_id());
            let leaf_state = nwa.add_state();
            nwa.set_final_weight(leaf_state, Weight::all());
            let start_state = nwa.add_state();
            nwa.start_states_mut().push(start_state);

            let roots = seed_root_nodes(
                &mut nwa,
                start_state,
                &simplified_id_map,
            );
            let seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;

            // ---- Step 6: Trie-walk NWA build ----
            let trie_build_started_at = Instant::now();
            let _build_profile = build_nwa_via_trie_walk(
                tokenizer,
                terminal_coloring,
                use_terminal_coloring,
                ignore_terminal,
                &mut nwa,
                leaf_state,
                simplified_id_map.num_tsids(),
                &full_tree.root,
                &roots,
                &mut pm_computer,
                None,
            );
            let trie_build_ms = trie_build_started_at.elapsed().as_secs_f64() * 1000.0;

            // ---- Step 7: Postprocess ----
            let always_allowed_started_at = Instant::now();
            let always_allowed = compute_always_allowed_follows(grammar);
            let always_allowed_ms = always_allowed_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_build = nwa.states().len();

            if debug_profile_enabled() {
                let non_empty_count = always_allowed.iter().filter(|v| !v.is_empty()).count();
                let total_entries: usize = always_allowed.iter().map(|v| v.len()).sum();
                eprintln!(
                    "[glrmask/debug][always_allowed] terminals_with_follows={}/{} total_entries={}",
                    non_empty_count, always_allowed.len(), total_entries,
                );
            }

            let collapse_started_at = Instant::now();
            collapse_always_allowed(&mut nwa, &always_allowed, grammar.num_terminals as usize);
            let collapse_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_collapse = nwa.states().len();

            let disallowed_started_at = Instant::now();
            apply_disallowed_follow_constraints(&mut nwa, disallowed_follows, grammar.num_terminals as usize);
            let disallowed_ms = disallowed_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_disallowed = nwa.states().len();

            let prune_started_at = Instant::now();
            prune_non_coreachable_states(&mut nwa);
            let prune_ms = prune_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_prune = nwa.states().len();

            let canonicalize_started_at = Instant::now();
            canonicalize_acyclic_nwa(&mut nwa);
            let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_canonicalize = nwa.states().len();

            // ---- Step 8: Determinize → minimize ----
            let determinize_started_at = Instant::now();
            let det = determinize(&nwa).expect("L2+ terminal NWA determinization failed");
            let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;

            let minimize_started_at = Instant::now();
            let dwa = minimize_from_env(&det, "GLRMASK_MINIMIZE_L2P", |dwa| {
                minimize_with_threshold(dwa, 50)
            });
            let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;
            let dwa_stats_before_compact = dwa.stats();
            let dwa_stats_after_compact = dwa.stats();

            (
                dwa,
                vocab_tree_ms,
                possible_matches_ms,
                seed_ms,
                trie_build_ms,
                always_allowed_ms,
                collapse_ms,
                disallowed_ms,
                prune_ms,
                canonicalize_ms,
                determinize_ms,
                minimize_ms,
                nwa_states_after_build,
                nwa_states_after_collapse,
                nwa_states_after_disallowed,
                nwa_states_after_prune,
                nwa_states_after_canonicalize,
                dwa_stats_before_compact,
                dwa_stats_after_compact,
                false,
            )
        },
    );
    if early_none {
        return None;
    }
    let postprocess_ms = always_allowed_ms + collapse_ms + disallowed_ms + prune_ms + canonicalize_ms;

    if compile_profile_enabled() || debug_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p] partition={} vocab_tokens={} active_terminals={} original_states={} tsids={} simplify_ms={:.3} simplified_states={} id_map_ms={:.3} tsid_fallback_ms={:.3} vocab_tree_ms={:.3} possible_matches_ms={:.3} seed_ms={:.3} terminal_nwa_build_ms={:.3} nwa_states={}->{}->{}->{}->{} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} prune_ms={:.3} canonicalize_ms={:.3} postprocess_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} minimize_states={} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            num_active_terminals,
            num_original_states,
            simplified_id_map.num_tsids(),
            simplify_ms,
            simplified_tok.num_states(),
            id_map_ms,
            tsid_fallback_ms,
            vocab_tree_ms,
            possible_matches_ms,
            seed_ms,
            trie_build_ms,
            nwa_states_after_build,
            nwa_states_after_collapse,
            nwa_states_after_disallowed,
            nwa_states_after_prune,
            nwa_states_after_canonicalize,
            always_allowed_ms,
            collapse_ms,
            disallowed_ms,
            prune_ms,
            canonicalize_ms,
            postprocess_ms,
            determinize_ms,
            minimize_ms,
            dwa_stats_before_compact.states,
            dwa_stats_before_compact.states,
            dwa_stats_before_compact.transitions,
            dwa_stats_before_compact.transition_pairs,
            dwa_stats_before_compact.interned_ranges,
            dwa_stats_after_compact.interned_ranges,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some(LocalIdMapTerminalDwa {
        id_map: simplified_id_map,
        dwa,
        original_to_local_state: orig_to_simplified,
        dropped_original_state_tsid_fallback: Some(dropped_original_state_tsid_fallback),
        profile: TerminalDwaPhaseProfile {
            id_map_ms: simplify_ms + id_map_ms + tsid_fallback_ms,
            terminal_dwa_ms: vocab_tree_ms
                + possible_matches_ms
                + seed_ms
                + trie_build_ms
                + postprocess_ms
                + determinize_ms
                + minimize_ms,
            compact_ms: 0.0,
        },
    })
}
