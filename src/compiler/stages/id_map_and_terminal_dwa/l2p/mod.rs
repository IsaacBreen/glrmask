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

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::minimize_configured;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::possible_matches::{
    PossibleMatchesComputer, collect_possible_matches_by_internal_tsid,
};
use crate::compiler::stages::id_map_and_terminal_dwa::merge::{
    LocalIdMapTerminalDwa, identity_original_to_local_state,
};
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::Weight;
use crate::Vocab;

use super::grammar_helpers::compute_always_allowed_follows;
use super::types::{TerminalColoring, TerminalDwaPhaseProfile, compile_profile_enabled, debug_profile_enabled};
use nwa_builder::{build_nwa_via_trie_walk, internal_vocab_entries, seed_root_nodes};
use postprocess::{
    apply_disallowed_follow_constraints, canonicalize_acyclic_nwa, collapse_always_allowed,
    prune_non_coreachable_states,
};

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

    let total_started_at = Instant::now();
    let num_original_states = tokenizer.num_states() as usize;

    // No DFA clone needed — active_terminals filtering is applied lazily
    // in TokenizerView::new_filtered (for equiv analysis) and at point-of-use
    // in the NWA builder, seed_root_nodes, and PossibleMatchesComputer.
    let simplify_started_at = Instant::now();
    let orig_to_simplified = identity_original_to_local_state(num_original_states);
    let simplify_ms = simplify_started_at.elapsed().as_secs_f64() * 1000.0;

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l2p_simplify] partition={} mode=lazy_filter original_states={} simplified_states={}",
            partition_label, num_original_states, tokenizer.num_states(),
        );
    }

    // ---- Step 1: Equivalence analysis (with active group filtering) ----
    let id_map_started_at = Instant::now();
    let simplified_id_map = equivalence_analysis::combined::analyze_equivalences_with_group_filter(
        tokenizer,
        vocab,
        disallowed_follows,
        ignore_terminal,
        Some(active_terminals),
        shared_vocab_dfa_cache,
        flat_trans,
    );
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    // ---- Step 2-3: Internal vocab + prefix tree ----
    let vocab_tree_started_at = Instant::now();
    let internal_vocab = internal_vocab_entries(vocab, &simplified_id_map);
    if internal_vocab.is_empty() {
        return None;
    }
    let full_tree = VocabPrefixTree::build_owned(
        internal_vocab
            .iter()
            .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
            .collect(),
    );
    let vocab_tree_ms = vocab_tree_started_at.elapsed().as_secs_f64() * 1000.0;

    // ---- Step 4: Possible matches ----
    let possible_matches_started_at = Instant::now();
    let mut pm_computer = PossibleMatchesComputer::new_filtered(tokenizer, active_terminals);
    let possible_matches_by_state = collect_possible_matches_by_internal_tsid(
        tokenizer,
        &full_tree.root,
        &mut pm_computer,
        &simplified_id_map.tokenizer_states,
    );
    let possible_matches_ms = possible_matches_started_at.elapsed().as_secs_f64() * 1000.0;

    // ---- Step 5: Create NWA and seed root nodes ----
    let seed_started_at = Instant::now();
    let mut nwa = NWA::new(simplified_id_map.num_tsids(), simplified_id_map.max_internal_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);

    let roots = seed_root_nodes(
        &mut nwa,
        start_state,
        tokenizer,
        &simplified_id_map,
        terminal_coloring,
        ignore_terminal,
        &possible_matches_by_state,
        Some(active_terminals),
    );
    let seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;

    // ---- Step 6: Trie-walk NWA build ----
    // active_terminals filtering is applied at point-of-use in the NWA builder.
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
        Some(active_terminals),
    );
    let trie_build_ms = trie_build_started_at.elapsed().as_secs_f64() * 1000.0;

    // ---- Step 7: Postprocess ----
    let always_allowed_started_at = Instant::now();
    let always_allowed = compute_always_allowed_follows(grammar);
    let always_allowed_ms = always_allowed_started_at.elapsed().as_secs_f64() * 1000.0;
    let nwa_states_after_build = nwa.states.len();

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
    let nwa_states_after_collapse = nwa.states.len();

    let disallowed_started_at = Instant::now();
    apply_disallowed_follow_constraints(&mut nwa, disallowed_follows, grammar.num_terminals as usize);
    let disallowed_ms = disallowed_started_at.elapsed().as_secs_f64() * 1000.0;
    let nwa_states_after_disallowed = nwa.states.len();

    let prune_started_at = Instant::now();
    prune_non_coreachable_states(&mut nwa);
    let prune_ms = prune_started_at.elapsed().as_secs_f64() * 1000.0;
    let nwa_states_after_prune = nwa.states.len();

    let canonicalize_started_at = Instant::now();
    canonicalize_acyclic_nwa(&mut nwa);
    let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;
    let nwa_states_after_canonicalize = nwa.states.len();
    let postprocess_ms = always_allowed_ms + collapse_ms + disallowed_ms + prune_ms + canonicalize_ms;

    // ---- Step 8: Determinize → minimize ----
    let determinize_started_at = Instant::now();
    let det = determinize(&nwa).expect("L2+ terminal NWA determinization failed");
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;

    let minimize_started_at = Instant::now();
    let dwa = minimize_configured(&det, 50);
    let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;
    let dwa_stats_before_compact = dwa.stats();
    let dwa_stats_after_compact = dwa.stats();

    if compile_profile_enabled() || debug_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p] partition={} vocab_tokens={} tsids={} simplify_ms={:.3} simplified_states={} id_map_ms={:.3} vocab_tree_ms={:.3} possible_matches_ms={:.3} seed_ms={:.3} terminal_nwa_build_ms={:.3} nwa_states={}->{}->{}->{}->{} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} prune_ms={:.3} canonicalize_ms={:.3} postprocess_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} minimize_states={} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            simplified_id_map.num_tsids(),
            simplify_ms,
            tokenizer.num_states(),
            id_map_ms,
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
        profile: TerminalDwaPhaseProfile {
            id_map_ms: simplify_ms + id_map_ms,
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
