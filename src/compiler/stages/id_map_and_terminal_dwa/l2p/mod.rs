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
pub(crate) mod specialized_dfa;

use std::collections::BTreeMap;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize_with_threshold;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::possible_matches::{
    PossibleMatchesComputer, collect_possible_matches_by_internal_tsid,
};
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::Weight;
use crate::Vocab;

use super::grammar_helpers::compute_always_allowed_follows;
use super::types::{TerminalColoring, compile_profile_enabled, debug_profile_enabled};
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
) -> Option<(InternalIdMap, DWA)> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();

    // ---- Step 0: Build quotient tokenizer and decide whether to use it ----
    // Skip quotient for partitions with many active terminals — the quotient
    // cost outweighs the benefit when there's little state reduction.
    let active_count = active_terminals.iter().filter(|&&a| a).count();
    let spec_started_at = Instant::now();
    let spec = if active_count <= active_terminals.len() / 4 {
        Some(specialized_dfa::build_specialized_tokenizer(tokenizer, active_terminals))
    } else {
        None
    };
    let spec_ms = spec_started_at.elapsed().as_secs_f64() * 1000.0;

    // Use the quotient if it reduces state count by at least 50%.
    let use_quotient = spec.as_ref().map_or(false, |s| {
        s.tokenizer.num_states() < tokenizer.num_states() / 2
    });

    let (effective_tok, original_to_quotient) = if use_quotient {
        let s = spec.as_ref().unwrap();
        (&s.tokenizer, Some(&s.original_to_quotient))
    } else {
        (tokenizer, None)
    };

    if debug_profile_enabled() {
        let q_states = spec.as_ref().map(|s| s.tokenizer.num_states()).unwrap_or(0);
        eprintln!(
            "[glrmask/debug][l2p] specialized_dfa: {} original_states -> {} quotient_states in {:.3}ms ({}) active_terminals={}",
            tokenizer.num_states(),
            q_states,
            spec_ms,
            if use_quotient { "USING quotient" } else if spec.is_some() { "SKIPPED (insufficient reduction)" } else { "SKIPPED (too many active)" },
            active_count,
        );
    }

    // ---- Step 1: Equivalence analysis ----
    let id_map_started_at = Instant::now();
    let eff_id_map = equivalence_analysis::combined::analyze_equivalences(
        effective_tok,
        vocab,
        disallowed_follows,
        ignore_terminal,
    );
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    // ---- Step 1b: Compose id_map for original DFA states (only if using quotient) ----
    let compose_started_at = Instant::now();
    let composed_id_map = if let Some(o2q) = original_to_quotient {
        compose_id_map(&eff_id_map, o2q, tokenizer.num_states() as usize)
    } else {
        eff_id_map.clone()
    };
    let compose_ms = compose_started_at.elapsed().as_secs_f64() * 1000.0;

    // ---- Step 2-3: Internal vocab + prefix tree ----
    let vocab_tree_started_at = Instant::now();
    let internal_vocab = internal_vocab_entries(vocab, &eff_id_map);
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
    let mut pm_computer = PossibleMatchesComputer::new(effective_tok);
    let possible_matches_by_state = collect_possible_matches_by_internal_tsid(
        effective_tok,
        &full_tree.root,
        &mut pm_computer,
        &eff_id_map.tokenizer_states,
    );
    let possible_matches_ms = possible_matches_started_at.elapsed().as_secs_f64() * 1000.0;

    // ---- Step 5: Create NWA and seed root nodes ----
    let seed_started_at = Instant::now();
    let mut nwa = NWA::new(eff_id_map.num_tsids(), eff_id_map.max_internal_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);

    let roots = seed_root_nodes(
        &mut nwa,
        start_state,
        effective_tok,
        &eff_id_map,
        terminal_coloring,
        ignore_terminal,
        &possible_matches_by_state,
    );
    let seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;

    // ---- Step 6: Trie-walk NWA build ----
    let trie_build_started_at = Instant::now();
    let _build_profile = build_nwa_via_trie_walk(
        effective_tok,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        &mut nwa,
        leaf_state,
        eff_id_map.num_tsids(),
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
    let dwa = minimize_with_threshold(&det, 50);
    let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;

    if compile_profile_enabled() || debug_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p] partition={} vocab_tokens={} tsids={} specialized_dfa_ms={:.3} compose_ms={:.3} id_map_ms={:.3} vocab_tree_ms={:.3} possible_matches_ms={:.3} seed_ms={:.3} terminal_nwa_build_ms={:.3} nwa_states={}->{}->{}->{}->{} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} prune_ms={:.3} canonicalize_ms={:.3} postprocess_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} minimize_states={} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            eff_id_map.num_tsids(),
            spec_ms,
            compose_ms,
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
            dwa.num_states(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some((composed_id_map, dwa))
}

/// Compose the spec_id_map (quotient → TSID) with the original_to_quotient
/// mapping (original DFA state → quotient state) to produce the final id_map
/// (original DFA state → TSID).
fn compose_id_map(
    spec_id_map: &InternalIdMap,
    original_to_quotient: &[u32],
    num_original_states: usize,
) -> InternalIdMap {
    // Build original_state → TSID by composing through quotient
    let original_to_tsid: Vec<u32> = (0..num_original_states)
        .map(|s| {
            let q = original_to_quotient[s] as usize;
            spec_id_map.tokenizer_states.original_to_internal[q]
        })
        .collect();

    let num_tsids = spec_id_map.tokenizer_states.num_internal_ids();

    // Build representative_original_ids: for each TSID, pick the smallest
    // original state that maps to it
    let mut representative_original_ids = vec![u32::MAX; num_tsids as usize];
    for (orig, &tsid) in original_to_tsid.iter().enumerate() {
        if (tsid as usize) < representative_original_ids.len()
            && representative_original_ids[tsid as usize] == u32::MAX
        {
            representative_original_ids[tsid as usize] = orig as u32;
        }
    }

    let tokenizer_states = ManyToOneIdMap::from_original_to_internal_with_representatives(
        original_to_tsid,
        num_tsids,
        representative_original_ids,
    );

    InternalIdMap {
        tokenizer_states,
        vocab_tokens: spec_id_map.vocab_tokens.clone(),
    }
}
