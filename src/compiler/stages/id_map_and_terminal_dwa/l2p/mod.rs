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
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::{minimize_from_env, minimize_with_threshold};
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::grammar::flat::TerminalID;
use crate::compiler::possible_matches::{
    PossibleMatchesComputer,
};
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::merge::LocalIdMapTerminalDwa;
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

const L2P_PATH_VALIDATION_ENV: &str = "GLRMASK_VALIDATE_L2P_TERMINAL_DWA_PATH_LENGTHS";
const L2P_PATH_VALIDATION_WALKS: u64 = 128;
const L2P_PATH_VALIDATION_TOKENS_PER_WEIGHT: usize = 8;

fn l2p_path_validation_enabled() -> bool {
    std::env::var_os(L2P_PATH_VALIDATION_ENV).is_some()
}

fn mixed_walk_index(seed: u64, step: usize, state: u32, options: usize) -> usize {
    debug_assert!(options > 0);
    let mut value = seed
        ^ ((step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ ((state as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    value ^= value >> 33;
    value = value.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    value ^= value >> 33;
    value = value.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    value ^= value >> 33;
    (value as usize) % options
}

fn sampled_internal_tokens(weight: &Weight, max_samples: usize) -> Vec<u32> {
    assert!(
        !weight.is_full(),
        "L2P terminal DWA validation expected concrete internal-token weights, got Weight::all()"
    );
    weight.token_union().iter().take(max_samples).collect()
}

fn validate_sampled_path_token_lengths(
    path: &[i32],
    accepted_weight: &Weight,
    id_map: &InternalIdMap,
    vocab: &Vocab,
) {
    for internal_token_id in sampled_internal_tokens(accepted_weight, L2P_PATH_VALIDATION_TOKENS_PER_WEIGHT) {
        let representative = id_map
            .vocab_tokens
            .representative_original_id_for_internal(internal_token_id)
            .unwrap_or_else(|| {
                panic!(
                    "L2P terminal DWA validation could not map internal token {} to a representative original token",
                    internal_token_id,
                )
            });
        let representative_bytes = vocab.entries.get(&representative).unwrap_or_else(|| {
            panic!(
                "L2P terminal DWA validation could not find bytes for representative original token {} (internal token {})",
                representative,
                internal_token_id,
            )
        });
        assert!(
            path.len() <= representative_bytes.len(),
            "L2P terminal DWA validation failed: sampled accepting path {:?} has {} terminals, but representative token {} (internal {}) has only {} bytes",
            path,
            path.len(),
            representative,
            internal_token_id,
            representative_bytes.len(),
        );
    }
}

fn validate_sampled_terminal_dwa_paths(
    dwa: &DWA,
    id_map: &InternalIdMap,
    vocab: &Vocab,
) {
    if dwa.states().is_empty() {
        return;
    }

    let max_rep_bytes = id_map
        .vocab_tokens
        .iter_representative_ids()
        .filter_map(|token_id| vocab.entries.get(&token_id).map(Vec::len))
        .max()
        .unwrap_or(0);
    if max_rep_bytes == 0 {
        return;
    }

    let max_path_len = max_rep_bytes.saturating_add(1);
    let mut sampled_paths = 0usize;
    let mut sampled_tokens = 0usize;

    for seed in 0..L2P_PATH_VALIDATION_WALKS {
        let mut state = dwa.start_state();
        let mut path: Vec<i32> = Vec::new();
        let mut prefix_weight = Weight::all();

        for step in 0..max_path_len {
            if !path.is_empty() {
                if let Some(final_weight) = dwa.states()[state as usize].final_weight.as_ref() {
                    let accepted_weight = prefix_weight.intersection(final_weight);
                    if !accepted_weight.is_empty() {
                        sampled_paths += 1;
                        let sampled = sampled_internal_tokens(
                            &accepted_weight,
                            L2P_PATH_VALIDATION_TOKENS_PER_WEIGHT,
                        );
                        sampled_tokens += sampled.len();
                        validate_sampled_path_token_lengths(&path, &accepted_weight, id_map, vocab);
                    }
                }
            }

            let transitions = &dwa.states()[state as usize].transitions;
            if transitions.is_empty() {
                break;
            }

            let index = mixed_walk_index(seed, step, state, transitions.len());
            let (&label, (next_state, edge_weight)) = transitions.iter().nth(index).unwrap();
            assert!(
                label >= 0,
                "L2P terminal DWA validation expected terminal labels, got negative label {} on path {:?}",
                label,
                path,
            );

            let next_weight = prefix_weight.intersection(edge_weight);
            if next_weight.is_empty() {
                break;
            }

            path.push(label);
            prefix_weight = next_weight;
            state = *next_state;
        }
    }

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l2p_path_validation] sampled_paths={} sampled_tokens={} max_path_len={} walks={}",
            sampled_paths,
            sampled_tokens,
            max_path_len,
            L2P_PATH_VALIDATION_WALKS,
        );
    }
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
    initial_state_map: Option<&ManyToOneIdMap>,
) -> Option<LocalIdMapTerminalDwa> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();
    let num_original_states = tokenizer.num_states() as usize;
    let num_active_terminals = active_terminals.iter().filter(|&&active| active).count();

    // Cheap pre-check: if any original DFA state has zero active-terminal
    // coverage, the simplified state map cannot be total and the result of
    // simplify_for_terminals will be discarded, so skip it entirely.
    let simplification_can_be_total =
        tokenizer.active_terminal_filter_can_preserve_total_state_map(active_terminals);

    let (simplified_tok, simplify_state_map, use_simplified_tok, simplify_ms, candidate_unmapped_original_states) =
        if simplification_can_be_total {
            let mut relevant_bytes = [false; 256];
            for bytes in vocab.entries.values() {
                for &byte in bytes {
                    relevant_bytes[byte as usize] = true;
                }
            }

            // Strip non-active terminal bits from DFA finalizers and minimize.
            // Relevant-byte transition pruning is only applied when
            // GLRMASK_FORCE_RELEVANT_BYTES is set, because the default must
            // preserve commit/mask equivalence. This keeps L2P construction on
            // the smaller DFA needed by this partition.
            let simplify_started_at = Instant::now();
            let (candidate_tok, candidate_state_map) = tokenizer.simplify_for_terminals(
                active_terminals,
                Some(&relevant_bytes),
            );
            let simplify_ms = simplify_started_at.elapsed().as_secs_f64() * 1000.0;
            let candidate_unmapped_original_states = candidate_state_map
                .original_to_internal
                .iter()
                .filter(|&&state| state == u32::MAX)
                .count();
            let use_simplified_tok = candidate_unmapped_original_states == 0;
            if use_simplified_tok {
                (candidate_tok, Some(candidate_state_map), true, simplify_ms, candidate_unmapped_original_states)
            } else {
                let identity = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                    (0..num_original_states as u32).collect(),
                    num_original_states as u32,
                );
                (tokenizer.clone(), Some(identity), false, simplify_ms, candidate_unmapped_original_states)
            }
        } else {
            let identity = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                (0..num_original_states as u32).collect(),
                num_original_states as u32,
            );
            (tokenizer.clone(), Some(identity), false, 0.0, num_original_states)
        };

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l2p_simplify] partition={} original_states={} simplified_states={} unmapped_original_states={} used={} total_candidate={}",
            partition_label,
            num_original_states,
            simplified_tok.num_states(),
            candidate_unmapped_original_states,
            use_simplified_tok,
            simplification_can_be_total,
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
        if use_simplified_tok { None } else { flat_trans },
        if use_simplified_tok { None } else { initial_state_map },
    );
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    // tsid_fallback is independent of the NWA build / postprocess /
    // determinize / minimize pipeline: it only feeds into the final
    // `LocalIdMapTerminalDwa` struct. Run it in parallel with steps 2-8
    // via rayon::join. On the critical partition this cuts ~150-200ms
    // off the per-partition sequential critical path.
    let tokenizer = &simplified_tok;
    let (
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
        ) = {
            // ---- Step 2-3: Internal vocab + prefix tree ----
            let vocab_tree_started_at = Instant::now();
            let internal_vocab = internal_vocab_entries(vocab, &simplified_id_map);
            if internal_vocab.is_empty() {
                // Signal early-None via a sentinel. Build a dummy DWA;
                // outer code will observe `early_none=true` and return.
                (
                    crate::automata::weighted_u32::dwa::DWA::new(0, 0),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0usize, 0usize, 0usize, 0usize, 0usize,
                    crate::automata::weighted_u32::dwa::DWA::new(0, 0).stats(),
                    crate::automata::weighted_u32::dwa::DWA::new(0, 0).stats(),
                    true,
                )
            } else {
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
            }
        };
    if early_none {
        return None;
    }
    let composed_id_map = if use_simplified_tok {
        let simplify_state_map = simplify_state_map.expect("simplified path must have a state map");
        InternalIdMap {
            tokenizer_states: simplify_state_map.compose(&simplified_id_map.tokenizer_states),
            vocab_tokens: simplified_id_map.vocab_tokens.clone(),
        }
    } else {
        simplified_id_map.clone()
    };
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
            0.0,
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

    if l2p_path_validation_enabled() {
        validate_sampled_terminal_dwa_paths(&dwa, &composed_id_map, vocab);
    }

    Some(LocalIdMapTerminalDwa {
        id_map: composed_id_map,
        dwa,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::flat::GrammarDef;

    #[test]
    fn test_build_terminal_dwa_has_one_clean_match_and_one_incomplete_future_edge() {
        let gdef = GrammarDef {
            rules: vec![
                crate::grammar::flat::Rule {
                    lhs: 0,
                    rhs: vec![crate::grammar::flat::Symbol::Terminal(0)],
                },
                crate::grammar::flat::Rule {
                    lhs: 0,
                    rhs: vec![crate::grammar::flat::Symbol::Terminal(1)],
                },
            ],
            start: 0,
            terminals: vec![
                crate::grammar::flat::Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                crate::grammar::flat::Terminal::Literal {
                    id: 1,
                    bytes: b"ab".to_vec(),
                },
            ],
            ..Default::default()
        };
        let analyzed = AnalyzedGrammar::from_grammar_def(&gdef);
        let tokenizer = crate::compiler::compile::build_tokenizer(&gdef);
        let vocab = Vocab::new(vec![(0, b"a".to_vec())], None);
        let active_terminals = vec![true, true];
        let terminal_coloring = TerminalColoring::identity(analyzed.num_terminals as usize);

        let built = build_l2p_id_map_and_terminal_dwa(
            "test",
            &tokenizer,
            &vocab,
            &terminal_coloring,
            false,
            None,
            &analyzed,
            &active_terminals,
            &BTreeMap::new(),
            None,
            None,
            None,
        )
        .expect("L2P terminal DWA should build for overlapping-prefix test case");
        let terminal_dwa = built.dwa;

        assert_eq!(terminal_dwa.num_transitions(), 2);

        let start_state = terminal_dwa.start_state() as usize;
        let start_transitions: Vec<(i32, u32)> = terminal_dwa.states()[start_state]
            .transitions
            .iter()
            .map(|(&label, (target, _))| (label, *target))
            .collect();
        assert_eq!(start_transitions.len(), 2);
        assert_eq!(
            start_transitions.iter().map(|(label, _)| *label).collect::<Vec<_>>(),
            vec![0, 1]
        );

        let clean_target = start_transitions
            .iter()
            .find(|(label, _)| *label == 0)
            .map(|(_, target)| *target)
            .expect("clean terminal edge should exist");
        assert!(terminal_dwa.states()[clean_target as usize].final_weight.is_some());
    }
}
