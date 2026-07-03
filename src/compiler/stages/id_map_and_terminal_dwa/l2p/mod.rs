//! L2+ terminal DWA: full NWA-based construction for terminals with path length ≥ 2.
//!
//! Uses the same structure as the pre-partition/path-length code (commit 67146d8):
//! build vocab trie → compute possible_matches → seed root nodes → trie-walk
//! NWA build → postprocess (always_allowed → collapse → disallowed → prune →
//! canonicalize) → determinize → minimize.
//!
//! The only structural difference from the old code is `active_terminals`
//! filtering: terminals not in the L2+ set are skipped during the trie walk.

use crate::automata::lexer::Lexer;
pub(crate) mod equivalence_analysis;
pub(crate) mod nwa_builder;
pub(crate) mod postprocess;
mod terminal_dwa_equivalence;
mod terminal_interchangeability;

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::minimize_owned;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::possible_matches::PossibleMatchesComputer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::mapped_artifact::MappedArtifact;
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

use super::grammar_helpers::compute_always_allowed_follows;
use super::types::{compile_profile_enabled, TerminalColoring, TerminalDwaPhaseProfile};
use nwa_builder::{
    build_nwa_via_trie_walk, build_transport_nwa_via_trie_walk, internal_vocab_entries,
    seed_root_nodes,
};
use terminal_interchangeability::TerminalInterchangeability;
use postprocess::{
    apply_disallowed_follow_constraints, canonicalize_acyclic_nwa, collapse_always_allowed,
    prune_non_coreachable_states,
};

fn l2p_timing_profile_enabled() -> bool {
    compile_profile_enabled() || std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
}





thread_local! {
    static TERMINAL_INTERCHANGEABILITY_SUPPRESS_DEPTH: Cell<u32> = const { Cell::new(0) };
}

struct SuppressTerminalInterchangeability;

impl SuppressTerminalInterchangeability {
    fn new() -> Self {
        TERMINAL_INTERCHANGEABILITY_SUPPRESS_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for SuppressTerminalInterchangeability {
    fn drop(&mut self) {
        TERMINAL_INTERCHANGEABILITY_SUPPRESS_DEPTH.with(|depth| {
            depth.set(depth.get().checked_sub(1).expect("unbalanced terminal interchangeability suppression"));
        });
    }
}

/// Enable the deliberately slow strict terminal-interchangeability reference
/// construction. It preserves raw lexer-state coordinates, uses a
/// transport-aware trie walk, and checks the completed local artifact against a
/// baseline build before returning it.
fn l2p_terminal_interchangeability_enabled() -> bool {
    if TERMINAL_INTERCHANGEABILITY_SUPPRESS_DEPTH.with(|depth| depth.get() != 0) {
        return false;
    }
    std::env::var("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

#[derive(Clone, Copy)]
struct L2PTokenLengthStats {
    max_len: usize,
    gt_4: usize,
    gt_8: usize,
    gt_16: usize,
    gt_32: usize,
    gt_64: usize,
}

fn l2p_token_length_stats(vocab: &Vocab) -> L2PTokenLengthStats {
    let mut stats = L2PTokenLengthStats {
        max_len: 0,
        gt_4: 0,
        gt_8: 0,
        gt_16: 0,
        gt_32: 0,
        gt_64: 0,
    };

    for bytes in vocab.entries.values() {
        let len = bytes.len();
        stats.max_len = stats.max_len.max(len);
        if len > 4 {
            stats.gt_4 += 1;
        }
        if len > 8 {
            stats.gt_8 += 1;
        }
        if len > 16 {
            stats.gt_16 += 1;
        }
        if len > 32 {
            stats.gt_32 += 1;
        }
        if len > 64 {
            stats.gt_64 += 1;
        }
    }

    stats
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
/// 7. Determinize → minimize → compact.
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
    shared_original_vocab_dfa_cache: Option<&equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_original_vocab_analysis_dfa_cache: Option<&equivalence_analysis::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_transition_cache: Option<&OnceLock<equivalence_analysis::compat::FlatTransitionCache>>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> Option<LocalIdMapTerminalDwa> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();
    let num_original_states = tokenizer.num_states() as usize;
    let num_active_terminals = active_terminals.iter().filter(|&&active| active).count();

    let mut relevant_bytes = [false; 256];
    for bytes in vocab.entries.values() {
        for &byte in bytes {
            relevant_bytes[byte as usize] = true;
        }
    }

    // Discover terminal interchangeability in the current vocabulary byte
    // partition. The L2+ active mask limits which terminals may form a class,
    // but a transported scanner must retain the complete terminal alphabet.
    let terminal_interchangeability = if l2p_terminal_interchangeability_enabled() {
        TerminalInterchangeability::build(
            tokenizer,
            active_terminals,
            &relevant_bytes,
            ignore_terminal,
        )
    } else {
        TerminalInterchangeability::identity(active_terminals)
    };
    let reference_terminal_expansion = !terminal_interchangeability.is_identity();
    let analysis_active_terminals = terminal_interchangeability.active_representatives();
    let terminal_nwa_visible_output_labels =
        terminal_interchangeability.visible_output_raw_labels();
    let num_analysis_active_terminals = analysis_active_terminals
        .iter()
        .filter(|&&active| active)
        .count();
    let tokenizer_for_build = tokenizer;
    let equivalence_initial_state_map = initial_state_map;

    // ---- Step 1: Equivalence analysis (raw tokenizer state IDs) ----
    let id_map_started_at = Instant::now();
    // Every L2P partition shares the raw tokenizer transition relation.
    // Derive its exact byte partition lazily from sparse lexer edges once.
    let shared_base_setup_started_at = Instant::now();
    let equivalence_vocab_dfa_cache = if reference_terminal_expansion {
        None
    } else if let (Some(original_cache), Some(transition_cache), Some(flat_trans)) = (
        shared_original_vocab_dfa_cache,
        shared_transition_cache,
        flat_trans,
    ) {
        original_cache.get_or_init(|| {
            let transition_cache = transition_cache.get_or_init(|| {
                equivalence_analysis::compat::derive_flat_transition_cache(
                    tokenizer_for_build,
                    Arc::clone(flat_trans),
                )
            });
            equivalence_analysis::vocab::fast::SharedVocabDfaBase::build_from_flat_transition_cache(
                transition_cache,
            )
        });
        Some(original_cache)
    } else {
        shared_vocab_dfa_cache
    };
    let shared_base_setup_ms = shared_base_setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let shared_analysis_dfa_cache = (!reference_terminal_expansion)
        .then_some(shared_original_vocab_analysis_dfa_cache)
        .flatten();
    // Do not remove this note. Do not remove the instruction saying not to
    // remove this note. L2P state/vocab equivalence must always run fully.
    // Max-length may be skipped in controlled cases, but the full exact
    // state/vocab equivalence pass must not be bypassed. Do not reintroduce
    // fast-sound, identity, lex-dedup, or similar shortcut id-map paths.
    let fast_sound_id_map_used = false;
    // Keep raw lexer-state coordinates for the final scanner, but restrict
    // equivalence observations to this L2P partition's active terminals. With
    // TI enabled this is the representative mask, so all three equivalence
    // passes ignore class members replaced by their representatives.
    let (mut simplified_id_map, equiv_profile) =
        equivalence_analysis::combined::analyze_equivalences_with_group_filter(
            partition_label,
            tokenizer_for_build,
            vocab,
            disallowed_follows,
            ignore_terminal,
            Some(analysis_active_terminals),
            equivalence_vocab_dfa_cache,
            shared_analysis_dfa_cache,
            shared_base_setup_ms,
            if reference_terminal_expansion { None } else { flat_trans },
            equivalence_initial_state_map,
        );

    // Transport modes are indexed by raw tokenizer states. Keep ordinary
    // equivalence reduction for the baseline path, but retain every raw state
    // coordinate while constructing the transported reference artifact.
    if reference_terminal_expansion {
        let states = (0..tokenizer_for_build.num_states()).collect::<Vec<u32>>();
        simplified_id_map.tokenizer_states =
            ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                states.clone(),
                states,
            );
    }

    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    // tsid_fallback is independent of the NWA build / postprocess /
    // determinize / minimize pipeline: it only feeds into the final
    // `LocalIdMapTerminalDwa` struct. Run it in parallel with steps 2-8
    // via rayon::join. On the critical partition this cuts ~150-200ms
    // off the per-partition sequential critical path.
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
        internal_vocab_count,
        nwa_states_after_build,
        nwa_states_after_collapse,
        nwa_states_after_disallowed,
        nwa_states_after_prune,
        nwa_states_after_canonicalize,
        dwa_stats_before_compact,
        early_none,
    ) = {
        // ---- Step 2-3: Internal vocab + prefix tree ----
        let vocab_tree_started_at = Instant::now();
        let internal_vocab = internal_vocab_entries(vocab, &simplified_id_map);
        let internal_vocab_count = internal_vocab.len();
        if internal_vocab.is_empty() {
            // Signal early-None via a sentinel. Build a dummy DWA;
            // outer code will observe `early_none=true` and return.
            (
                crate::automata::weighted_u32::dwa::DWA::new(0, 0),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0usize,
                0usize,
                0usize,
                0usize,
                0usize,
                0usize,
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
            let mut pm_computer = PossibleMatchesComputer::new(tokenizer_for_build);
            let possible_matches_ms = 0.0;

            // ---- Step 5: Create NWA and seed root nodes ----
            let seed_started_at = Instant::now();
            let mut nwa = NWA::new(
                simplified_id_map.num_tsids(),
                simplified_id_map.max_internal_token_id(),
            );
            let leaf_state = nwa.add_state();
            nwa.set_final_weight(leaf_state, Weight::all());
            let start_state = nwa.add_state();
            nwa.start_states_mut().push(start_state);

            let transport_modes = reference_terminal_expansion
                .then(|| terminal_interchangeability.terminal_nwa_transport_modes())
                .flatten();
            let seed_ms;

            // ---- Step 6: Trie-walk NWA build ----
            let trie_build_started_at = Instant::now();
            let _build_profile = if let Some(modes) = transport_modes.as_deref() {
                seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;
                build_transport_nwa_via_trie_walk(
                    tokenizer_for_build,
                    ignore_terminal,
                    &mut nwa,
                    start_state,
                    leaf_state,
                    &simplified_id_map,
                    &full_tree.root,
                    &mut pm_computer,
                    &terminal_nwa_visible_output_labels,
                    modes,
                )
            } else {
                let roots = seed_root_nodes(&mut nwa, start_state, &simplified_id_map);
                seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;
                build_nwa_via_trie_walk(
                    tokenizer_for_build,
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
                )
            };
            let trie_build_ms = trie_build_started_at.elapsed().as_secs_f64() * 1000.0;

            let always_allowed_started_at = Instant::now();
            let always_allowed = compute_always_allowed_follows(grammar);
            let always_allowed_ms = always_allowed_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_build = nwa.states().len();

            let collapse_started_at = Instant::now();
            collapse_always_allowed(&mut nwa, &always_allowed, grammar.num_terminals as usize);
            let collapse_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;
            let nwa_states_after_collapse = nwa.states().len();

            let disallowed_started_at = Instant::now();
            apply_disallowed_follow_constraints(
                &mut nwa,
                disallowed_follows,
                grammar.num_terminals as usize,
                ignore_terminal,
            );
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
            let skip_minimize = std::env::var("GLRMASK_SKIP_L2P_MINIMIZE")
                .map(|value| {
                    let trimmed = value.trim();
                    trimmed.is_empty() || trimmed == "1" || trimmed.eq_ignore_ascii_case("true")
                })
                .unwrap_or(false);
            let dwa = if skip_minimize { det } else { minimize_owned(det) };
            let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;
            let dwa_stats_before_compact = dwa.stats();

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
                internal_vocab_count,
                nwa_states_after_build,
                nwa_states_after_collapse,
                nwa_states_after_disallowed,
                nwa_states_after_prune,
                nwa_states_after_canonicalize,
                dwa_stats_before_compact,
                false,
            )
        }
    };
    if early_none {
        return None;
    }
    let composed_id_map = simplified_id_map.clone();
    let postprocess_ms =
        always_allowed_ms + collapse_ms + disallowed_ms + prune_ms + canonicalize_ms;
    let max_length_reduction_pct = if equiv_profile.initial_states_considered == 0 {
        0.0
    } else {
        100.0
            * (1.0
                - equiv_profile.max_length_reps as f64
                    / equiv_profile.initial_states_considered as f64)
    };
    let exact_reduction_pct = if equiv_profile.initial_states_considered == 0 {
        0.0
    } else {
        100.0
            * (1.0
                - equiv_profile.exact_reps as f64 / equiv_profile.initial_states_considered as f64)
    };

    let profiling = compile_profile_enabled();
    let mut mapped_dwa = MappedArtifact::new(dwa, composed_id_map);
    let compact_started_at = Instant::now();
    if profiling {
        mapped_dwa.compact_dimensions_fast_with_stats();
    } else {
        mapped_dwa.compact_dimensions_fast();
    }
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
    let dwa_stats_after_compact = mapped_dwa.artifact().stats();
    let (dwa, id_map) = mapped_dwa.into_parts();

    if l2p_timing_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p] partition={} vocab_tokens={} active_terminals={} original_states={} tsids={} internal_vocab_entries={} initial_states_considered={} max_length_skipped={} max_token_len={} token_len_gt_4={} token_len_gt_8={} token_len_gt_16={} token_len_gt_32={} token_len_gt_64={} prepare_inputs_ms={:.3} byte_class_setup_ms={:.3} token_dedup_ms={:.3} max_length_state_equiv_ms={:.3} vocab_equiv_ms={:.3} exact_state_equiv_ms={:.3} id_map_finalize_ms={:.3} max_length_reps={} exact_reps={} exact_rep_confirmation_used={} fast_sound_id_map_used={} max_length_reduction_pct={:.2} exact_reduction_pct={:.2} restricted_observation_state_equiv_ms={:.3} restricted_observation_reps={} id_map_ms={:.3} tsid_fallback_ms={:.3} vocab_tree_ms={:.3} possible_matches_ms={:.3} seed_ms={:.3} terminal_nwa_build_ms={:.3} nwa_states={}->{}->{}->{}->{} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} prune_ms={:.3} canonicalize_ms={:.3} postprocess_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} compact_ms={:.3} minimize_states={} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            num_active_terminals,
            num_original_states,
            simplified_id_map.num_tsids(),
            internal_vocab_count,
            equiv_profile.initial_states_considered,
            equiv_profile.max_length_skipped,
            equiv_profile.max_token_len,
            equiv_profile.token_len_gt_4,
            equiv_profile.token_len_gt_8,
            equiv_profile.token_len_gt_16,
            equiv_profile.token_len_gt_32,
            equiv_profile.token_len_gt_64,
            equiv_profile.prepare_inputs_ms,
            equiv_profile.byte_class_setup_ms,
            equiv_profile.token_dedup_ms,
            equiv_profile.max_length_state_equiv_ms,
            equiv_profile.vocab_equiv_ms,
            equiv_profile.exact_state_equiv_ms,
            equiv_profile.id_map_finalize_ms,
            equiv_profile.max_length_reps,
            equiv_profile.exact_reps,
            equiv_profile.exact_rep_confirmation_used,
            fast_sound_id_map_used,
            max_length_reduction_pct,
            exact_reduction_pct,
            equiv_profile.restricted_observation_state_equiv_ms,
            equiv_profile.restricted_observation_reps,
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
            compact_ms,
            dwa_stats_before_compact.states,
            dwa_stats_after_compact.states,
            dwa_stats_after_compact.transitions,
            dwa_stats_after_compact.transition_pairs,
            dwa_stats_before_compact.interned_ranges,
            dwa_stats_after_compact.interned_ranges,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let output = LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            id_map_ms,
            terminal_dwa_ms: vocab_tree_ms
                + possible_matches_ms
                + seed_ms
                + trie_build_ms
                + postprocess_ms
                + determinize_ms
                + minimize_ms,
            compact_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    };


    if reference_terminal_expansion {
        // Rebuild the same local L2P artifact with the feature suppressed, then
        // compare the completed weighted terminal languages after expanding both
        // id maps into original tokenizer-state and token coordinates. This is
        // intentionally expensive: it is the correctness gate for the reference
        // construction.
        let baseline = {
            let _suppress = SuppressTerminalInterchangeability::new();
            build_l2p_id_map_and_terminal_dwa(
                partition_label,
                tokenizer,
                vocab,
                terminal_coloring,
                use_terminal_coloring,
                ignore_terminal,
                grammar,
                active_terminals,
                disallowed_follows,
                shared_vocab_dfa_cache,
                shared_original_vocab_dfa_cache,
                shared_original_vocab_analysis_dfa_cache,
                shared_transition_cache,
                flat_trans,
                initial_state_map,
            )
            .expect("terminal interchangeability baseline L2P build unexpectedly returned None")
        };
        terminal_dwa_equivalence::compare(&baseline, &output).unwrap_or_else(|mismatch| {
            panic!(
                "terminal interchangeability candidate differed from baseline: partition={} {}",
                partition_label,
                mismatch,
            )
        });
    }

    Some(output)
}
