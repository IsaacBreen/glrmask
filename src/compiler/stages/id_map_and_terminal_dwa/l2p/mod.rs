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

use std::collections::BTreeMap;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::minimize_owned;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::possible_matches::PossibleMatchesComputer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::Vocab;
use rustc_hash::FxHashMap;

use super::grammar_helpers::compute_always_allowed_follows;
use super::types::{compile_profile_enabled, TerminalColoring, TerminalDwaPhaseProfile};
use nwa_builder::{build_nwa_via_trie_walk, internal_vocab_entries, seed_root_nodes};
use postprocess::{
    apply_disallowed_follow_constraints, canonicalize_acyclic_nwa, collapse_always_allowed,
    prune_non_coreachable_states, SharedDisallowedFollowDfaCache,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SimplifyCacheKey {
    active_words: Vec<u64>,
    relevant_words: [u64; 4],
}

#[derive(Default)]
pub(crate) struct SharedSimplifyCache {
    entries: Mutex<FxHashMap<SimplifyCacheKey, Arc<SimplifyCacheEntry>>>,
}

struct SimplifyCacheEntry {
    result: Mutex<Option<Result<Arc<(Tokenizer, ManyToOneIdMap)>, Arc<str>>>>,
    ready: Condvar,
}

impl SimplifyCacheEntry {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            ready: Condvar::new(),
        }
    }
}

impl SharedSimplifyCache {
    fn key(active_terminals: &[bool], relevant_bytes: &[bool; 256]) -> SimplifyCacheKey {
        let mut active_words = vec![0u64; active_terminals.len().div_ceil(64)];
        for (idx, &active) in active_terminals.iter().enumerate() {
            if active {
                active_words[idx >> 6] |= 1u64 << (idx & 63);
            }
        }

        let mut relevant_words = [0u64; 4];
        for (idx, &relevant) in relevant_bytes.iter().enumerate() {
            if relevant {
                relevant_words[idx >> 6] |= 1u64 << (idx & 63);
            }
        }

        SimplifyCacheKey {
            active_words,
            relevant_words,
        }
    }

    fn simplify_for_terminals(
        &self,
        tokenizer: &Tokenizer,
        active_terminals: &[bool],
        relevant_bytes: &[bool; 256],
    ) -> (Tokenizer, ManyToOneIdMap, bool) {
        let key = Self::key(active_terminals, relevant_bytes);
        let (entry, owns_compute) = {
            let mut entries = self.entries.lock().unwrap();
            if let Some(entry) = entries.get(&key) {
                (entry.clone(), false)
            } else {
                let entry = Arc::new(SimplifyCacheEntry::new());
                entries.insert(key, entry.clone());
                (entry, true)
            }
        };

        if owns_compute {
            match catch_unwind(AssertUnwindSafe(|| {
                Arc::new(tokenizer.simplify_for_terminals(active_terminals, Some(relevant_bytes)))
            })) {
                Ok(computed) => {
                    *entry.result.lock().unwrap() = Some(Ok(computed.clone()));
                    entry.ready.notify_all();
                    return (computed.0.clone(), computed.1.clone(), false);
                }
                Err(payload) => {
                    *entry.result.lock().unwrap() =
                        Some(Err("tokenizer.simplify_for_terminals panicked".into()));
                    entry.ready.notify_all();
                    resume_unwind(payload);
                }
            }
        }

        let mut result = entry.result.lock().unwrap();
        loop {
            match result.as_ref() {
                Some(Ok(cached)) => {
                    return (cached.0.clone(), cached.1.clone(), true);
                }
                Some(Err(message)) => {
                    panic!("{message}");
                }
                None => {
                    result = entry.ready.wait(result).unwrap();
                }
            }
        }
    }
}

fn project_initial_state_map_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_L2P_PROJECT_INITIAL_STATE_MAP")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
}

fn l2p_tokenizer_simplify_disabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_L2P_DISABLE_TOKENIZER_SIMPLIFY")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
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

struct ProjectInitialStateMapProfile {
    used: bool,
    reason: &'static str,
    simplified_state_count: usize,
    projected_simplified_states: usize,
    unmapped_simplified_states_before_fill: usize,
    projected_initial_classes_before_compaction: usize,
    projected_initial_classes_after_compaction: usize,
    dead_class_added: bool,
}

impl ProjectInitialStateMapProfile {
    fn unused(reason: &'static str, simplified_state_count: usize) -> Self {
        Self {
            used: false,
            reason,
            simplified_state_count,
            projected_simplified_states: 0,
            unmapped_simplified_states_before_fill: 0,
            projected_initial_classes_before_compaction: 0,
            projected_initial_classes_after_compaction: 0,
            dead_class_added: false,
        }
    }
}

fn project_initial_state_map_for_simplified_tokenizer(
    initial_state_map: &ManyToOneIdMap,
    simplify_state_map: &ManyToOneIdMap,
) -> (Option<ManyToOneIdMap>, ProjectInitialStateMapProfile) {
    let simplified_state_count = simplify_state_map.num_internal_ids() as usize;
    if simplified_state_count == 0 {
        return (
            None,
            ProjectInitialStateMapProfile::unused("empty_simplified", simplified_state_count),
        );
    }

    let mut projected = vec![u32::MAX; simplified_state_count];
    let mut has_projected_state = false;

    for (original_state, &simplified_state) in
        simplify_state_map.original_to_internal.iter().enumerate()
    {
        if simplified_state == u32::MAX {
            continue;
        }

        let initial_class = initial_state_map
            .original_to_internal
            .get(original_state)
            .copied()
            .unwrap_or(u32::MAX);
        if initial_class == u32::MAX {
            continue;
        }

        let slot = &mut projected[simplified_state as usize];
        if *slot == u32::MAX {
            *slot = initial_class;
            has_projected_state = true;
        } else if *slot != initial_class {
            let projected_simplified_states = projected
                .iter()
                .filter(|&&initial_class| initial_class != u32::MAX)
                .count();
            let projected_initial_classes_before_compaction = projected
                .iter()
                .copied()
                .filter(|&initial_class| initial_class != u32::MAX)
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let unmapped_simplified_states_before_fill =
                simplified_state_count.saturating_sub(projected_simplified_states);
            return (
                None,
                ProjectInitialStateMapProfile {
                    used: false,
                    reason: "mixed_initial_class",
                    simplified_state_count,
                    projected_simplified_states,
                    unmapped_simplified_states_before_fill,
                    projected_initial_classes_before_compaction,
                    projected_initial_classes_after_compaction: 0,
                    dead_class_added: false,
                },
            );
        }
    }

    if !has_projected_state {
        return (
            None,
            ProjectInitialStateMapProfile::unused("no_projected_states", simplified_state_count),
        );
    }

    let projected_simplified_states = projected
        .iter()
        .filter(|&&initial_class| initial_class != u32::MAX)
        .count();
    let projected_initial_classes_before_compaction = projected
        .iter()
        .copied()
        .filter(|&initial_class| initial_class != u32::MAX)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let unmapped_simplified_states_before_fill =
        simplified_state_count.saturating_sub(projected_simplified_states);

    let mut remapped_classes = vec![u32::MAX; initial_state_map.num_internal_ids() as usize];
    let mut next_class = 0u32;
    let compacted_projected: Vec<u32> = projected
        .into_iter()
        .map(|initial_class| {
            if initial_class == u32::MAX {
                return u32::MAX;
            }
            let slot = &mut remapped_classes[initial_class as usize];
            if *slot == u32::MAX {
                *slot = next_class;
                next_class += 1;
            }
            *slot
        })
        .collect();

    let dead_class_added = compacted_projected
        .iter()
        .any(|&initial_class| initial_class == u32::MAX);
    let projected_initial_classes_after_compaction = next_class as usize;
    (
        Some(
            ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                compacted_projected,
                next_class,
            )
            .fill_unmapped_with_new_class(),
        ),
        ProjectInitialStateMapProfile {
            used: true,
            reason: "used",
            simplified_state_count,
            projected_simplified_states,
            unmapped_simplified_states_before_fill,
            projected_initial_classes_before_compaction,
            projected_initial_classes_after_compaction,
            dead_class_added,
        },
    )
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
    shared_simplify_cache: Option<&SharedSimplifyCache>,
    shared_disallowed_follow_dfa_cache: Option<&SharedDisallowedFollowDfaCache>,
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

    // Strip non-active terminal bits from DFA finalizers and minimize.
    // When every terminal remains active, reuse the original tokenizer.
    // Otherwise simplify to the smaller partition-local DFA.
    //
    // Unmapped original states (states with no active-terminal future
    // under this partition) are filled into a dead class via
    // `fill_unmapped_with_new_class` after composition, so we always
    // use the simplified tokenizer.
    let simplify_started_at = Instant::now();
    let can_skip_simplify = l2p_tokenizer_simplify_disabled() || num_active_terminals == active_terminals.len();
    let (simplified_tok_storage, simplify_state_map, simplify_cache_hit) = if can_skip_simplify {
        (None, None, false)
    } else if let Some(cache) = shared_simplify_cache {
        let (tok, map, cache_hit) =
            cache.simplify_for_terminals(tokenizer, active_terminals, &relevant_bytes);
        (Some(tok), Some(map), cache_hit)
    } else {
        let (tok, map) = tokenizer.simplify_for_terminals(active_terminals, Some(&relevant_bytes));
        (Some(tok), Some(map), false)
    };
    let simplify_ms = simplify_started_at.elapsed().as_secs_f64() * 1000.0;
    let use_simplified_tok = simplified_tok_storage.is_some();
    let tokenizer_for_build = simplified_tok_storage.as_ref().unwrap_or(tokenizer);
    let candidate_unmapped_original_states = simplify_state_map.as_ref().map_or(0, |state_map| {
        state_map
            .original_to_internal
            .iter()
            .filter(|&&state| state == u32::MAX)
            .count()
    });
    let projection_enabled = project_initial_state_map_enabled();
    let (projected_initial_state_map, projection_profile) = if !projection_enabled {
        (
            None,
            ProjectInitialStateMapProfile::unused(
                "env_disabled",
                simplify_state_map
                    .as_ref()
                    .map(|simplified| simplified.num_internal_ids() as usize)
                    .unwrap_or(0),
            ),
        )
    } else if initial_state_map.is_none() {
        (
            None,
            ProjectInitialStateMapProfile::unused(
                "no_initial_map",
                simplify_state_map
                    .as_ref()
                    .map(|simplified| simplified.num_internal_ids() as usize)
                    .unwrap_or(0),
            ),
        )
    } else if simplify_state_map.is_none() {
        (
            None,
            ProjectInitialStateMapProfile::unused("no_simplify_map", 0),
        )
    } else {
        project_initial_state_map_for_simplified_tokenizer(
            initial_state_map.expect("checked above"),
            simplify_state_map.as_ref().expect("checked above"),
        )
    };
    let equivalence_initial_state_map = if use_simplified_tok {
        projected_initial_state_map.as_ref()
    } else {
        initial_state_map
    };
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p_projection] partition={} projection_enabled={} simplify_branch_active={} projected_initial_state_map_used={} reason={} simplified_state_count={} projected_simplified_states={} unmapped_simplified_states_before_fill={} projected_initial_classes_before_compaction={} projected_initial_classes_after_compaction={} dead_class_added={}",
            partition_label,
            projection_enabled,
            use_simplified_tok,
            projection_profile.used,
            projection_profile.reason,
            projection_profile.simplified_state_count,
            projection_profile.projected_simplified_states,
            projection_profile.unmapped_simplified_states_before_fill,
            projection_profile.projected_initial_classes_before_compaction,
            projection_profile.projected_initial_classes_after_compaction,
            projection_profile.dead_class_added,
        );
    }

    // ---- Step 1: Equivalence analysis (on simplified tokenizer) ----
    let id_map_started_at = Instant::now();
    // Do not remove this note. Do not remove the instruction saying not to
    // remove this note. L2P state/vocab equivalence must always run fully.
    // Max-length may be skipped in controlled cases, but the full exact
    // state/vocab equivalence pass must not be bypassed. Do not reintroduce
    // fast-sound, identity, lex-dedup, or similar shortcut id-map paths.
    let fast_sound_id_map_used = false;
    let (simplified_id_map, equiv_profile) =
        equivalence_analysis::combined::analyze_equivalences_with_group_filter(
            partition_label,
            tokenizer_for_build,
            vocab,
            disallowed_follows,
            ignore_terminal,
            if use_simplified_tok { None } else { Some(active_terminals) },
            shared_vocab_dfa_cache,
            if use_simplified_tok { None } else { flat_trans },
            equivalence_initial_state_map,
        );
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
        dwa_stats_after_compact,
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

            let roots = seed_root_nodes(&mut nwa, start_state, &simplified_id_map);
            let seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;

            // ---- Step 6: Trie-walk NWA build ----
            let trie_build_started_at = Instant::now();
            let _build_profile = build_nwa_via_trie_walk(
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
            );
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
                shared_disallowed_follow_dfa_cache,
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
                internal_vocab_count,
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
        InternalIdMap {
            tokenizer_states: simplify_state_map
                .as_ref()
                .expect("simplify_state_map missing for simplified tokenizer")
                .compose(&simplified_id_map.tokenizer_states)
                .fill_unmapped_with_new_class(),
            vocab_tokens: simplified_id_map.vocab_tokens.clone(),
        }
    } else {
        simplified_id_map.clone()
    };
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

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p] partition={} vocab_tokens={} active_terminals={} original_states={} tsids={} internal_vocab_entries={} initial_states_considered={} max_length_skipped={} max_token_len={} token_len_gt_4={} token_len_gt_8={} token_len_gt_16={} token_len_gt_32={} token_len_gt_64={} prepare_inputs_ms={:.3} byte_class_setup_ms={:.3} token_dedup_ms={:.3} max_length_state_equiv_ms={:.3} vocab_equiv_ms={:.3} exact_state_equiv_ms={:.3} id_map_finalize_ms={:.3} max_length_reps={} exact_reps={} exact_rep_confirmation_used={} fast_sound_id_map_used={} max_length_reduction_pct={:.2} exact_reduction_pct={:.2} simplify_ms={:.3} simplify_cache_hit={} simplified_states={} id_map_ms={:.3} tsid_fallback_ms={:.3} vocab_tree_ms={:.3} possible_matches_ms={:.3} seed_ms={:.3} terminal_nwa_build_ms={:.3} nwa_states={}->{}->{}->{}->{} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} prune_ms={:.3} canonicalize_ms={:.3} postprocess_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} minimize_states={} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} total_ms={:.3}",
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
            simplify_ms,
            simplify_cache_hit,
            tokenizer_for_build.num_states(),
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
            ..TerminalDwaPhaseProfile::default()
        },
    })
}
