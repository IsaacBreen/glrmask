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
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::minimize;
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
use rustc_hash::FxHashMap;

use super::grammar_helpers::compute_always_allowed_follows;
use super::types::{TerminalColoring, TerminalDwaPhaseProfile, compile_profile_enabled};
use nwa_builder::{build_nwa_via_trie_walk, internal_vocab_entries, seed_root_nodes};
use postprocess::{
    apply_disallowed_follow_constraints, canonicalize_acyclic_nwa, collapse_always_allowed,
    prune_non_coreachable_states,
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
                Arc::new(tokenizer.simplify_for_terminals(
                    active_terminals,
                    Some(relevant_bytes),
                ))
            })) {
                Ok(computed) => {
                    *entry.result.lock().unwrap() = Some(Ok(computed.clone()));
                    entry.ready.notify_all();
                    return (computed.0.clone(), computed.1.clone(), false);
                }
                Err(payload) => {
                    *entry.result.lock().unwrap() = Some(Err(
                        "tokenizer.simplify_for_terminals panicked".into(),
                    ));
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
    let can_skip_simplify = num_active_terminals == active_terminals.len();
    let (simplified_tok_storage, simplify_state_map, simplify_cache_hit) =
        if can_skip_simplify {
            (None, None, false)
        } else if let Some(cache) = shared_simplify_cache {
            let (tok, map, cache_hit) = cache.simplify_for_terminals(tokenizer, active_terminals, &relevant_bytes);
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

    // ---- Step 1: Equivalence analysis (on simplified tokenizer) ----
    let id_map_started_at = Instant::now();
    let simplified_id_map = equivalence_analysis::combined::analyze_equivalences_with_group_filter(
        tokenizer_for_build,
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
                let dwa = minimize(&det);
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
    let postprocess_ms = always_allowed_ms + collapse_ms + disallowed_ms + prune_ms + canonicalize_ms;

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p] partition={} vocab_tokens={} active_terminals={} original_states={} tsids={} simplify_ms={:.3} simplify_cache_hit={} simplified_states={} id_map_ms={:.3} tsid_fallback_ms={:.3} vocab_tree_ms={:.3} possible_matches_ms={:.3} seed_ms={:.3} terminal_nwa_build_ms={:.3} nwa_states={}->{}->{}->{}->{} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} prune_ms={:.3} canonicalize_ms={:.3} postprocess_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} minimize_states={} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            num_active_terminals,
            num_original_states,
            simplified_id_map.num_tsids(),
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
        },
    })
}

