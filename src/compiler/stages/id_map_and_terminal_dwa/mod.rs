//! Top-level id_map + terminal DWA builder.
//!
//! The canonical path splits the vocabulary into 3 character-type partitions,
//! builds a per-partition `(InternalIdMap, DWA)`, and merges the results into
//! the final global `(InternalIdMap, DWA)`.

pub(crate) mod classify;
pub(crate) mod grammar_helpers;
pub(crate) mod l1;
pub(crate) mod l2p;
pub(crate) mod merge;
pub(crate) mod partition;
pub(crate) mod types;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::bitset::BitSet;
use crate::Vocab;

use classify::classify_vocab_char_type;
use types::{TerminalColoring, TerminalDwaPhaseProfile, compile_profile_enabled, debug_profile_enabled};

/// Build the global `(InternalIdMap, DWA)` for the full vocabulary.
///
/// 1. Splits vocab into 3 partitions by leading-byte character type.
/// 2. Builds each partition's `(InternalIdMap, DWA)` in parallel via
///    [`partition::build_partition_id_map_and_terminal_dwa`].
/// 3. Merges the 3 results via [`merge::merge_id_maps_and_terminal_dwas`].
pub(crate) fn build_id_map_and_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    external_classify_cache: Option<&classify::SharedClassifyCache>,
) -> (InternalIdMap, DWA, TerminalDwaPhaseProfile) {
    let total_started_at = Instant::now();
    let force_all_l2p = std::env::var("GLRMASK_FORCE_ALL_L2P").map_or(false, |v| v == "1");
    let mut profile = TerminalDwaPhaseProfile::default();

    // Split vocab into 4 partitions by character type (or single partition if forced).
    let partition_vocab_started_at = Instant::now();
    let mut partition_entries: [Vec<(u32, Vec<u8>)>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (&token_id, bytes) in &vocab.entries {
        if force_all_l2p {
            partition_entries[0].push((token_id, bytes.clone()));
        } else {
            let idx = classify_vocab_char_type(bytes) as usize;
            partition_entries[idx].push((token_id, bytes.clone()));
        }
    }
    let sub_vocabs: Vec<Vocab> = partition_entries
        .into_iter()
        .map(|entries| Vocab::new(entries, None))
        .collect();
    let partition_vocab_ms = partition_vocab_started_at.elapsed().as_secs_f64() * 1000.0;
    profile.id_map_ms += partition_vocab_ms;

    // Build flat DFA transition table once (shared across all partitions).
    let flat_trans_started_at = Instant::now();
    let flat_trans: Arc<[u32]> = Arc::from(l1::build_flat_transition_table(tokenizer));
    profile.terminal_dwa_ms += flat_trans_started_at.elapsed().as_secs_f64() * 1000.0;

    // Shared cache for vocab DFA base (byte classes, transition table, self-loop
    // bytes). Eagerly initialized before partitions start to avoid all 4 partition
    // threads blocking on OnceLock during parallel execution.
    // Since filter_for_terminals only modifies finalizers/possible_future_group_ids
    // without changing transitions, the cached base is valid for all partitions.
    let shared_vocab_dfa_cache = l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new();
    {
        let all_active: Vec<bool> = vec![true; grammar.num_terminals as usize];
        let temp_view = l2p::equivalence_analysis::compat::TokenizerView::new_filtered_from_flat_trans(
            &flat_trans, tokenizer, &all_active,
        );
        shared_vocab_dfa_cache.get_or_init(|| {
            l2p::equivalence_analysis::vocab::fast::SharedVocabDfaBase::build_from_dfa(temp_view.dfa())
        });
    }

    // Shared cache for terminal classification byte sets. The DFA scanning
    // (reachable_bytes, first_bytes, last_bytes) is identical across partitions;
    // only the vocab-dependent classification differs. Reuse external cache if
    // provided (already populated by compile.rs pre-classification), otherwise
    // create a fresh one for partition sharing.
    let owned_classify_cache = classify::SharedClassifyCache::new();
    let shared_classify_cache = external_classify_cache.unwrap_or(&owned_classify_cache);

    // Build each partition in parallel.
    let ((p0, p1), (p2, p3)) = rayon::join(
        || {
            rayon::join(
                || {
                    let started_at = Instant::now();
                    partition::build_partition_id_map_and_terminal_dwa(
                        "p0",
                        tokenizer,
                        &sub_vocabs[0],
                        terminal_coloring,
                        use_terminal_coloring,
                        ignore_terminal,
                        grammar,
                        disallowed_follows,
                        &flat_trans,
                        Some(&shared_vocab_dfa_cache),
                        Some(&shared_classify_cache),
                    ).map(|pair| (pair, started_at.elapsed().as_secs_f64() * 1000.0))
                },
                || {
                    let started_at = Instant::now();
                    partition::build_partition_id_map_and_terminal_dwa(
                        "p1",
                        tokenizer,
                        &sub_vocabs[1],
                        terminal_coloring,
                        use_terminal_coloring,
                        ignore_terminal,
                        grammar,
                        disallowed_follows,
                        &flat_trans,
                        Some(&shared_vocab_dfa_cache),
                        Some(&shared_classify_cache),
                    ).map(|pair| (pair, started_at.elapsed().as_secs_f64() * 1000.0))
                },
            )
        },
        || {
            rayon::join(
                || {
                    let started_at = Instant::now();
                    partition::build_partition_id_map_and_terminal_dwa(
                        "p2",
                        tokenizer,
                        &sub_vocabs[2],
                        terminal_coloring,
                        use_terminal_coloring,
                        ignore_terminal,
                        grammar,
                        disallowed_follows,
                        &flat_trans,
                        Some(&shared_vocab_dfa_cache),
                        Some(&shared_classify_cache),
                    ).map(|pair| (pair, started_at.elapsed().as_secs_f64() * 1000.0))
                },
                || {
                    let started_at = Instant::now();
                    partition::build_partition_id_map_and_terminal_dwa(
                        "p3",
                        tokenizer,
                        &sub_vocabs[3],
                        terminal_coloring,
                        use_terminal_coloring,
                        ignore_terminal,
                        grammar,
                        disallowed_follows,
                        &flat_trans,
                        Some(&shared_vocab_dfa_cache),
                        Some(&shared_classify_cache),
                    ).map(|pair| (pair, started_at.elapsed().as_secs_f64() * 1000.0))
                },
            )
        },
    );

    let p0_ms = p0.as_ref().map(|(_, ms)| *ms).unwrap_or(0.0);
    let p1_ms = p1.as_ref().map(|(_, ms)| *ms).unwrap_or(0.0);
    let p2_ms = p2.as_ref().map(|(_, ms)| *ms).unwrap_or(0.0);
    let p3_ms = p3.as_ref().map(|(_, ms)| *ms).unwrap_or(0.0);
    let dominant_partition_profile = [
        p0.as_ref().map(|(pair, ms)| (pair.profile, *ms)),
        p1.as_ref().map(|(pair, ms)| (pair.profile, *ms)),
        p2.as_ref().map(|(pair, ms)| (pair.profile, *ms)),
        p3.as_ref().map(|(pair, ms)| (pair.profile, *ms)),
    ]
    .into_iter()
    .flatten()
    .max_by(|(_, left_ms), (_, right_ms)| left_ms.total_cmp(right_ms))
    .map(|(phase_profile, _)| phase_profile)
    .unwrap_or_default();

    // Collect non-None results.
    let mut pairs: Vec<merge::LocalIdMapTerminalDwa> = Vec::new();
    if let Some((pair, _)) = p0 {
        pairs.push(pair);
    }
    if let Some((pair, _)) = p1 {
        pairs.push(pair);
    }
    if let Some((pair, _)) = p2 {
        pairs.push(pair);
    }
    if let Some((pair, _)) = p3 {
        pairs.push(pair);
    }

    if pairs.is_empty() {
        let num_states = tokenizer.num_states() as usize;
        let empty_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: vec![0u32; num_states],
                internal_to_originals: vec![(0..num_states as u32).collect()],
                representative_original_ids: vec![0],
            },
            vocab_tokens: ManyToOneIdMap {
                original_to_internal: Vec::new(),
                internal_to_originals: Vec::new(),
                representative_original_ids: Vec::new(),
            },
        };
        return (empty_map, DWA::new(1, 0), profile);
    }

    let num_tokenizer_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();

    let merge_started_at = Instant::now();
    let (merged, global_merge_profile) = if pairs.len() == 1 {
        // Single partition — already compacted by partition merge. Skip redundant global compact.
        (pairs.into_iter().next().unwrap(), TerminalDwaPhaseProfile::default())
    } else {
        let merged = merge::merge_id_maps_and_terminal_dwas(
            "global",
            pairs,
            num_tokenizer_states,
            max_token_id,
        );
        let global_merge_profile = merged.profile;
        (merged, global_merge_profile)
    };
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
    profile.add_assign(dominant_partition_profile);
    profile.add_assign(global_merge_profile);

    if compile_profile_enabled() || debug_profile_enabled() {
        eprintln!(
            "[glrmask/profile][split_terminal_dwa] partition_vocab_ms={:.3} p0_tokens={} p0_ms={:.3} p1_tokens={} p1_ms={:.3} p2_tokens={} p2_ms={:.3} p3_tokens={} p3_ms={:.3} global_merge_ms={:.3} accounted_id_map_ms={:.3} accounted_terminal_dwa_ms={:.3} accounted_compact_ms={:.3} accounted_total_ms={:.3} total_ms={:.3}",
            partition_vocab_ms,
            sub_vocabs[0].entries.len(),
            p0_ms,
            sub_vocabs[1].entries.len(),
            p1_ms,
            sub_vocabs[2].entries.len(),
            p2_ms,
            sub_vocabs[3].entries.len(),
            p3_ms,
            merge_ms,
            profile.id_map_ms,
            profile.terminal_dwa_ms,
            profile.compact_ms,
            profile.total_ms(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    if std::env::var("GLRMASK_DEBUG_DWA_DUMP").map_or(false, |v| v == "1") {
        emit_merged_token_map(&merged.dwa, vocab, &merged.id_map);
        emit_merged_dwa_dump(&merged.dwa);
    }

    (merged.id_map, merged.dwa, profile)
}

fn emit_merged_token_map(dwa: &DWA, vocab: &Vocab, id_map: &InternalIdMap) {
    use l2p::nwa_builder::internal_vocab_entries;
    let internal_vocab = internal_vocab_entries(vocab, id_map);
    let internal_bytes: std::collections::BTreeMap<u32, &[u8]> =
        internal_vocab.iter().map(|(id, bytes)| (*id, bytes.as_slice())).collect();
    let mut referenced_tokens = std::collections::BTreeSet::new();
    for state in &dwa.states {
        for (_, (_, weight)) in &state.transitions {
            for tid in weight.token_union().iter() {
                referenced_tokens.insert(tid);
            }
        }
        if let Some(fw) = &state.final_weight {
            for tid in fw.token_union().iter() {
                referenced_tokens.insert(tid);
            }
        }
    }
    for tid in &referenced_tokens {
        if let Some(bytes) = internal_bytes.get(tid) {
            let originals = id_map.vocab_tokens.internal_to_originals.get(*tid as usize)
                .map(|v| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
                .unwrap_or_else(|| "?".into());
            eprintln!(
                "[glrmask/debug][terminal_dwa][token_map] internal={} originals=[{}] bytes={:?}",
                tid, originals, String::from_utf8_lossy(bytes)
            );
        }
    }
}

fn emit_merged_dwa_dump(dwa: &DWA) {
    let num_states = dwa.num_states() as usize;
    let start_state = dwa.start_state as usize;
    let mut incoming_counts = vec![0usize; num_states];
    let mut outgoing_counts = vec![0usize; num_states];
    let mut final_states = 0usize;
    let mut self_loops = 0usize;
    let mut transitions_to_start = 0usize;
    let mut transitions_from_start = 0usize;
    let mut transitions_from_start_to_start = 0usize;

    for (state_id, state) in dwa.states.iter().enumerate() {
        outgoing_counts[state_id] = state.transitions.len();
        for (_, (target, _)) in &state.transitions {
            incoming_counts[*target as usize] += 1;
            if *target as usize == start_state {
                transitions_to_start += 1;
            }
            if state_id == start_state {
                transitions_from_start += 1;
            }
            if state_id == start_state && *target as usize == start_state {
                transitions_from_start_to_start += 1;
            }
            if *target as usize == state_id {
                self_loops += 1;
            }
        }
        if state.final_weight.is_some() {
            final_states += 1;
        }
    }

    eprintln!(
        "[glrmask/debug][terminal_dwa][dump] states={} final_states={} self_loops={} to_start={} from_start={} from_start_to_start={}",
        num_states, final_states, self_loops, transitions_to_start, transitions_from_start, transitions_from_start_to_start,
    );

    for (state_id, state) in dwa.states.iter().enumerate() {
        let incoming = incoming_counts[state_id];
        let outgoing = outgoing_counts[state_id];
        let to_start = state
            .transitions
            .values()
            .filter(|(to, _)| *to as usize == start_state)
            .count();
        let self_loop_count = state
            .transitions
            .values()
            .filter(|(to, _)| *to as usize == state_id)
            .count();
        let final_weight = state
            .final_weight
            .as_ref()
            .map(|weight| format!("{weight}"))
            .unwrap_or_else(|| "none".to_string());
        let start_mark = if state_id == start_state {
            " [START]"
        } else {
            ""
        };

        eprintln!(
            "[glrmask/debug][terminal_dwa][state] id={}{} incoming={} outgoing={} to_start={} self_loops={} final={}",
            state_id,
            start_mark,
            incoming,
            outgoing,
            to_start,
            self_loop_count,
            final_weight,
        );

        for (label, (target, weight)) in &state.transitions {
            eprintln!("    {label} -> State {target}");
            eprintln!("      weight: {weight}");
        }
    }
}
