//! Merge multiple (InternalIdMap, DWA) pairs into one.
//!
//! Handles both overlapping vocabs (e.g., L1 + L2+ from the same partition)
//! and disjoint vocabs (e.g., different character-type partitions) uniformly
//! via composite-key refinement.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize_owned;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::mapped_artifact::MappedArtifact;
use crate::ds::weight::Weight;

use super::types::{LocalIdMapTerminalDwa, TerminalDwaPhaseProfile, compile_profile_enabled};

fn minimize_merged_terminal_dwa_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_MINIMIZE_MERGED_TERMINAL_DWA")
            .map(|value| {
                let trimmed = value.trim();
                !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
            })
            .unwrap_or(true)
    })
}

fn compact_merged_terminal_dwa_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_COMPACT_MERGED_TERMINAL_DWA")
            .map(|value| {
                let trimmed = value.trim();
                !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
            })
            .unwrap_or(true)
    })
}

/// Return whether each source owns a disjoint set of original vocabulary tokens.
///
/// After remapping, this is stronger than a representation detail: every
/// non-empty weight from a source is contained in that source's global token
/// domain.  Consequently intersections of contributions from different
/// sources are empty, which lets `direct_union_disjoint_token_domain_dwas`
/// distribute the weighted path intersection over the branch union exactly.
fn inputs_have_disjoint_token_domains(inputs: &[LocalIdMapTerminalDwa], max_token_id: u32) -> bool {
    let mut owner = vec![u32::MAX; max_token_id as usize + 1];

    for (input_index, input) in inputs.iter().enumerate() {
        for (original_token, &local_token) in input
            .id_map
            .vocab_tokens
            .original_to_internal
            .iter()
            .enumerate()
        {
            if local_token == u32::MAX {
                continue;
            }
            let slot = &mut owner[original_token];
            if *slot != u32::MAX {
                return false;
            }
            *slot = input_index as u32;
        }
    }

    true
}

/// Remap a deterministic weighted automaton directly, preserving the same
/// branch-local token domain handling as `remap_nwa_with_maps`.
fn remap_dwa_with_maps(
    dwa: &mut DWA,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global_tsid_count: usize,
) {
    let mut weight_cache = HashMap::<usize, Weight>::new();
    let mut token_cache = HashMap::<usize, Arc<RangeSetBlaze<u32>>>::new();

    for state in dwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = remap_weight_cached(
                final_weight,
                local_to_global_tsids,
                local_to_global_tokens,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
            if final_weight.is_empty() {
                state.final_weight = None;
            }
        }

        for (_, weight) in state.transitions.values_mut() {
            *weight = remap_weight_cached(
                weight,
                local_to_global_tsids,
                local_to_global_tokens,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
        }
        state.transitions.retain(|_, (_, weight)| !weight.is_empty());
    }
}

/// Deterministically union automata whose global token domains are disjoint.
///
/// For a word `x`, source `i` contributes `W_i(x)`, the intersection of its
/// edge and final weights.  Disjoint source domains imply
/// `(⋃ A_i) ∩ (⋃ B_i) = ⋃ (A_i ∩ B_i)`, so the subset construction need only
/// remember one deterministic state per source rather than a weighted subset.
/// Missing component states represent dead paths.  The output is exactly the
/// ordinary NWA-union determinization, before minimization.
fn direct_union_disjoint_token_domain_dwas(inputs: &[DWA]) -> DWA {
    assert!(!inputs.is_empty());

    const DEAD: u32 = u32::MAX;

    let mut output = DWA::new(0, 0);
    let start_tuple: Vec<u32> = inputs.iter().map(DWA::start_state).collect();
    let mut state_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut worklist = VecDeque::<Vec<u32>>::new();
    state_ids.insert(start_tuple.clone(), output.start_state());
    worklist.push_back(start_tuple);

    while let Some(component_states) = worklist.pop_front() {
        let from_state = state_ids[&component_states];

        let final_weight = Weight::union_all(
            component_states
                .iter()
                .enumerate()
                .filter_map(|(input_index, &state_id)| {
                    (state_id != DEAD)
                        .then(|| inputs[input_index].states()[state_id as usize].final_weight.as_ref())
                        .flatten()
                }),
        );
        if !final_weight.is_empty() {
            output.set_final_weight(from_state, final_weight);
        }

        let mut by_label = FxHashMap::<i32, Vec<(usize, u32, Weight)>>::default();
        for (input_index, &state_id) in component_states.iter().enumerate() {
            if state_id == DEAD {
                continue;
            }
            for (&label, &(target, ref weight)) in &inputs[input_index].states()[state_id as usize].transitions {
                by_label
                    .entry(label)
                    .or_default()
                    .push((input_index, target, weight.clone()));
            }
        }

        for (label, transitions) in by_label {
            let edge_weight = Weight::union_all(transitions.iter().map(|(_, _, weight)| weight));
            if edge_weight.is_empty() {
                continue;
            }

            let mut target_tuple = vec![DEAD; inputs.len()];
            for (input_index, target, _) in transitions {
                target_tuple[input_index] = target;
            }

            let to_state = if let Some(&existing) = state_ids.get(&target_tuple) {
                existing
            } else {
                let new_state = output.add_state();
                state_ids.insert(target_tuple.clone(), new_state);
                worklist.push_back(target_tuple);
                new_state
            };
            output.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    debug_assert!(output.is_acyclic());
    output
}

/// Build the exact deterministic union without subset determinization whenever
/// the source vocabularies are disjoint.  The generic NWA route remains the
/// fallback for overlapping domains.
fn try_merge_disjoint_token_domain_dwas(
    inputs: &[LocalIdMapTerminalDwa],
    global_id_map: &InternalIdMap,
    direct_local_to_global_token_maps: Option<&Vec<Vec<u32>>>,
    max_token_id: u32,
) -> Option<DWA> {
    if !inputs_have_disjoint_token_domains(inputs, max_token_id) {
        return None;
    }

    let mut remapped = Vec::with_capacity(inputs.len());
    for (input_index, input) in inputs.iter().enumerate() {
        let tsid_map = build_local_to_global_tsid_map(&input.id_map, global_id_map);
        let token_map = direct_local_to_global_token_maps
            .and_then(|maps| maps.get(input_index))
            .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
            .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, global_id_map));
        let mut dwa = input.dwa.clone();
        remap_dwa_with_maps(
            &mut dwa,
            &tsid_map,
            &token_map,
            global_id_map.num_tsids() as usize,
        );
        remapped.push(dwa);
    }

    Some(direct_union_disjoint_token_domain_dwas(&remapped))
}

/// Merge local branch outputs for a single partition into one compacted DWA.
pub(crate) fn merge_local_id_maps_and_terminal_dwas(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    assert!(!inputs.is_empty(), "merge_local_id_maps_and_terminal_dwas called with empty inputs");

    if inputs.len() == 1 {
        let mut input = inputs.into_iter().next().unwrap();
        input.profile = TerminalDwaPhaseProfile::default();
        return input;
    }

    let input_refs: Vec<&LocalIdMapTerminalDwa> = inputs.iter().collect();
    let id_map_refs: Vec<&InternalIdMap> = input_refs.iter().map(|input| &input.id_map).collect();
    let (global_id_map, direct_local_to_global_token_maps) =
        build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);

    let pre_minimize = try_merge_disjoint_token_domain_dwas(
        &inputs,
        &global_id_map,
        direct_local_to_global_token_maps.as_ref(),
        max_token_id,
    )
    .unwrap_or_else(|| {
        let mut global_nwa = NWA::new(
            global_id_map.num_tsids(),
            global_id_map.max_internal_token_id(),
        );
        let mut global_body = global_nwa.body();

        for local in &inputs {
            let mut nwa = local.dwa.to_nwa();
            let tsid_map = build_local_to_global_tsid_map(&local.id_map, &global_id_map);
            let token_map = build_local_to_global_token_map(&local.id_map, &global_id_map);
            remap_nwa_with_maps(
                &mut nwa,
                &tsid_map,
                &token_map,
                global_id_map.num_tsids() as usize,
            );
            global_body = global_nwa.union_in_place(&nwa, &global_body);
        }
        global_nwa.set_start_states(global_body.start_states);
        determinize(&global_nwa).expect("merge terminal NWA determinization failed")
    });
    let compact_started_at = Instant::now();
    let mut mapped_dwa = MappedArtifact::new(minimize_owned(pre_minimize), global_id_map);
    mapped_dwa.compact_dimensions_fast();
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
    let (dwa, id_map) = mapped_dwa.into_parts();

    LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            compact_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    }
}

/// Merge already-compacted partition outputs into one global DWA.
///
/// Partition-local merges have already minimized their local outputs. By
/// default the final cross-partition merge also minimizes after determinizing;
/// `GLRMASK_MINIMIZE_MERGED_TERMINAL_DWA=0` disables that final exact pass for
/// A/B comparison. The merged result is then compacted before returning unless
/// `GLRMASK_COMPACT_MERGED_TERMINAL_DWA=0` disables the final compaction pass.
pub(crate) fn merge_id_maps_and_terminal_dwas(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    assert!(!inputs.is_empty(), "merge_id_maps_and_terminal_dwas called with empty inputs");

    if inputs.len() == 1 {
        let mut input = inputs.into_iter().next().unwrap();
        input.profile = TerminalDwaPhaseProfile::default();
        return input;
    }

    let total_started_at = Instant::now();

    let build_unified_global_id_map_started_at = Instant::now();
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    let (global_id_map, direct_local_to_global_token_maps) =
        build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);
    let build_unified_global_id_map_ms =
        build_unified_global_id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let remap_and_union_started_at = Instant::now();
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();
    let profiling = compile_profile_enabled();
    let mut to_nwa_ms = 0.0;
    let mut build_tsid_map_ms = 0.0;
    let mut build_token_map_ms = 0.0;
    let mut remap_nwa_ms = 0.0;
    let mut union_ms = 0.0;

    for (input_idx, input) in inputs.iter().enumerate() {
        let started_at = profiling.then(Instant::now);
        let mut nwa = input.dwa.to_nwa();
        if let Some(started_at) = started_at {
            to_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let started_at = profiling.then(Instant::now);
        let tsid_map = build_local_to_global_tsid_map(&input.id_map, &global_id_map);
        if let Some(started_at) = started_at {
            build_tsid_map_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let started_at = profiling.then(Instant::now);
        let token_map = direct_local_to_global_token_maps
            .as_ref()
            .and_then(|maps| maps.get(input_idx))
            .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
            .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, &global_id_map));
        if let Some(started_at) = started_at {
            build_token_map_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let started_at = profiling.then(Instant::now);
        remap_nwa_with_maps(
            &mut nwa,
            &tsid_map,
            &token_map,
            global_id_map.num_tsids() as usize,
        );
        if let Some(started_at) = started_at {
            remap_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let started_at = profiling.then(Instant::now);
        global_body = global_nwa.union_in_place(&nwa, &global_body);
        if let Some(started_at) = started_at {
            union_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
    }
    global_nwa.set_start_states(global_body.start_states);
    let remap_and_union_ms = remap_and_union_started_at.elapsed().as_secs_f64() * 1000.0;
    let nwa_states_before_determinize = global_nwa.num_states();
    let nwa_transitions_before_determinize = global_nwa.num_transitions();

    let determinize_started_at = Instant::now();
    let det = determinize(&global_nwa)
        .expect("merge terminal NWA determinization failed");
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
    let det_states = det.num_states();
    let det_transitions = det.num_transitions();

    let minimize_enabled = minimize_merged_terminal_dwa_enabled();
    let minimize_started_at = Instant::now();
    let dwa = if minimize_enabled {
        minimize_owned(det)
    } else {
        det
    };
    let minimize_ms = if minimize_enabled {
        minimize_started_at.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let mut mapped_dwa = MappedArtifact::new(dwa, global_id_map);
    let before_compact_stats = profiling.then(|| mapped_dwa.artifact().stats());
    let before_compact_range_counts = profiling.then(|| mapped_dwa.interned_range_counts());
    let before_num_tsids = profiling.then(|| mapped_dwa.id_map().num_tsids());
    let before_num_tokens = profiling.then(|| mapped_dwa.id_map().num_internal_tokens());
    let compact_enabled = compact_merged_terminal_dwa_enabled();
    let (compact_report, compact_ms) = if compact_enabled {
        let compact_started_at = Instant::now();
        let compact_report = if profiling {
            mapped_dwa.compact_dimensions_fast_with_stats()
        } else {
            mapped_dwa.compact_dimensions_fast()
        };
        let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
        (Some(compact_report), compact_ms)
    } else {
        (None, 0.0)
    };
    let after_compact_stats = profiling.then(|| mapped_dwa.artifact().stats());
    let after_compact_range_counts = profiling.then(|| mapped_dwa.interned_range_counts());
    let total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;
    let (dwa, id_map) = mapped_dwa.into_parts();

    if profiling {
        let before_compact_stats = before_compact_stats.unwrap();
        let before_compact_range_counts = before_compact_range_counts.unwrap();
        let before_num_tsids = before_num_tsids.unwrap();
        let before_num_tokens = before_num_tokens.unwrap();
        let after_compact_stats = after_compact_stats.unwrap();
        let after_compact_range_counts = after_compact_range_counts.unwrap();
        let profile_stats = compact_report.and_then(|report| report.profile_stats);
        eprintln!(
            "[glrmask/profile][terminal_dwa_global_merge] inputs={} build_unified_global_id_map_ms={:.3} remap_and_union_ms={:.3} to_nwa_ms={:.3} build_tsid_map_ms={:.3} build_token_map_ms={:.3} remap_nwa_ms={:.3} union_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} compact_ms={:.3} total_ms={:.3} nwa_states_before_determinize={} nwa_transitions_before_determinize={} det_states={} det_transitions={} dwa_states_before_compact={} dwa_transitions_before_compact={} dwa_states_after_compact={} dwa_transitions_after_compact={}",
            inputs.len(),
            build_unified_global_id_map_ms,
            remap_and_union_ms,
            to_nwa_ms,
            build_tsid_map_ms,
            build_token_map_ms,
            remap_nwa_ms,
            union_ms,
            determinize_ms,
            minimize_ms,
            compact_ms,
            total_ms,
            nwa_states_before_determinize,
            nwa_transitions_before_determinize,
            det_states,
            det_transitions,
            before_compact_stats.states,
            before_compact_stats.transitions,
            after_compact_stats.states,
            after_compact_stats.transitions,
        );
        eprintln!(
            "[glrmask/profile][merged_terminal_dwa] minimize_enabled={} compact_enabled={} states_before_compact={} transitions_before_compact={} interned_ranges_before_compact={} token_ranges_before_compact={} states_after_compact={} transitions_after_compact={} interned_ranges_after_compact={} token_ranges_after_compact={} tsids_before_compact={} tsids_after_compact={} tokens_before_compact={} tokens_after_compact={} compact_ms={:.3}",
            minimize_merged_terminal_dwa_enabled(),
            compact_enabled,
            before_compact_stats.states,
            before_compact_stats.transitions,
            before_compact_stats.interned_ranges,
            before_compact_range_counts.token_ranges,
            after_compact_stats.states,
            after_compact_stats.transitions,
            after_compact_stats.interned_ranges,
            after_compact_range_counts.token_ranges,
            before_num_tsids,
            profile_stats.map(|stats| stats.tsids_after).unwrap_or(before_num_tsids as usize),
            before_num_tokens,
            profile_stats.map(|stats| stats.tokens_after).unwrap_or(before_num_tokens as usize),
            compact_ms,
        );
    }

    LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            compact_ms,
            global_merge_ms: total_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    }
}

fn build_unified_global_id_map(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> (InternalIdMap, Option<Vec<Vec<u32>>>) {
    let mut composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut state_o2i = vec![0u32; num_tokenizer_states];
    let mut state_i2o: Vec<Vec<u32>> = Vec::new();
    let mut state_reps: Vec<u32> = Vec::new();

    for state in 0..num_tokenizer_states {
        let composite: Vec<u32> = inputs
            .iter()
            .map(|input| input.tokenizer_states.original_to_internal[state])
            .collect();
        let next_id = state_i2o.len() as u32;
        let class = *composite_to_class.entry(composite).or_insert_with(|| {
            state_i2o.push(Vec::new());
            state_reps.push(state as u32);
            next_id
        });
        state_o2i[state] = class;
        state_i2o[class as usize].push(state as u32);
    }

    reorder_classes(composite_to_class, &mut state_o2i, &mut state_i2o, &mut state_reps);

    let (vocab_tokens, direct_local_to_global_token_maps) =
        build_unified_global_token_id_map_disjoint(inputs, max_token_id)
            .unwrap_or_else(|| (build_unified_global_token_id_map_generic(inputs, max_token_id), None));
    (
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: state_o2i,
                internal_to_originals: state_i2o,
                representative_original_ids: state_reps,
            },
            vocab_tokens,
        },
        direct_local_to_global_token_maps,
    )
}

fn build_unified_global_token_id_map_generic(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> ManyToOneIdMap {
    let mut token_composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut token_o2i = vec![u32::MAX; max_token_id as usize + 1];
    let mut token_i2o: Vec<Vec<u32>> = Vec::new();
    let mut token_reps: Vec<u32> = Vec::new();

    for token_id in 0..=max_token_id {
        let composite: Vec<u32> = inputs
            .iter()
            .map(|input| {
                input
                    .vocab_tokens
                    .original_to_internal
                    .get(token_id as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        if composite.iter().all(|&value| value == u32::MAX) {
            continue;
        }

        let next_id = token_i2o.len() as u32;
        let class = *token_composite_to_class
            .entry(composite)
            .or_insert_with(|| {
                token_i2o.push(Vec::new());
                token_reps.push(token_id);
                next_id
            });
        token_o2i[token_id as usize] = class;
        token_i2o[class as usize].push(token_id);
    }

    reorder_classes_with_sentinel(
        token_composite_to_class,
        &mut token_o2i,
        &mut token_i2o,
        &mut token_reps,
        u32::MAX,
    );

    ManyToOneIdMap {
        original_to_internal: token_o2i,
        internal_to_originals: token_i2o,
        representative_original_ids: token_reps,
    }
}

fn build_unified_global_token_id_map_disjoint(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> Option<(ManyToOneIdMap, Option<Vec<Vec<u32>>>)> {
    let mut owner_by_token = vec![usize::MAX; max_token_id as usize + 1];

    for (input_idx, input) in inputs.iter().enumerate() {
        for (token_id, &local_class) in input.vocab_tokens.original_to_internal.iter().enumerate() {
            if local_class == u32::MAX {
                continue;
            }
            let owner = &mut owner_by_token[token_id];
            if *owner != usize::MAX && *owner != input_idx {
                return None;
            }
            *owner = input_idx;
        }
    }

    let mut token_o2i = vec![u32::MAX; max_token_id as usize + 1];
    let mut token_i2o: Vec<Vec<u32>> = Vec::new();
    let mut token_reps: Vec<u32> = Vec::new();
    let mut direct_local_to_global_token_maps = Vec::with_capacity(inputs.len());

    for input in inputs {
        let mut local_to_global = vec![u32::MAX; input.num_internal_tokens() as usize];

        for (local_class, originals) in input.vocab_tokens.internal_to_originals.iter().enumerate() {
            if originals.is_empty() {
                continue;
            }

            let global_class = token_i2o.len() as u32;
            local_to_global[local_class] = global_class;
            token_reps.push(input.vocab_tokens.representative_original_ids[local_class]);
            token_i2o.push(originals.clone());
            for &token_id in originals {
                token_o2i[token_id as usize] = global_class;
            }
        }

        direct_local_to_global_token_maps.push(local_to_global);
    }

    Some((
        ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
        Some(direct_local_to_global_token_maps),
    ))
}

fn build_direct_local_to_global_token_map(local_to_global: &[u32]) -> Vec<Vec<u32>> {
    local_to_global
        .iter()
        .map(|&global_class| {
            if global_class == u32::MAX {
                Vec::new()
            } else {
                vec![global_class]
            }
        })
        .collect()
}

fn reorder_classes(
    composite_to_class: HashMap<Vec<u32>, u32>,
    o2i: &mut [u32],
    i2o: &mut Vec<Vec<u32>>,
    reps: &mut Vec<u32>,
) {
    let num_classes = i2o.len();
    if num_classes <= 1 {
        return;
    }

    let mut sorted: Vec<(Vec<u32>, u32)> = composite_to_class.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut old_to_new = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        old_to_new[*old_id as usize] = new_id as u32;
    }

    for val in o2i.iter_mut() {
        *val = old_to_new[*val as usize];
    }

    let mut new_i2o = vec![Vec::new(); num_classes];
    let mut new_reps = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        new_i2o[new_id] = std::mem::take(&mut i2o[*old_id as usize]);
        new_reps[new_id] = reps[*old_id as usize];
    }
    *i2o = new_i2o;
    *reps = new_reps;
}

fn reorder_classes_with_sentinel(
    composite_to_class: HashMap<Vec<u32>, u32>,
    o2i: &mut [u32],
    i2o: &mut Vec<Vec<u32>>,
    reps: &mut Vec<u32>,
    sentinel: u32,
) {
    let num_classes = i2o.len();
    if num_classes <= 1 {
        return;
    }

    let mut sorted: Vec<(Vec<u32>, u32)> = composite_to_class.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut old_to_new = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        old_to_new[*old_id as usize] = new_id as u32;
    }

    for val in o2i.iter_mut() {
        if *val != sentinel {
            *val = old_to_new[*val as usize];
        }
    }

    let mut new_i2o = vec![Vec::new(); num_classes];
    let mut new_reps = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        new_i2o[new_id] = std::mem::take(&mut i2o[*old_id as usize]);
        new_reps[new_id] = reps[*old_id as usize];
    }
    *i2o = new_i2o;
    *reps = new_reps;
}

/// Map local TSIDs to global TSIDs via original-state lookup.
fn build_local_to_global_tsid_map(
    local_id_map: &InternalIdMap,
    global_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local_id_map.num_tsids() as usize;
    let mut local_to_global = vec![BTreeSet::new(); num_local];

    for (state, &local_tsid) in local_id_map
        .tokenizer_states
        .original_to_internal
        .iter()
        .enumerate()
    {
        let global_tsid = global_id_map.tokenizer_states.original_to_internal[state];
        local_to_global[local_tsid as usize].insert(global_tsid);
    }

    local_to_global
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

/// Map local internal token classes to global internal token classes.
fn build_local_to_global_token_map(
    local_id_map: &InternalIdMap,
    global_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local_id_map.num_internal_tokens() as usize;
    let mut local_to_global = vec![BTreeSet::new(); num_local];

    for (orig, &local_class) in local_id_map
        .vocab_tokens
        .original_to_internal
        .iter()
        .enumerate()
    {
        if local_class == u32::MAX {
            continue;
        }
        let global_class = global_id_map
            .vocab_tokens
            .original_to_internal
            .get(orig)
            .copied()
            .unwrap_or(u32::MAX);
        if global_class == u32::MAX {
            continue;
        }
        local_to_global[local_class as usize].insert(global_class);
    }

    local_to_global
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

/// Remap all weights in an NWA from local TSID/token space to global space.
fn remap_nwa_with_maps(
    nwa: &mut NWA,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global_tsid_count: usize,
) {
    let mut weight_cache = HashMap::<usize, Weight>::new();
    let mut token_cache = HashMap::<usize, Arc<RangeSetBlaze<u32>>>::new();

    for state in  nwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = remap_weight_cached(
                final_weight,
                local_to_global_tsids,
                local_to_global_tokens,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
            if final_weight.is_empty() {
                state.final_weight = None;
            }
        }

        for targets in state.transitions.values_mut() {
            for (_, weight) in targets.iter_mut() {
                *weight = remap_weight_cached(
                    weight,
                    local_to_global_tsids,
                    local_to_global_tokens,
                    global_tsid_count,
                    &mut weight_cache,
                    &mut token_cache,
                );
            }
            targets.retain(|(_, weight)| !weight.is_empty());
        }
        state.transitions.retain(|_, targets| !targets.is_empty());

        for (_, weight) in state.epsilons.iter_mut() {
            *weight = remap_weight_cached(
                weight,
                local_to_global_tsids,
                local_to_global_tokens,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
        }
        state.epsilons.retain(|(_, weight)| !weight.is_empty());
    }

}

fn remap_weight_cached(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global_tsid_count: usize,
    cache: &mut HashMap<usize, Weight>,
    token_cache: &mut HashMap<usize, Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    let ptr = Arc::as_ptr(&weight.0) as usize;
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    let remapped = remap_weight_general(
        weight,
        local_to_global_tsids,
        local_to_global_tokens,
        global_tsid_count,
        token_cache,
    );
    cache.insert(ptr, remapped.clone());
    remapped
}

fn remap_weight_general(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global_tsid_count: usize,
    token_cache: &mut HashMap<usize, Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    if weight.is_empty() {
        return weight.clone();
    }

    if weight.is_full() {
        let mut all_global_tokens = RangeSetBlaze::new();
        for globals in local_to_global_tokens {
            for &global_token in globals {
                all_global_tokens.insert(global_token);
            }
        }
        if all_global_tokens.is_empty() {
            return Weight::empty();
        }

        let mut all_global_tsids = BTreeSet::new();
        for globals in local_to_global_tsids {
            for &global_tsid in globals {
                if (global_tsid as usize) < global_tsid_count {
                    all_global_tsids.insert(global_tsid);
                }
            }
        }
        if all_global_tsids.is_empty() {
            return Weight::empty();
        }

        return Weight::from_per_tsid_token_sets(
            all_global_tsids
                .into_iter()
                .map(|global_tsid| (global_tsid, all_global_tokens.clone())),
        );
    }

    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };
    use crate::ds::weight::{finalize_weight_map, shared_rangeset};

    let mut tokens_by_global_tsid = Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new();

    for (start, end, tokens) in entries {
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = if let Some(mapped_tokens) = token_cache.get(&token_key) {
            mapped_tokens.clone()
        } else {
            let mapped_tokens = {
                let mut result = RangeSetBlaze::new();
                for local_token in tokens.iter() {
                    if let Some(globals) = local_to_global_tokens.get(local_token as usize) {
                        for &g in globals {
                            result.insert(g);
                        }
                    }
                }
                Arc::new(result)
            };
            token_cache.insert(token_key, mapped_tokens.clone());
            mapped_tokens
        };

        for local_tsid in start..=end {
            let Some(global_tsids) = local_to_global_tsids.get(local_tsid as usize) else {
                continue;
            };
            for &global_tsid in global_tsids {
                let idx = global_tsid as usize;
                if idx >= global_tsid_count {
                    continue;
                }
                tokens_by_global_tsid.push((idx as u32, Arc::clone(&mapped_tokens)));
            }
        }
    }

    if tokens_by_global_tsid.is_empty() {
        return Weight::empty();
    }

    tokens_by_global_tsid.sort_unstable_by_key(|(global_tsid, _)| *global_tsid);

    let mut merged_by_global_tsid = Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new();
    for (global_tsid, tokens) in tokens_by_global_tsid {
        if let Some((last_tsid, last_tokens)) = merged_by_global_tsid.last_mut() {
            if *last_tsid == global_tsid {
                if !Arc::ptr_eq(last_tokens, &tokens) && last_tokens.as_ref() != tokens.as_ref() {
                    *last_tokens = shared_rangeset(last_tokens.as_ref() | tokens.as_ref());
                }
                continue;
            }
        }
        merged_by_global_tsid.push((global_tsid, tokens));
    }

    // Build WeightMap by scanning sparse sorted TSID/token-set entries for contiguous runs.
    use range_set_blaze::RangeMapBlaze;

    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start: Option<u32> = None;
    let mut run_end: u32 = 0;
    let mut run_shared: Option<Arc<RangeSetBlaze<u32>>> = None;

    for (global_tsid, tokens) in merged_by_global_tsid {
        if let Some(ref current) = run_shared {
            if global_tsid == run_end.wrapping_add(1)
                && (Arc::ptr_eq(current, &tokens) || current.as_ref() == tokens.as_ref())
            {
                run_end = global_tsid;
                continue;
            }
            map.extend_simple(std::iter::once((
                run_start.unwrap()..=run_end,
                Arc::clone(current),
            )));
        }
        run_start = Some(global_tsid);
        run_end = global_tsid;
        run_shared = Some(tokens);
    }
    if let Some(current) = run_shared {
        map.extend_simple(std::iter::once((
            run_start.unwrap()..=run_end,
            current,
        )));
    }

    finalize_weight_map(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn singleton_weight(token: u32) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(std::iter::once(token..=token)),
        )))
    }

    fn generic_union(inputs: &[DWA]) -> DWA {
        let mut nwa = NWA::new(1, 2);
        let mut body = nwa.body();
        for input in inputs {
            let branch = input.to_nwa();
            body = nwa.union_in_place(&branch, &body);
        }
        nwa.set_start_states(body.start_states);
        determinize(&nwa).expect("test NWA should determinize")
    }

    fn all_words(labels: &[i32], max_len: usize) -> Vec<Vec<i32>> {
        let mut words = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for prefix in frontier {
                for &label in labels {
                    let mut word = prefix.clone();
                    word.push(label);
                    words.push(word.clone());
                    next.push(word);
                }
            }
            frontier = next;
        }
        words
    }

    #[test]
    fn direct_disjoint_union_matches_generic_subset_determinization() {
        let left_weight = singleton_weight(0);
        let mut left = DWA::new(1, 1);
        let left_mid = left.add_state();
        let left_end = left.add_state();
        left.add_transition(left.start_state(), 7, left_mid, left_weight.clone());
        left.add_transition(left_mid, 8, left_end, left_weight.clone());
        left.set_final_weight(left_mid, left_weight.clone());
        left.set_final_weight(left_end, left_weight);

        let right_weight = singleton_weight(1);
        let mut right = DWA::new(1, 1);
        let right_mid = right.add_state();
        let right_end = right.add_state();
        right.add_transition(right.start_state(), 7, right_mid, right_weight.clone());
        right.add_transition(right_mid, 9, right_end, right_weight.clone());
        right.set_final_weight(right_mid, right_weight.clone());
        right.set_final_weight(right_end, right_weight);

        let inputs = vec![left, right];
        let direct = direct_union_disjoint_token_domain_dwas(&inputs);
        let generic = generic_union(&inputs);

        for word in all_words(&[7, 8, 9], 3) {
            assert_eq!(
                direct.eval_word(&word),
                generic.eval_word(&word),
                "word={word:?}"
            );
        }
    }
}
