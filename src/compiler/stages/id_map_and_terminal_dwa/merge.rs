//! Merge multiple (InternalIdMap, DWA) pairs into one.
//!
//! Handles both overlapping vocabs (e.g., L1 + L2+ from the same partition)
//! and disjoint vocabs (e.g., different character-type partitions) uniformly
//! via composite-key refinement.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;

use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::{minimize, minimize_from_env};
use crate::automata::weighted::nwa::NWA;
use crate::compiler::stages::compact::{compact_from_env, CompactMode};
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::Weight;

use super::types::{TerminalDwaPhaseProfile, compile_profile_enabled, debug_profile_enabled};

#[derive(Debug, Clone)]
pub(crate) struct LocalIdMapTerminalDwa {
    pub(crate) id_map: InternalIdMap,
    pub(crate) dwa: DWA,
    pub(crate) original_to_local_state: Vec<u32>,
    pub(crate) profile: TerminalDwaPhaseProfile,
}

pub(crate) fn identity_original_to_local_state(num_tokenizer_states: usize) -> Vec<u32> {
    (0..num_tokenizer_states as u32).collect()
}

fn merge_phase_profile(
    global_id_map_ms: f64,
    remap_union_ms: f64,
    determinize_ms: f64,
    minimize_ms: f64,
    compact_ms: f64,
) -> TerminalDwaPhaseProfile {
    TerminalDwaPhaseProfile {
        id_map_ms: global_id_map_ms,
        terminal_dwa_ms: remap_union_ms + determinize_ms + minimize_ms,
        compact_ms,
    }
}

pub(crate) fn merge_local_id_maps_and_terminal_dwas(
    label: &str,
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    assert!(!inputs.is_empty(), "merge_local_id_maps_and_terminal_dwas called with empty inputs");

    let total_started_at = Instant::now();

    if inputs.len() == 1 {
        let mut input = inputs.into_iter().next().unwrap();
        input.id_map = expand_local_id_map_to_original_space(&input, num_tokenizer_states);
        let compact_started_at = Instant::now();
        compact_from_env(&mut input.dwa, &mut input.id_map, "GLRMASK_COMPACT_MERGE", CompactMode::Fast, false);
        let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
        if compile_profile_enabled() || debug_profile_enabled() {
            eprintln!(
                "[glrmask/profile][merge] label={} inputs=1 build_global_id_map_ms=0.000 remap_union_ms=0.000 determinize_ms=0.000 minimize_ms=0.000 compact_ms={:.3} total_ms={:.3}",
                label,
                compact_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        input.profile = TerminalDwaPhaseProfile {
            id_map_ms: 0.0,
            terminal_dwa_ms: 0.0,
            compact_ms,
        };
        return input;
    }

    let global_id_map_started_at = Instant::now();
    let input_refs: Vec<&LocalIdMapTerminalDwa> = inputs.iter().collect();
    let global_id_map = build_unified_global_id_map_from_local_inputs(
        &input_refs,
        num_tokenizer_states,
        max_token_id,
    );
    let global_id_map_ms = global_id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let remap_union_started_at = Instant::now();
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();

    for local in &inputs {
        let mut nwa = local.dwa.to_nwa();
        let tsid_map = build_local_to_global_tsid_map_from_local_input(local, &global_id_map);
        let token_map = build_local_to_global_token_map(&local.id_map, &global_id_map);
        remap_nwa_with_maps(
            &mut nwa,
            &tsid_map,
            &token_map,
            global_id_map.num_tsids() as usize,
        );
        global_body = global_nwa.union_in_place(&nwa, &global_body);
    }
    global_nwa.start_states = global_body.start_states;
    let remap_union_ms = remap_union_started_at.elapsed().as_secs_f64() * 1000.0;

    let determinize_started_at = Instant::now();
    let det = determinize(&global_nwa)
        .expect("merge terminal NWA determinization failed");
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;

    let minimize_started_at = Instant::now();
    let mut dwa = minimize_from_env(&det, "GLRMASK_MINIMIZE_MERGE", minimize);
    let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut global = global_id_map;
    let compact_started_at = Instant::now();
    compact_from_env(&mut dwa, &mut global, "GLRMASK_COMPACT_MERGE", CompactMode::Fast, false);
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
    let profile = merge_phase_profile(
        global_id_map_ms,
        remap_union_ms,
        determinize_ms,
        minimize_ms,
        compact_ms,
    );

    if compile_profile_enabled() || debug_profile_enabled() {
        eprintln!(
            "[glrmask/profile][merge] label={} inputs={} build_global_id_map_ms={:.3} remap_union_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} compact_ms={:.3} total_ms={:.3}",
            label,
            inputs.len(),
            global_id_map_ms,
            remap_union_ms,
            determinize_ms,
            minimize_ms,
            compact_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    LocalIdMapTerminalDwa {
        id_map: global,
        dwa,
        original_to_local_state: identity_original_to_local_state(num_tokenizer_states),
        profile,
    }
}

/// Merge multiple `(InternalIdMap, DWA)` pairs into a single pair.
pub(crate) fn merge_id_maps_and_terminal_dwas(
    label: &str,
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    assert!(!inputs.is_empty(), "merge_id_maps_and_terminal_dwas called with empty inputs");

    let total_started_at = Instant::now();

    if inputs.len() == 1 {
        let mut input = inputs.into_iter().next().unwrap();
        let compact_started_at = Instant::now();
        compact_from_env(&mut input.dwa, &mut input.id_map, "GLRMASK_COMPACT_MERGE_GLOBAL", CompactMode::Fast, false);
        let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
        if compile_profile_enabled() || debug_profile_enabled() {
            eprintln!(
                "[glrmask/profile][merge] label={} inputs=1 build_global_id_map_ms=0.000 remap_union_ms=0.000 determinize_ms=0.000 minimize_ms=0.000 compact_ms={:.3} total_ms={:.3}",
                label,
                compact_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        input.profile = TerminalDwaPhaseProfile {
            id_map_ms: 0.0,
            terminal_dwa_ms: 0.0,
            compact_ms,
        };
        return input;
    }

    // 1. Build global id_map via composite-key refinement.
    let global_id_map_started_at = Instant::now();
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    let mut global_id_map = build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);
    let global_id_map_ms = global_id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    // 2. Convert each DWA → NWA, remap to global space, union.
    let remap_union_started_at = Instant::now();
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();

    for input in &inputs {
        let mut nwa = input.dwa.to_nwa();
        let tsid_map = build_local_to_global_tsid_map(&input.id_map, &global_id_map);
        let token_map = build_local_to_global_token_map(&input.id_map, &global_id_map);
        remap_nwa_with_maps(
            &mut nwa,
            &tsid_map,
            &token_map,
            global_id_map.num_tsids() as usize,
        );
        global_body = global_nwa.union_in_place(&nwa, &global_body);
    }
    global_nwa.start_states = global_body.start_states;
    let remap_union_ms = remap_union_started_at.elapsed().as_secs_f64() * 1000.0;

    // 2b. Optimize ID ordering to minimize weight range fragmentation.
    let reorder_started_at = Instant::now();
    optimize_nwa_id_ordering(&mut global_nwa, &mut global_id_map);
    let reorder_ms = reorder_started_at.elapsed().as_secs_f64() * 1000.0;

    // 3. Determinize + minimize.
    let determinize_started_at = Instant::now();
    let det = determinize(&global_nwa)
        .expect("merge terminal NWA determinization failed");
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;

    let det_states = det.states.len();
    let minimize_started_at = Instant::now();
    let mut dwa = if label == "global" {
        minimize_from_env(&det, "GLRMASK_MINIMIZE_MERGE_GLOBAL", minimize)
    } else {
        minimize_from_env(&det, "GLRMASK_MINIMIZE_MERGE", minimize)
    };
    let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;
    let min_states = dwa.states.len();

    // 4. Compact.
    let mut global = global_id_map;
    let compact_started_at = Instant::now();
    compact_from_env(&mut dwa, &mut global, "GLRMASK_COMPACT_MERGE_GLOBAL", CompactMode::Fast, false);
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
    let profile = merge_phase_profile(
        global_id_map_ms,
        remap_union_ms,
        determinize_ms,
        minimize_ms,
        compact_ms,
    );

    if compile_profile_enabled() || debug_profile_enabled() {
        eprintln!(
            "[glrmask/profile][merge] label={} inputs={} build_global_id_map_ms={:.3} remap_union_ms={:.3} reorder_ms={:.3} determinize_ms={:.3} det_states={} minimize_ms={:.3} min_states={} compact_ms={:.3} total_ms={:.3}",
            label,
            inputs.len(),
            global_id_map_ms,
            remap_union_ms,
            reorder_ms,
            determinize_ms,
            det_states,
            minimize_ms,
            min_states,
            compact_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    LocalIdMapTerminalDwa {
        id_map: global,
        dwa,
        original_to_local_state: identity_original_to_local_state(num_tokenizer_states),
        profile,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a unified global InternalIdMap from multiple inputs.
///
/// Re-order class IDs so that composite keys are in lexicographic order.
/// This groups classes by partition: for disjoint partitions, all classes from
/// the same partition get consecutive global IDs, minimizing range fragmentation
/// in weights during determinize/minimize.
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

/// Like `reorder_classes`, but skips elements in `o2i` that equal `sentinel`.
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

/// State refinement: composite key `(class_in_0, class_in_1, ...)`.
/// Token refinement: composite key `(class_in_0_or_MAX, class_in_1_or_MAX, ...)`.
///
/// This handles both overlapping and disjoint vocabs correctly:
/// - Overlapping tokens get a compound key that refines both inputs.
/// - Disjoint tokens get a compound key with `u32::MAX` for absent inputs.
fn build_unified_global_id_map(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> InternalIdMap {
    // --- State refinement ---
    let mut composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut state_o2i = vec![0u32; num_tokenizer_states];
    let mut state_i2o: Vec<Vec<u32>> = Vec::new();
    let mut state_reps: Vec<u32> = Vec::new();

    for state in 0..num_tokenizer_states {
        let composite: Vec<u32> = inputs
            .iter()
            .map(|m| m.tokenizer_states.original_to_internal[state])
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

    // Re-order state classes so composite keys are in lexicographic order.
    // This groups classes by partition, keeping related TSIDs contiguous and
    // minimizing range fragmentation in weights during determinize.
    reorder_classes(composite_to_class, &mut state_o2i, &mut state_i2o, &mut state_reps);

    // --- Token refinement ---
    let mut token_composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut token_o2i = vec![u32::MAX; max_token_id as usize + 1];
    let mut token_i2o: Vec<Vec<u32>> = Vec::new();
    let mut token_reps: Vec<u32> = Vec::new();

    for token_id in 0..=max_token_id {
        let composite: Vec<u32> = inputs
            .iter()
            .map(|m| {
                m.vocab_tokens
                    .original_to_internal
                    .get(token_id as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        if composite.iter().all(|&c| c == u32::MAX) {
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

    // Re-order token classes the same way. For disjoint partitions, this ensures
    // all tokens from the same partition get consecutive global IDs.
    reorder_classes_with_sentinel(
        token_composite_to_class,
        &mut token_o2i,
        &mut token_i2o,
        &mut token_reps,
        u32::MAX,
    );

    InternalIdMap {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: state_o2i,
            internal_to_originals: state_i2o,
            representative_original_ids: state_reps,
        },
        vocab_tokens: ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
    }
}

pub(crate) fn expand_local_id_map_to_original_space(
    input: &LocalIdMapTerminalDwa,
    num_tokenizer_states: usize,
) -> InternalIdMap {
    let num_internal = input.id_map.num_tsids() as usize;
    let mut state_o2i = vec![0u32; num_tokenizer_states];
    let mut state_i2o = vec![Vec::new(); num_internal.max(1)];
    let mut state_reps = vec![0u32; num_internal.max(1)];
    let mut seen_rep = vec![false; num_internal.max(1)];

    for original_state in 0..num_tokenizer_states {
        let local_state = input.original_to_local_state[original_state];
        let internal = if local_state == u32::MAX {
            0
        } else {
            input.id_map.tokenizer_states.original_to_internal[local_state as usize]
        } as usize;
        state_o2i[original_state] = internal as u32;
        state_i2o[internal].push(original_state as u32);
        if !seen_rep[internal] {
            state_reps[internal] = original_state as u32;
            seen_rep[internal] = true;
        }
    }

    InternalIdMap {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: state_o2i,
            internal_to_originals: state_i2o,
            representative_original_ids: state_reps,
        },
        vocab_tokens: input.id_map.vocab_tokens.clone(),
    }
}

fn build_unified_global_id_map_from_local_inputs(
    inputs: &[&LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> InternalIdMap {
    let mut composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut state_o2i = vec![0u32; num_tokenizer_states];
    let mut state_i2o: Vec<Vec<u32>> = Vec::new();
    let mut state_reps: Vec<u32> = Vec::new();

    for state in 0..num_tokenizer_states {
        let composite: Vec<u32> = inputs
            .iter()
            .map(|input| {
                let local_state = input.original_to_local_state[state];
                if local_state == u32::MAX {
                    u32::MAX
                } else {
                    input.id_map.tokenizer_states.original_to_internal[local_state as usize]
                }
            })
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

    let mut token_composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut token_o2i = vec![u32::MAX; max_token_id as usize + 1];
    let mut token_i2o: Vec<Vec<u32>> = Vec::new();
    let mut token_reps: Vec<u32> = Vec::new();

    for token_id in 0..=max_token_id {
        let composite: Vec<u32> = inputs
            .iter()
            .map(|input| {
                input
                    .id_map
                    .vocab_tokens
                    .original_to_internal
                    .get(token_id as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        if composite.iter().all(|&c| c == u32::MAX) {
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

    InternalIdMap {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: state_o2i,
            internal_to_originals: state_i2o,
            representative_original_ids: state_reps,
        },
        vocab_tokens: ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
    }
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

fn build_local_to_global_tsid_map_from_local_input(
    local: &LocalIdMapTerminalDwa,
    global_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local.id_map.num_tsids() as usize;
    let mut local_to_global = vec![BTreeSet::new(); num_local];

    for (state, &local_state) in local.original_to_local_state.iter().enumerate() {
        if local_state == u32::MAX {
            continue;
        }
        let local_tsid = local.id_map.tokenizer_states.original_to_internal[local_state as usize];
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

    for state in &mut nwa.states {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = remap_weight_cached(
                final_weight,
                local_to_global_tsids,
                local_to_global_tokens,
                global_tsid_count,
                &mut weight_cache,
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
    );
    cache.insert(ptr, remapped.clone());
    remapped
}

fn remap_weight_general(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global_tsid_count: usize,
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

    // Cache remapped token sets by Arc pointer.
    let mut token_cache = HashMap::<usize, Arc<RangeSetBlaze<u32>>>::new();

    let mut tokens_by_global_tsid: Vec<Option<Arc<RangeSetBlaze<u32>>>> =
        vec![None; global_tsid_count];
    let mut any_set = false;

    for (start, end, tokens) in entries {
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_cache
            .entry(token_key)
            .or_insert_with(|| {
                let mut result = RangeSetBlaze::new();
                for local_token in tokens.iter() {
                    if let Some(globals) = local_to_global_tokens.get(local_token as usize) {
                        for &g in globals {
                            result.insert(g);
                        }
                    }
                }
                Arc::new(result)
            })
            .clone();

        for local_tsid in start..=end {
            let Some(global_tsids) = local_to_global_tsids.get(local_tsid as usize) else {
                continue;
            };
            for &global_tsid in global_tsids {
                let idx = global_tsid as usize;
                if idx >= global_tsid_count {
                    continue;
                }
                match &mut tokens_by_global_tsid[idx] {
                    Some(existing) => {
                        let merged = existing.as_ref() | mapped_tokens.as_ref();
                        *existing = shared_rangeset(merged);
                    }
                    slot @ None => {
                        *slot = Some(Arc::clone(&mapped_tokens));
                    }
                }
                any_set = true;
            }
        }
    }

    if !any_set {
        return Weight::empty();
    }

    // Build WeightMap by scanning dense Vec for contiguous runs.
    use range_set_blaze::RangeMapBlaze;

    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start: Option<u32> = None;
    let mut run_end: u32 = 0;
    let mut run_shared: Option<Arc<RangeSetBlaze<u32>>> = None;

    for (idx, slot) in tokens_by_global_tsid.iter().enumerate() {
        let global_tsid = idx as u32;
        if let Some(tokens) = slot {
            if let Some(ref current) = run_shared {
                if Arc::ptr_eq(current, tokens) || current.as_ref() == tokens.as_ref() {
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
            run_shared = Some(Arc::clone(tokens));
        } else if let Some(ref current) = run_shared {
            map.extend_simple(std::iter::once((
                run_start.unwrap()..=run_end,
                Arc::clone(current),
            )));
            run_start = None;
            run_shared = None;
        }
    }
    if let Some(current) = run_shared {
        map.extend_simple(std::iter::once((
            run_start.unwrap()..=run_end,
            current,
        )));
    }

    finalize_weight_map(map)
}

/// Optimize NWA ID ordering for both TSID and token dimensions.
///
/// Computes "signature" for each ID based on which NWA weights cover it,
/// then sorts IDs by signature so IDs with identical coverage are adjacent.
/// This minimizes range fragmentation in weights during determinize/minimize.
pub(crate) fn optimize_nwa_id_ordering(
    nwa: &mut NWA,
    id_map: &mut InternalIdMap,
) {
    let num_tsids = id_map.num_tsids() as usize;
    let num_tokens = (id_map.max_internal_token_id() + 1) as usize;
    if num_tsids <= 1 && num_tokens <= 1 {
        return;
    }

    // 1. Collect all unique weights and token sets from the NWA.
    let mut weight_ptrs: HashMap<usize, usize> = HashMap::new();
    let mut weight_entries_list: Vec<Vec<(u32, u32, Arc<RangeSetBlaze<u32>>)>> = Vec::new();
    let mut token_set_ptrs: HashMap<usize, usize> = HashMap::new();
    let mut token_set_list: Vec<Arc<RangeSetBlaze<u32>>> = Vec::new();

    let mut register_weight = |w: &Weight| {
        let ptr = Arc::as_ptr(&w.0) as usize;
        if weight_ptrs.contains_key(&ptr) {
            return;
        }
        let idx = weight_entries_list.len();
        weight_ptrs.insert(ptr, idx);
        if let Some(entries) = w.compact_entries() {
            for &(_, _, ref ts) in &entries {
                let ts_ptr = Arc::as_ptr(ts) as usize;
                token_set_ptrs.entry(ts_ptr).or_insert_with(|| {
                    let i = token_set_list.len();
                    token_set_list.push(Arc::clone(ts));
                    i
                });
            }
            weight_entries_list.push(entries);
        } else {
            weight_entries_list.push(Vec::new());
        }
    };

    for state in &nwa.states {
        if let Some(ref w) = state.final_weight {
            register_weight(w);
        }
        for targets in state.transitions.values() {
            for (_, w) in targets {
                register_weight(w);
            }
        }
        for (_, w) in &state.epsilons {
            register_weight(w);
        }
    }

    // 2. Compute TSID signatures: for each TSID, which weight indices cover it.
    let tsid_perm = if num_tsids > 1 {
        let mut tsid_sigs: Vec<Vec<usize>> = vec![Vec::new(); num_tsids];
        for (idx, entries) in weight_entries_list.iter().enumerate() {
            for &(start, end, _) in entries {
                for tsid in start..=end {
                    if (tsid as usize) < num_tsids {
                        tsid_sigs[tsid as usize].push(idx);
                    }
                }
            }
        }
        compute_permutation_from_signatures(tsid_sigs)
    } else {
        vec![0u32; num_tsids]
    };

    // 3. Compute token signatures: for each token, which token set indices cover it.
    let token_perm = if num_tokens > 1 {
        let mut token_sigs: Vec<Vec<usize>> = vec![Vec::new(); num_tokens];
        for (idx, ts) in token_set_list.iter().enumerate() {
            for token_id in ts.iter() {
                if (token_id as usize) < num_tokens {
                    token_sigs[token_id as usize].push(idx);
                }
            }
        }
        compute_permutation_from_signatures(token_sigs)
    } else {
        vec![0u32; num_tokens]
    };

    // 4. Apply permutations to NWA weights.
    let mut weight_cache: HashMap<usize, Weight> = HashMap::new();
    for state in &mut nwa.states {
        if let Some(ref w) = state.final_weight {
            let remapped = remap_weight_with_permutation(w, &tsid_perm, &token_perm, &mut weight_cache);
            if remapped.is_empty() {
                state.final_weight = None;
            } else {
                state.final_weight = Some(remapped);
            }
        }
        for targets in state.transitions.values_mut() {
            for (_, w) in targets.iter_mut() {
                *w = remap_weight_with_permutation(w, &tsid_perm, &token_perm, &mut weight_cache);
            }
            targets.retain(|(_, w)| !w.is_empty());
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
        for (_, w) in state.epsilons.iter_mut() {
            *w = remap_weight_with_permutation(w, &tsid_perm, &token_perm, &mut weight_cache);
        }
        state.epsilons.retain(|(_, w)| !w.is_empty());
    }

    // 5. Apply permutations to id_map.
    apply_permutation_to_many_to_one(&mut id_map.tokenizer_states, &tsid_perm);
    apply_permutation_to_many_to_one_with_sentinel(&mut id_map.vocab_tokens, &token_perm, u32::MAX);

    if compile_profile_enabled() || debug_profile_enabled() {
        let old_tsid_is_identity = tsid_perm.iter().enumerate().all(|(i, &v)| v == i as u32);
        let old_token_is_identity = token_perm.iter().enumerate().all(|(i, &v)| v == i as u32);
        eprintln!(
            "[glrmask/debug][reorder_ids] num_tsids={} num_tokens={} tsid_changed={} token_changed={}",
            num_tsids,
            num_tokens,
            !old_tsid_is_identity,
            !old_token_is_identity,
        );
    }
}

/// Compute permutation from signatures: sort IDs by their signature vectors.
/// Returns perm[old_id] = new_id.
fn compute_permutation_from_signatures(signatures: Vec<Vec<usize>>) -> Vec<u32> {
    let n = signatures.len();
    // Sort indices by their signatures
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| signatures[a].cmp(&signatures[b]));
    // Build old→new permutation
    let mut perm = vec![0u32; n];
    for (new_pos, &old_id) in order.iter().enumerate() {
        perm[old_id] = new_pos as u32;
    }
    perm
}

/// Remap a Weight using TSID and token permutations.
fn remap_weight_with_permutation(
    weight: &Weight,
    tsid_perm: &[u32],
    token_perm: &[u32],
    cache: &mut HashMap<usize, Weight>,
) -> Weight {
    let ptr = Arc::as_ptr(&weight.0) as usize;
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }

    let result = if weight.is_empty() || weight.is_full() {
        weight.clone()
    } else if let Some(entries) = weight.compact_entries() {
        // Cache remapped token sets.
        let mut token_cache: HashMap<usize, Arc<RangeSetBlaze<u32>>> = HashMap::new();

        // Collect (new_tsid, remapped_token_set) pairs.
        let mut pairs: Vec<(u32, Arc<RangeSetBlaze<u32>>)> = Vec::new();
        for (start, end, tokens) in entries {
            let ts_ptr = Arc::as_ptr(&tokens) as usize;
            let remapped_tokens = token_cache
                .entry(ts_ptr)
                .or_insert_with(|| {
                    let mut new_set = RangeSetBlaze::new();
                    for old_token in tokens.iter() {
                        if let Some(&new_token) = token_perm.get(old_token as usize) {
                            new_set.insert(new_token);
                        }
                    }
                    Arc::new(new_set)
                })
                .clone();

            for old_tsid in start..=end {
                if let Some(&new_tsid) = tsid_perm.get(old_tsid as usize) {
                    pairs.push((new_tsid, remapped_tokens.clone()));
                }
            }
        }

        // Sort by new_tsid and build Weight via CompactRangeBuilder pattern.
        pairs.sort_by_key(|p| p.0);

        // Build using from_per_tsid_shared which merges adjacent TSIDs with same token set.
        Weight::from_per_tsid_shared(
            pairs.into_iter().map(|(tsid, ts)| (tsid, ts)),
        )
    } else {
        weight.clone()
    };

    cache.insert(ptr, result.clone());
    result
}

/// Apply a permutation to a ManyToOneIdMap.
/// perm[old_internal] = new_internal
fn apply_permutation_to_many_to_one(
    map: &mut ManyToOneIdMap,
    perm: &[u32],
) {
    let n = map.internal_to_originals.len();
    // Update original_to_internal
    for val in map.original_to_internal.iter_mut() {
        *val = perm[*val as usize];
    }
    // Rebuild internal_to_originals and representative_original_ids
    let mut new_i2o = vec![Vec::new(); n];
    let mut new_reps = vec![0u32; n];
    for old in 0..n {
        let new = perm[old] as usize;
        new_i2o[new] = std::mem::take(&mut map.internal_to_originals[old]);
        new_reps[new] = map.representative_original_ids[old];
    }
    map.internal_to_originals = new_i2o;
    map.representative_original_ids = new_reps;
}

/// Apply a permutation to a ManyToOneIdMap, skipping sentinel values.
fn apply_permutation_to_many_to_one_with_sentinel(
    map: &mut ManyToOneIdMap,
    perm: &[u32],
    sentinel: u32,
) {
    let n = map.internal_to_originals.len();
    for val in map.original_to_internal.iter_mut() {
        if *val != sentinel {
            *val = perm[*val as usize];
        }
    }
    let mut new_i2o = vec![Vec::new(); n];
    let mut new_reps = vec![0u32; n];
    for old in 0..n {
        let new = perm[old] as usize;
        new_i2o[new] = std::mem::take(&mut map.internal_to_originals[old]);
        new_reps[new] = map.representative_original_ids[old];
    }
    map.internal_to_originals = new_i2o;
    map.representative_original_ids = new_reps;
}

#[cfg(test)]
mod tests {
    use super::remap_weight_general;
    use crate::ds::weight::Weight;
    use range_set_blaze::RangeSetBlaze;

    #[test]
    fn remap_full_weight_is_bounded_to_mapped_space() {
        let remapped = remap_weight_general(
            &Weight::all(),
            &[vec![1, 3], vec![2], vec![]],
            &[vec![10, 11], vec![12], vec![]],
            5,
        );

        let expected_tokens: RangeSetBlaze<u32> = [10..=12].into_iter().collect();
        let expected = Weight::from_per_tsid_token_sets([
            (1, expected_tokens.clone()),
            (2, expected_tokens.clone()),
            (3, expected_tokens),
        ]);

        assert_eq!(remapped, expected);
    }

    #[test]
    fn remap_full_weight_without_mappings_is_empty() {
        let remapped = remap_weight_general(&Weight::all(), &[vec![]], &[vec![]], 4);
        assert!(remapped.is_empty());
    }
}
