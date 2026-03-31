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
use crate::automata::weighted::minimize::{minimize, minimize_with_threshold};
use crate::automata::weighted::nwa::NWA;
use crate::compiler::stages::compact::compact_dwa_dimensions_fast;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::Weight;

use super::types::{compile_profile_enabled, debug_profile_enabled};

/// Merge multiple `(InternalIdMap, DWA)` pairs into a single pair.
///
/// 1. Builds a global `InternalIdMap` as the finest common refinement of all
///    input id_maps (for states) and a unified token mapping.
/// 2. Converts each DWA to an NWA, remaps it to the global space, and unions
///    all NWAs together.
/// 3. Determinizes and minimizes the union.
/// 4. Runs dimension compaction.
/// 5. Returns the merged `(InternalIdMap, DWA)`.
pub(crate) fn merge_id_maps_and_terminal_dwas(
    label: &str,
    inputs: Vec<(InternalIdMap, DWA)>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> (InternalIdMap, DWA) {
    assert!(!inputs.is_empty(), "merge_id_maps_and_terminal_dwas called with empty inputs");

    let total_started_at = Instant::now();

    if inputs.len() == 1 {
        let (mut id_map, mut dwa) = inputs.into_iter().next().unwrap();
        let compact_started_at = Instant::now();
        compact_dwa_dimensions_fast(&mut dwa, &mut id_map);
        if compile_profile_enabled() || debug_profile_enabled() {
            eprintln!(
                "[glrmask/profile][merge] label={} inputs=1 build_global_id_map_ms=0.000 remap_union_ms=0.000 determinize_ms=0.000 minimize_ms=0.000 compact_ms={:.3} total_ms={:.3}",
                label,
                compact_started_at.elapsed().as_secs_f64() * 1000.0,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return (id_map, dwa);
    }

    // 1. Build global id_map via composite-key refinement.
    let global_id_map_started_at = Instant::now();
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|(m, _)| m).collect();
    let global_id_map = build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);
    let global_id_map_ms = global_id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    // 2. Convert each DWA → NWA, remap to global space, union.
    let remap_union_started_at = Instant::now();
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();

    for (local_id_map, dwa) in &inputs {
        let mut nwa = dwa.to_nwa();
        let tsid_map = build_local_to_global_tsid_map(local_id_map, &global_id_map);
        let token_map = build_local_to_global_token_map(local_id_map, &global_id_map);
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

    // 3. Determinize + minimize.
    let determinize_started_at = Instant::now();
    let det = determinize(&global_nwa)
        .expect("merge terminal NWA determinization failed");
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;

    let minimize_started_at = Instant::now();
    let mut dwa = if label == "global" {
        minimize_with_threshold(&det, 50)
    } else {
        minimize(&det)
    };
    let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;

    // 4. Compact.
    let mut global = global_id_map;
    let compact_started_at = Instant::now();
    compact_dwa_dimensions_fast(&mut dwa, &mut global);
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;

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

    (global, dwa)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a unified global InternalIdMap from multiple inputs.
///
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
    if weight.is_empty() || weight.is_full() {
        return weight.clone();
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
