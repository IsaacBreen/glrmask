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
use crate::automata::weighted::minimize::minimize;
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
    let (global_id_map, _) =
        build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);

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

    let det = determinize(&global_nwa)
        .expect("merge terminal NWA determinization failed");
    let dwa = minimize(&det);

    LocalIdMapTerminalDwa {
        id_map: global_id_map,
        dwa,
        profile: TerminalDwaPhaseProfile::default(),
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

    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    let (global_id_map, direct_local_to_global_token_maps) =
        build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);

    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();

    for (input_idx, input) in inputs.iter().enumerate() {
        let mut nwa = input.dwa.to_nwa();

        let tsid_map = build_local_to_global_tsid_map(&input.id_map, &global_id_map);

        let token_map = direct_local_to_global_token_maps
            .as_ref()
            .and_then(|maps| maps.get(input_idx))
            .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
            .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, &global_id_map));

        remap_nwa_with_maps(
            &mut nwa,
            &tsid_map,
            &token_map,
            global_id_map.num_tsids() as usize,
        );

        global_body = global_nwa.union_in_place(&nwa, &global_body);
    }
    global_nwa.set_start_states(global_body.start_states);

    let det = determinize(&global_nwa)
        .expect("merge terminal NWA determinization failed");

    let dwa = if minimize_merged_terminal_dwa_enabled() {
        minimize(&det)
    } else {
        det.clone()
    };
    let mut mapped_dwa = MappedArtifact::new(dwa, global_id_map);
    let before_compact_stats = mapped_dwa.artifact().stats();
    let before_compact_range_counts = mapped_dwa.interned_range_counts();
    let before_num_tsids = mapped_dwa.id_map().num_tsids();
    let before_num_tokens = mapped_dwa.id_map().num_internal_tokens();
    let compact_enabled = compact_merged_terminal_dwa_enabled();
    let (compact_report, compact_ms) = if compact_enabled {
        let compact_started_at = Instant::now();
        let compact_report = mapped_dwa.compact_dimensions_fast_with_stats();
        let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
        (Some(compact_report), compact_ms)
    } else {
        (None, 0.0)
    };
    let after_compact_stats = mapped_dwa.artifact().stats();
    let after_compact_range_counts = mapped_dwa.interned_range_counts();
    let (dwa, id_map) = mapped_dwa.into_parts();

    if compile_profile_enabled() {
        let profile_stats = compact_report.and_then(|report| report.profile_stats);
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

    for state in  nwa.states_mut() {
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
