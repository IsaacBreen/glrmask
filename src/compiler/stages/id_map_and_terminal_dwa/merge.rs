//! Merge multiple (InternalIdMap, DWA) pairs into one.
//!
//! Handles both overlapping vocabs (e.g., L1 + L2+ from the same partition)
//! and disjoint vocabs (e.g., different character-type partitions) uniformly
//! via composite-key refinement.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::equivalence::find_difference;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize_owned;
use crate::automata::weighted::minimize_token_deterministic_nwa::{
    minimize_token_deterministic_nwa_owned, quotient_disjoint_source_nwa_owned,
};
use crate::automata::weighted::nwa::NWA;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::mapped_artifact::MappedArtifact;
use crate::ds::weight::Weight;

use super::types::{LocalIdMapTerminalDwa, TerminalDwaPhaseProfile, compile_profile_enabled};

type RemapCache<T> = FxHashMap<usize, T>;

fn merged_terminal_dwa_phase_enabled(variable: &str, default: bool) -> bool {
    std::env::var(variable)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(default)
}

fn minimize_merged_terminal_dwa_enabled(default: bool) -> bool {
    merged_terminal_dwa_phase_enabled("GLRMASK_MINIMIZE_MERGED_TERMINAL_DWA", default)
}

fn compact_merged_terminal_dwa_enabled(default: bool) -> bool {
    merged_terminal_dwa_phase_enabled("GLRMASK_COMPACT_MERGED_TERMINAL_DWA", default)
}

fn assert_direct_global_terminal_dwa_merge_equivalence_enabled() -> bool {
    std::env::var_os("GLRMASK_ASSERT_DIRECT_GLOBAL_TERMINAL_DWA_MERGE_EQUIVALENCE").is_some()
}

fn minimize_token_deterministic_terminal_nwa_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENTAL_MINIMIZE_TOKEN_DETERMINISTIC_TERMINAL_NWA").is_some()
}

fn quotient_token_deterministic_terminal_nwa_by_source_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENTAL_QUOTIENT_TOKEN_NWA_BY_SOURCE").is_some()
}

fn refine_token_deterministic_terminal_nwa_after_source_quotient_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENTAL_REFINE_TOKEN_NWA_AFTER_SOURCE_QUOTIENT").is_some()
}

fn fast_disjoint_terminal_nwa_id_map_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENTAL_FAST_DISJOINT_TERMINAL_NWA_ID_MAP").is_some()
}

fn primary_token_nwa_tsid_source() -> Option<usize> {
    std::env::var("GLRMASK_EXPERIMENTAL_ORDER_TOKEN_NWA_TSID_BY_SOURCE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

fn select_primary_token_nwa_tsid_source(inputs: &[LocalIdMapTerminalDwa]) -> Option<usize> {
    primary_token_nwa_tsid_source().or_else(|| {
        inputs
            .iter()
            .enumerate()
            .max_by_key(|(_, input)| input.id_map.num_tsids())
            .map(|(index, _)| index)
    })
}

fn token_nwa_tsid_source_order(source_count: usize) -> Option<Vec<usize>> {
    let value = std::env::var("GLRMASK_EXPERIMENTAL_TOKEN_NWA_TSID_ORDER").ok()?;
    let order: Vec<usize> = value
        .split(',')
        .map(|part| part.trim().parse::<usize>().ok())
        .collect::<Option<_>>()?;
    let mut seen = vec![false; source_count];
    if order.len() != source_count
        || order.iter().any(|&source| source >= source_count || std::mem::replace(&mut seen[source], true))
    {
        return None;
    }
    Some(order)
}

#[derive(Default)]
struct TokenNwaRemapDetail {
    unique_weights: usize,
    full_weights: usize,
    compact_entries: usize,
    local_tsid_visits: usize,
    global_tsid_expansions: usize,
    max_local_entry_span: usize,
    max_global_entry_expansion: usize,
}

fn profile_token_nwa_remap_detail(
    dwa: &DWA,
    local_to_global_tsids: &[Vec<u32>],
) -> TokenNwaRemapDetail {
    let mut detail = TokenNwaRemapDetail::default();
    let mut seen = FxHashSet::<usize>::default();
    let mut inspect = |weight: &Weight| {
        if weight.is_empty() || !seen.insert(weight.ptr_key()) {
            return;
        }
        detail.unique_weights += 1;
        if weight.is_full() {
            detail.full_weights += 1;
            detail.global_tsid_expansions += local_to_global_tsids.iter().map(Vec::len).sum::<usize>();
            return;
        }
        let Some(entries) = weight.compact_entries() else {
            return;
        };
        detail.compact_entries += entries.len();
        for (start, end, _) in entries {
            let span = end.saturating_sub(start) as usize + 1;
            detail.local_tsid_visits += span;
            detail.max_local_entry_span = detail.max_local_entry_span.max(span);
            let mut entry_expansion = 0usize;
            for local_tsid in start..=end {
                let expansion = local_to_global_tsids
                    .get(local_tsid as usize)
                    .map_or(0, Vec::len);
                detail.global_tsid_expansions += expansion;
                entry_expansion += expansion;
            }
            detail.max_global_entry_expansion = detail.max_global_entry_expansion.max(entry_expansion);
        }
    };
    for state in dwa.states() {
        if let Some(final_weight) = &state.final_weight {
            inspect(final_weight);
        }
        for (_, weight) in state.transitions.values() {
            inspect(weight);
        }
    }
    detail
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

/// Token-class remapping used while merging id-map-local weights.
///
/// Disjoint global vocabularies are laid out as contiguous source blocks, so
/// their token mapping is an affine shift rather than a per-token relation.
enum TokenRemap<'a> {
    Explicit(&'a [Vec<u32>]),
    Offset { offset: u32, local_token_count: u32 },
}

impl TokenRemap<'_> {
    fn map_tokens(
        &self,
        token_key: usize,
        tokens: &RangeSetBlaze<u32>,
        token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
    ) -> Arc<RangeSetBlaze<u32>> {
        if let Some(existing) = token_cache.get(&token_key) {
            return Arc::clone(existing);
        }
        let mapped = match self {
            Self::Offset { offset, .. } => RangeSetBlaze::from_iter(tokens.ranges().map(|range| {
                let start = (*range.start())
                    .checked_add(*offset)
                    .expect("global token offset overflow");
                let end = (*range.end())
                    .checked_add(*offset)
                    .expect("global token offset overflow");
                start..=end
            })),
            Self::Explicit(local_to_global_tokens) => {
                let mut result = RangeSetBlaze::new();
                for local_token in tokens.iter() {
                    if let Some(globals) = local_to_global_tokens.get(local_token as usize) {
                        for &global_token in globals {
                            result.insert(global_token);
                        }
                    }
                }
                result
            }
        };
        let mapped = Arc::new(mapped);
        token_cache.insert(token_key, Arc::clone(&mapped));
        mapped
    }

    fn all_global_tokens(&self) -> RangeSetBlaze<u32> {
        match self {
            Self::Offset { offset, local_token_count } => {
                if *local_token_count == 0 {
                    RangeSetBlaze::new()
                } else {
                    RangeSetBlaze::from_iter(std::iter::once(
                        *offset..=offset
                            .checked_add(*local_token_count - 1)
                            .expect("global token offset overflow"),
                    ))
                }
            }
            Self::Explicit(local_to_global_tokens) => {
                let mut all_global_tokens = RangeSetBlaze::new();
                for globals in *local_to_global_tokens {
                    for &global_token in globals {
                        all_global_tokens.insert(global_token);
                    }
                }
                all_global_tokens
            }
        }
    }
}

fn build_local_to_global_tsid_runs(
    local_to_global_tsids: &[Vec<u32>],
) -> Vec<SmallVec<[(u32, u32); 1]>> {
    local_to_global_tsids
        .iter()
        .map(|globals| {
            let mut runs = SmallVec::<[(u32, u32); 1]>::new();
            for &global_tsid in globals {
                if let Some((_, end)) = runs.last_mut() {
                    if global_tsid == end.saturating_add(1) {
                        *end = global_tsid;
                        continue;
                    }
                }
                runs.push((global_tsid, global_tsid));
            }
            runs
        })
        .collect()
}

/// Remap a deterministic weighted automaton directly, preserving the same
/// branch-local token domain handling as `remap_nwa_with_maps`.
fn remap_dwa_with_maps(
    dwa: &mut DWA,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global_tsid_count: usize,
) {
    remap_dwa_with_token_remap(
        dwa,
        local_to_global_tsids,
        TokenRemap::Explicit(local_to_global_tokens),
        global_tsid_count,
    );
}

fn remap_dwa_with_token_offset(
    dwa: &mut DWA,
    local_to_global_tsids: &[Vec<u32>],
    token_offset: u32,
    local_token_count: u32,
    global_tsid_count: usize,
) {
    remap_dwa_with_token_remap(
        dwa,
        local_to_global_tsids,
        TokenRemap::Offset {
            offset: token_offset,
            local_token_count,
        },
        global_tsid_count,
    );
}

fn remap_dwa_with_token_remap(
    dwa: &mut DWA,
    local_to_global_tsids: &[Vec<u32>],
    token_remap: TokenRemap<'_>,
    global_tsid_count: usize,
) {
    let profile = compile_profile_enabled();
    let local_to_global_tsid_runs = build_local_to_global_tsid_runs(local_to_global_tsids);
    let mut weight_cache = RemapCache::<Weight>::default();
    let mut token_cache = RemapCache::<Arc<RangeSetBlaze<u32>>>::default();
    let mut weights_seen = 0usize;
    let input_states = dwa.num_states();
    let input_transitions = dwa.num_transitions();

    for state in dwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            weights_seen += 1;
            *final_weight = remap_weight_cached_with_tsid_runs(
                final_weight,
                local_to_global_tsids,
                &local_to_global_tsid_runs,
                &token_remap,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
            if final_weight.is_empty() {
                state.final_weight = None;
            }
        }

        for (_, weight) in state.transitions.values_mut() {
            weights_seen += 1;
            *weight = remap_weight_cached_with_tsid_runs(
                weight,
                local_to_global_tsids,
                &local_to_global_tsid_runs,
                &token_remap,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
        }
        state.transitions.retain(|_, (_, weight)| !weight.is_empty());
    }
    if profile {
        eprintln!(
            "[glrmask/profile][terminal_weight_remap] states={} transitions={} weights_seen={} unique_weights={} weight_cache_hits={} unique_token_sets={}",
            input_states,
            input_transitions,
            weights_seen,
            weight_cache.len(),
            weights_seen.saturating_sub(weight_cache.len()),
            token_cache.len(),
        );
    }
}


fn contiguous_tsid_blocks(local_to_global_tsids: &[Vec<u32>]) -> Option<Vec<(u32, u32)>> {
    let mut blocks = Vec::with_capacity(local_to_global_tsids.len());
    let mut expected_start = 0u32;
    for globals in local_to_global_tsids {
        let start = *globals.first()?;
        let end = *globals.last()?;
        if start != expected_start
            || globals
                .iter()
                .enumerate()
                .any(|(offset, &global)| start.checked_add(offset as u32) != Some(global))
        {
            return None;
        }
        expected_start = end.checked_add(1)?;
        blocks.push((start, end));
    }
    Some(blocks)
}

fn remap_dwa_with_contiguous_tsid_blocks_offset(
    dwa: &mut DWA,
    blocks: &[(u32, u32)],
    token_offset: u32,
    local_token_count: u32,
) {
    let token_remap = TokenRemap::Offset {
        offset: token_offset,
        local_token_count,
    };
    let mut weight_cache = RemapCache::<Weight>::default();
    let mut token_cache = RemapCache::<Arc<RangeSetBlaze<u32>>>::default();
    for state in dwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = remap_weight_with_contiguous_tsid_blocks_cached(
                final_weight,
                blocks,
                &token_remap,
                &mut weight_cache,
                &mut token_cache,
            );
            if final_weight.is_empty() {
                state.final_weight = None;
            }
        }
        for (_, weight) in state.transitions.values_mut() {
            *weight = remap_weight_with_contiguous_tsid_blocks_cached(
                weight,
                blocks,
                &token_remap,
                &mut weight_cache,
                &mut token_cache,
            );
        }
        state.transitions.retain(|_, (_, weight)| !weight.is_empty());
    }
}

fn remap_weight_with_contiguous_tsid_blocks_cached(
    weight: &Weight,
    blocks: &[(u32, u32)],
    token_remap: &TokenRemap<'_>,
    cache: &mut RemapCache<Weight>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    let key = Arc::as_ptr(&weight.0) as usize;
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }
    let remapped = remap_weight_with_contiguous_tsid_blocks(weight, blocks, token_remap, token_cache);
    cache.insert(key, remapped.clone());
    remapped
}

fn remap_weight_with_contiguous_tsid_blocks(
    weight: &Weight,
    blocks: &[(u32, u32)],
    token_remap: &TokenRemap<'_>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    use crate::ds::weight::finalize_weight_map;
    use range_set_blaze::RangeMapBlaze;

    if weight.is_empty() || blocks.is_empty() {
        return Weight::empty();
    }
    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    if weight.is_full() {
        let tokens = Arc::new(token_remap.all_global_tokens());
        if tokens.is_empty() {
            return Weight::empty();
        }
        map.extend_simple(std::iter::once((blocks[0].0..=blocks.last().unwrap().1, tokens)));
        return finalize_weight_map(map);
    }
    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };
    for (start, end, tokens) in entries {
        let Some((global_start, _)) = blocks.get(start as usize).copied() else {
            continue;
        };
        let Some((_, global_end)) = blocks.get(end as usize).copied() else {
            continue;
        };
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_remap.map_tokens(token_key, tokens.as_ref(), token_cache);
        if !mapped_tokens.is_empty() {
            map.extend_simple(std::iter::once((global_start..=global_end, mapped_tokens)));
        }
    }
    finalize_weight_map(map)
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

fn contiguous_token_offset(local_to_global: &[u32]) -> Option<u32> {
    let offset = *local_to_global.first()?;
    if offset == u32::MAX {
        return None;
    }
    local_to_global
        .iter()
        .enumerate()
        .all(|(local, &global)| {
            global != u32::MAX && offset.checked_add(local as u32) == Some(global)
        })
        .then_some(offset)
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

    let profiling = compile_profile_enabled();
    let mut build_maps_ms = 0.0;
    let mut clone_ms = 0.0;
    let mut remap_ms = 0.0;
    let mut remapped = Vec::with_capacity(inputs.len());
    for (input_index, input) in inputs.iter().enumerate() {
        let started_at = profiling.then(Instant::now);
        let tsid_map = build_local_to_global_tsid_map(&input.id_map, global_id_map);
        let direct_token_map = direct_local_to_global_token_maps
            .and_then(|maps| maps.get(input_index));
        let token_offset = direct_token_map.and_then(|map| contiguous_token_offset(map));
        if profiling {
            let tsid_pairs = tsid_map.iter().map(Vec::len).sum::<usize>();
            let tsid_one_to_one = tsid_map.iter().filter(|targets| targets.len() == 1).count();
            let tsid_max_fanout = tsid_map.iter().map(Vec::len).max().unwrap_or(0);
            eprintln!(
                "[glrmask/profile][direct_disjoint_terminal_dwa_maps] input={} local_tsids={} tsid_pairs={} tsid_one_to_one={} tsid_max_fanout={} local_tokens={} token_offset={:?}",
                input_index,
                tsid_map.len(),
                tsid_pairs,
                tsid_one_to_one,
                tsid_max_fanout,
                input.id_map.num_internal_tokens(),
                token_offset,
            );
        }
        if let Some(started_at) = started_at {
            build_maps_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        let mut dwa = input.dwa.clone();
        if let Some(started_at) = started_at {
            clone_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        if let Some(token_offset) = token_offset {
            remap_dwa_with_token_offset(
                &mut dwa,
                &tsid_map,
                token_offset,
                input.id_map.num_internal_tokens(),
                global_id_map.num_tsids() as usize,
            );
        } else {
            let token_map = direct_token_map
                .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
                .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, global_id_map));
            remap_dwa_with_maps(
                &mut dwa,
                &tsid_map,
                &token_map,
                global_id_map.num_tsids() as usize,
            );
        }
        if let Some(started_at) = started_at {
            remap_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        remapped.push(dwa);
    }
    let union_started_at = profiling.then(Instant::now);
    let result = direct_union_disjoint_token_domain_dwas(&remapped);
    if let Some(union_started_at) = union_started_at {
        eprintln!(
            "[glrmask/profile][direct_disjoint_terminal_dwa_union] inputs={} build_maps_ms={:.3} clone_ms={:.3} remap_ms={:.3} union_ms={:.3} states={} transitions={}",
            inputs.len(),
            build_maps_ms,
            clone_ms,
            remap_ms,
            union_started_at.elapsed().as_secs_f64() * 1000.0,
            result.num_states(),
            result.num_transitions(),
        );
    }

    Some(result)
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

/// Build the ordinary global NWA union and determinize it. This is used only
/// as an opt-in semantic oracle for the specialized disjoint-domain merge.
fn generic_global_union_determinization(
    inputs: &[LocalIdMapTerminalDwa],
    global_id_map: &InternalIdMap,
    direct_local_to_global_token_maps: Option<&Vec<Vec<u32>>>,
) -> DWA {
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();
    for (input_index, input) in inputs.iter().enumerate() {
        let mut nwa = input.dwa.to_nwa();
        let tsid_map = build_local_to_global_tsid_map(&input.id_map, global_id_map);
        let token_map = direct_local_to_global_token_maps
            .and_then(|maps| maps.get(input_index))
            .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
            .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, global_id_map));
        remap_nwa_with_maps(
            &mut nwa,
            &tsid_map,
            &token_map,
            global_id_map.num_tsids() as usize,
        );
        global_body = global_nwa.union_in_place(&nwa, &global_body);
    }
    global_nwa.set_start_states(global_body.start_states);
    determinize(&global_nwa).expect("global terminal NWA reference determinization failed")
}

fn prune_unreachable_nwa_with_sources(
    nwa: NWA,
    state_sources: Vec<Option<usize>>,
) -> (NWA, Vec<Option<usize>>) {
    assert_eq!(state_sources.len(), nwa.states().len());
    let mut reachable = vec![false; nwa.states().len()];
    let mut stack = nwa.start_states().to_vec();
    while let Some(state_id) = stack.pop() {
        let index = state_id as usize;
        if index >= reachable.len() || reachable[index] {
            continue;
        }
        reachable[index] = true;
        let state = &nwa.states()[index];
        for branches in state.transitions.values() {
            stack.extend(branches.iter().map(|(target, _)| *target));
        }
        stack.extend(state.epsilons.iter().map(|(target, _)| *target));
    }

    let mut old_to_new = vec![u32::MAX; reachable.len()];
    let mut states = Vec::new();
    let mut sources = Vec::new();
    for (old, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            old_to_new[old] = states.len() as u32;
            states.push(nwa.states()[old].clone());
            sources.push(state_sources[old]);
        }
    }
    for state in &mut states {
        for branches in state.transitions.values_mut() {
            for (target, _) in branches {
                *target = old_to_new[*target as usize];
            }
        }
        for (target, _) in &mut state.epsilons {
            *target = old_to_new[*target as usize];
        }
    }
    let starts = nwa
        .start_states()
        .iter()
        .map(|start| old_to_new[*start as usize])
        .collect();
    (NWA::from_parts(states, starts), sources)
}


/// Build a global token-deterministic NWA when every source owns a disjoint
/// vocabulary domain. Multiple branches for one label are then deterministic
/// with respect to the source token, so parser construction can consume them
/// directly without materializing a product DWA.
pub(crate) fn try_merge_id_maps_and_token_deterministic_nwa(
    inputs: &[LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> Option<(NWA, InternalIdMap, TerminalDwaPhaseProfile)> {
    if inputs.len() < 2 || !inputs_have_disjoint_token_domains(inputs, max_token_id) {
        return None;
    }

    let total_started_at = Instant::now();
    let primary_tsid_source = select_primary_token_nwa_tsid_source(inputs)
        .filter(|source| *source < inputs.len());
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    let id_map_started_at = Instant::now();
    let (global_id_map, direct_local_to_global_token_maps, precomputed_tsid_maps) =
        if fast_disjoint_terminal_nwa_id_map_enabled() {
            let (id_map, direct_maps, tsid_maps) = build_unified_global_id_map_disjoint_fast(
                &id_map_refs,
                num_tokenizer_states,
                max_token_id,
                primary_tsid_source,
            );
            (id_map, Some(direct_maps), Some(tsid_maps))
        } else {
            let (id_map, direct_maps) =
                build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);
            (id_map, direct_maps, None)
        };
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;
    let all_tsid_maps: Vec<Vec<Vec<u32>>> = precomputed_tsid_maps.unwrap_or_else(|| {
        inputs
            .iter()
            .map(|input| build_local_to_global_tsid_map(&input.id_map, &global_id_map))
            .collect()
    });
    let primary_tsid_blocks = primary_tsid_source
        .and_then(|source| contiguous_tsid_blocks(&all_tsid_maps[source]));

    let remap_started_at = Instant::now();
    let profiling = compile_profile_enabled();
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut state_sources = Vec::<Option<usize>>::new();
    let mut combined_start: Option<u32> = None;
    let tsid_map_ms = 0.0;
    let mut token_map_ms = 0.0;
    let mut clone_ms = 0.0;
    let mut remap_ms = 0.0;
    let mut to_nwa_ms = 0.0;
    let mut append_ms = 0.0;
    let mut fuse_start_ms = 0.0;
    for (input_index, input) in inputs.iter().enumerate() {
        let tsid_map = &all_tsid_maps[input_index];
        let remap_detail = std::env::var_os("GLRMASK_PROFILE_TOKEN_NWA_REMAP_DETAIL")
            .is_some()
            .then(|| profile_token_nwa_remap_detail(&input.dwa, tsid_map));
        let started_at = profiling.then(Instant::now);
        let direct_token_map = direct_local_to_global_token_maps
            .as_ref()
            .and_then(|maps| maps.get(input_index));
        let token_offset = direct_token_map.and_then(|map| contiguous_token_offset(map));
        if let Some(started_at) = started_at {
            token_map_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        let mut dwa = input.dwa.clone();
        if let Some(started_at) = started_at {
            clone_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        if primary_tsid_source == Some(input_index) {
            if let (Some(token_offset), Some(blocks)) = (token_offset, primary_tsid_blocks.as_deref()) {
                remap_dwa_with_contiguous_tsid_blocks_offset(
                    &mut dwa,
                    blocks,
                    token_offset,
                    input.id_map.num_internal_tokens(),
                );
            } else if let Some(token_offset) = token_offset {
                remap_dwa_with_token_offset(
                    &mut dwa,
                    tsid_map,
                    token_offset,
                    input.id_map.num_internal_tokens(),
                    global_id_map.num_tsids() as usize,
                );
            } else {
                let token_map = direct_token_map
                    .map(|map| build_direct_local_to_global_token_map(map))
                    .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, &global_id_map));
                remap_dwa_with_maps(
                    &mut dwa,
                    tsid_map,
                    &token_map,
                    global_id_map.num_tsids() as usize,
                );
            }
        } else if let Some(token_offset) = token_offset {
            remap_dwa_with_token_offset(
                &mut dwa,
                tsid_map,
                token_offset,
                input.id_map.num_internal_tokens(),
                global_id_map.num_tsids() as usize,
            );
        } else {
            let token_map = direct_token_map
                .map(|map| build_direct_local_to_global_token_map(map))
                .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, &global_id_map));
            remap_dwa_with_maps(
                &mut dwa,
                tsid_map,
                &token_map,
                global_id_map.num_tsids() as usize,
            );
        }
        if let Some(started_at) = started_at {
            let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            remap_ms += elapsed_ms;
            let tsid_pairs = tsid_map.iter().map(Vec::len).sum::<usize>();
            let tsid_max_fanout = tsid_map.iter().map(Vec::len).max().unwrap_or(0);
            let tsid_runs = tsid_map
                .iter()
                .map(|globals| {
                    globals
                        .iter()
                        .enumerate()
                        .filter(|(index, global)| {
                            *index == 0 || **global != globals[*index - 1].saturating_add(1)
                        })
                        .count()
                })
                .sum::<usize>();
            eprintln!(
                "[glrmask/profile][token_deterministic_terminal_nwa_input] input={} states={} transitions={} local_tsids={} tsid_pairs={} tsid_runs={} tsid_max_fanout={} remap_ms={:.3}",
                input_index,
                input.dwa.num_states(),
                input.dwa.num_transitions(),
                input.id_map.num_tsids(),
                tsid_pairs,
                tsid_runs,
                tsid_max_fanout,
                elapsed_ms,
            );
        }
        if let Some(detail) = remap_detail {
            eprintln!(
                "[glrmask/profile][token_nwa_remap_detail] input={} unique_weights={} full_weights={} compact_entries={} local_tsid_visits={} global_tsid_expansions={} max_local_entry_span={} max_global_entry_expansion={}",
                input_index,
                detail.unique_weights,
                detail.full_weights,
                detail.compact_entries,
                detail.local_tsid_visits,
                detail.global_tsid_expansions,
                detail.max_local_entry_span,
                detail.max_global_entry_expansion,
            );
        }
        let started_at = profiling.then(Instant::now);
        let source_nwa = dwa.to_nwa();
        if let Some(started_at) = started_at {
            to_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        let before_append = global_nwa.num_states() as usize;
        let body = global_nwa.append_with_body(&source_nwa);
        if let Some(started_at) = started_at {
            append_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        state_sources.extend(std::iter::repeat_n(Some(input_index), global_nwa.num_states() as usize - before_append));
        debug_assert_eq!(body.start_states.len(), 1);
        let source_start = body.start_states[0];
        let started_at = profiling.then(Instant::now);
        if let Some(combined_start) = combined_start {
            let source = global_nwa.states()[source_start as usize].clone();
            if let Some(final_weight) = source.final_weight {
                let target = &mut global_nwa.states_mut()[combined_start as usize].final_weight;
                *target = Some(match target.take() {
                    Some(existing) => existing.union(&final_weight),
                    None => final_weight,
                });
            }
            for (label, branches) in source.transitions {
                for (target, weight) in branches {
                    global_nwa.add_transition(combined_start, label, target, weight);
                }
            }
            assert!(source.epsilons.is_empty(), "source DWA conversion must not create epsilon edges");
        } else {
            combined_start = Some(source_start);
            state_sources[source_start as usize] = None;
        }
        if let Some(started_at) = started_at {
            fuse_start_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
    }
    let started_at = profiling.then(Instant::now);
    global_nwa.set_start_states(vec![combined_start.expect("at least one input")]);
    let (mut global_nwa, state_sources) = prune_unreachable_nwa_with_sources(global_nwa, state_sources);
    let prune_ms = started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
    let mut source_quotient_ms = 0.0;
    let mut refine_ms = 0.0;
    if quotient_token_deterministic_terminal_nwa_by_source_enabled() {
        let started_at = profiling.then(Instant::now);
        global_nwa = quotient_disjoint_source_nwa_owned(global_nwa, &state_sources)
            .expect("token-deterministic source quotient failed");
        if let Some(started_at) = started_at {
            source_quotient_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        }
        if refine_token_deterministic_terminal_nwa_after_source_quotient_enabled() {
            let started_at = profiling.then(Instant::now);
            global_nwa = minimize_token_deterministic_nwa_owned(global_nwa)
                .expect("token-deterministic refinement after source quotient failed");
            if let Some(started_at) = started_at {
                refine_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            }
        }
    } else if minimize_token_deterministic_terminal_nwa_enabled() {
        let started_at = profiling.then(Instant::now);
        global_nwa = minimize_token_deterministic_nwa_owned(global_nwa)
            .expect("token-deterministic terminal NWA minimization failed");
        if let Some(started_at) = started_at {
            refine_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        }
    }
    debug_assert!(global_nwa.is_acyclic());
    debug_assert!(global_nwa.states().iter().all(|state| state.epsilons.is_empty()));
    let remap_and_union_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

    if assert_direct_global_terminal_dwa_merge_equivalence_enabled() {
        let reference = generic_global_union_determinization(
            inputs,
            &global_id_map,
            direct_local_to_global_token_maps.as_ref(),
        );
        let determinized = determinize(&global_nwa)
            .expect("token-deterministic terminal NWA assertion determinization failed");
        let difference = find_difference(&determinized, &reference)
            .expect("token-deterministic terminal NWA equivalence requires acyclic inputs");
        assert!(
            difference.is_none(),
            "token-deterministic terminal NWA differs from generic NWA union on labels {:?}",
            difference,
        );
        if profiling {
            eprintln!("[glrmask/profile][token_deterministic_terminal_nwa_equivalence] result=equivalent");
        }
    }

    if profiling {
        eprintln!(
            "[glrmask/profile][token_deterministic_terminal_nwa_merge] inputs={} id_map_ms={:.3} remap_and_union_ms={:.3} tsid_map_ms={:.3} token_map_ms={:.3} clone_ms={:.3} remap_ms={:.3} to_nwa_ms={:.3} append_ms={:.3} fuse_start_ms={:.3} prune_ms={:.3} source_quotient_ms={:.3} refine_ms={:.3} states={} transitions={} total_ms={:.3}",
            inputs.len(),
            id_map_ms,
            remap_and_union_ms,
            tsid_map_ms,
            token_map_ms,
            clone_ms,
            remap_ms,
            to_nwa_ms,
            append_ms,
            fuse_start_ms,
            prune_ms,
            source_quotient_ms,
            refine_ms,
            global_nwa.num_states(),
            global_nwa.num_transitions(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some((
        global_nwa,
        global_id_map,
        TerminalDwaPhaseProfile {
            id_map_ms,
            global_merge_ms: total_started_at.elapsed().as_secs_f64() * 1000.0,
            ..TerminalDwaPhaseProfile::default()
        },
    ))
}

/// Merge already-compacted partition outputs into one global DWA.
///
/// Partition-local merges have already minimized their local outputs. When
/// global token domains are disjoint, an exact deterministic source-product is
/// used and remains unminimized/uncompacted by default; this avoids constructing
/// a large subset DWA only to erase it immediately. Overlapping domains retain
/// the generic determinize-minimize-compact route. `GLRMASK_MINIMIZE_MERGED_TERMINAL_DWA`
/// and `GLRMASK_COMPACT_MERGED_TERMINAL_DWA` explicitly override either default.
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
    // The direct product is exact when token domains are disjoint. Unlike the
    // generic NWA route it keeps source-local deterministic contexts separate,
    // avoiding a large subset expansion that global minimization later erases.
    let direct_global = try_merge_disjoint_token_domain_dwas(
        &inputs,
        &global_id_map,
        direct_local_to_global_token_maps.as_ref(),
        max_token_id,
);
    let used_direct_global = direct_global.is_some();
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

    if direct_global.is_none() {
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
    }
    let remap_and_union_ms = remap_and_union_started_at.elapsed().as_secs_f64() * 1000.0;
    let nwa_states_before_determinize = global_nwa.num_states();
    let nwa_transitions_before_determinize = global_nwa.num_transitions();

    let determinize_started_at = Instant::now();
    let det = direct_global.unwrap_or_else(|| {
        determinize(&global_nwa).expect("merge terminal NWA determinization failed")
    });
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
    let det_states = det.num_states();
    let det_transitions = det.num_transitions();

    if used_direct_global && assert_direct_global_terminal_dwa_merge_equivalence_enabled() {
        let reference = generic_global_union_determinization(
            &inputs,
            &global_id_map,
            direct_local_to_global_token_maps.as_ref(),
        );
        let difference = find_difference(&det, &reference)
            .expect("direct global terminal DWA equivalence requires acyclic inputs");
        assert!(
            difference.is_none(),
            "direct global terminal DWA differs from generic NWA union on labels {:?}",
            difference,
        );
        if profiling {
            eprintln!("[glrmask/profile][direct_global_terminal_dwa_equivalence] result=equivalent");
        }
    }

    // A direct product is already exact. Its slightly larger topology is much
    // cheaper to construct than generic subset expansion and remains suitable
    // for the downstream joint reconciliation pass, so do not minimize it by
    // default. Explicit environment values still override this strategy.
    let minimize_enabled = minimize_merged_terminal_dwa_enabled(!used_direct_global);
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
    let compact_enabled = compact_merged_terminal_dwa_enabled(!used_direct_global);
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
            minimize_enabled,
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

/// Fast exact global map construction for an already-proven disjoint token
/// partition. The input order and original-state order are deterministic, so
/// first-seen class IDs are stable for this merge and avoid a costly sort and
/// relabel pass used only for cross-order canonicalization.
fn build_unified_global_id_map_disjoint_fast(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
    primary_source: Option<usize>,
) -> (InternalIdMap, Vec<Vec<u32>>, Vec<Vec<Vec<u32>>>) {
    let mut composite_to_class = FxHashMap::<SmallVec<[u32; 8]>, u32>::default();
    let mut state_o2i = vec![0u32; num_tokenizer_states];
    let mut state_i2o: Vec<Vec<u32>> = Vec::new();
    let mut state_reps: Vec<u32> = Vec::new();
    let mut class_keys = Vec::<SmallVec<[u32; 8]>>::new();

    for state in 0..num_tokenizer_states {
        let mut composite = SmallVec::<[u32; 8]>::with_capacity(inputs.len());
        composite.extend(
            inputs
                .iter()
                .map(|input| input.tokenizer_states.original_to_internal[state]),
        );
        let next_id = state_i2o.len() as u32;
        let class = if let Some(&existing) = composite_to_class.get(&composite) {
            existing
        } else {
            let class = next_id;
            class_keys.push(composite.clone());
            composite_to_class.insert(composite, class);
            state_i2o.push(Vec::new());
            state_reps.push(state as u32);
            class
        };
        state_o2i[state] = class;
        state_i2o[class as usize].push(state as u32);
    }

    let source_order = token_nwa_tsid_source_order(inputs.len()).or_else(|| {
        primary_source
            .filter(|source| *source < inputs.len())
            .map(|primary| {
                std::iter::once(primary)
                    .chain((0..inputs.len()).filter(move |&source| source != primary))
                    .collect()
            })
    });
    if let Some(source_order) = source_order {
        let mut order: Vec<usize> = (0..class_keys.len()).collect();
        order.sort_unstable_by(|&left, &right| {
            for &source in &source_order {
                let comparison = class_keys[left][source].cmp(&class_keys[right][source]);
                if !comparison.is_eq() {
                    return comparison;
                }
            }
            left.cmp(&right)
        });
        let mut old_to_new = vec![0u32; class_keys.len()];
        for (new_id, &old_id) in order.iter().enumerate() {
            old_to_new[old_id] = new_id as u32;
        }
        for class in &mut state_o2i {
            *class = old_to_new[*class as usize];
        }
        let mut new_keys = Vec::with_capacity(class_keys.len());
        let mut new_i2o = Vec::with_capacity(state_i2o.len());
        let mut new_reps = Vec::with_capacity(state_reps.len());
        for old_id in order {
            new_keys.push(std::mem::take(&mut class_keys[old_id]));
            new_i2o.push(std::mem::take(&mut state_i2o[old_id]));
            new_reps.push(state_reps[old_id]);
        }
        class_keys = new_keys;
        state_i2o = new_i2o;
        state_reps = new_reps;
    }

    let mut local_to_global_tsids: Vec<Vec<Vec<u32>>> = inputs
        .iter()
        .map(|input| vec![Vec::new(); input.num_tsids() as usize])
        .collect();
    for (global_tsid, composite) in class_keys.iter().enumerate() {
        for (input_index, &local_tsid) in composite.iter().enumerate() {
            local_to_global_tsids[input_index][local_tsid as usize].push(global_tsid as u32);
        }
    }

    let (vocab_tokens, direct_local_to_global_token_maps) =
        build_unified_global_token_id_map_assume_disjoint(inputs, max_token_id);
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
        local_to_global_tsids,
    )
}

fn build_unified_global_token_id_map_assume_disjoint(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> (ManyToOneIdMap, Vec<Vec<u32>>) {
    let mut token_o2i = vec![u32::MAX; max_token_id as usize + 1];
    let token_class_count: usize = inputs
        .iter()
        .map(|input| input.vocab_tokens.internal_to_originals.len())
        .sum();
    let mut token_i2o = Vec::with_capacity(token_class_count);
    let mut token_reps = Vec::with_capacity(token_class_count);
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

    (
        ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
        direct_local_to_global_token_maps,
    )
}

/// Fast exact global map construction for an already-proven disjoint token
/// partition. The input order and original-state order are deterministic, so
/// first-seen class IDs are stable for this merge and avoid a costly sort and
/// relabel pass used only for cross-order canonicalization.
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

/// Map local TSIDs to the global composite TSIDs.
///
/// `build_unified_global_id_map` constructs each global TSID from the tuple of
/// all input-local TSIDs. Therefore any representative original tokenizer
/// state of a global class identifies the unique local class that owns it.
/// Iterating global representatives emits each destination once, in ascending
/// global-ID order, avoiding the old per-input `BTreeSet` aggregation.
fn build_local_to_global_tsid_map(
    local_id_map: &InternalIdMap,
    global_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let mut local_to_global = vec![Vec::new(); local_id_map.num_tsids() as usize];
    for (global_tsid, &representative_state) in global_id_map
        .tokenizer_states
        .representative_original_ids
        .iter()
        .enumerate()
    {
        let local_tsid = local_id_map.tokenizer_states.original_to_internal
            [representative_state as usize] as usize;
        local_to_global[local_tsid].push(global_tsid as u32);

        #[cfg(debug_assertions)]
        for &original_state in &global_id_map.tokenizer_states.internal_to_originals[global_tsid] {
            debug_assert_eq!(
                local_id_map.tokenizer_states.original_to_internal[original_state as usize],
                local_tsid as u32,
                "a global composite TSID must refine every local TSID map",
            );
        }
    }
    local_to_global
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
    let mut weight_cache = RemapCache::<Weight>::default();
    let mut token_cache = RemapCache::<Arc<RangeSetBlaze<u32>>>::default();

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
    cache: &mut RemapCache<Weight>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    remap_weight_cached_with_token_remap(
        weight,
        local_to_global_tsids,
        &TokenRemap::Explicit(local_to_global_tokens),
        global_tsid_count,
        cache,
        token_cache,
    )
}

fn remap_weight_cached_with_tsid_runs(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tsid_runs: &[SmallVec<[(u32, u32); 1]>],
    token_remap: &TokenRemap<'_>,
    global_tsid_count: usize,
    cache: &mut RemapCache<Weight>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    let ptr = Arc::as_ptr(&weight.0) as usize;
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    let remapped = remap_weight_with_tsid_runs(
        weight,
        local_to_global_tsids,
        local_to_global_tsid_runs,
        token_remap,
        global_tsid_count,
        token_cache,
    );
    cache.insert(ptr, remapped.clone());
    remapped
}

fn remap_weight_with_tsid_runs(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tsid_runs: &[SmallVec<[(u32, u32); 1]>],
    token_remap: &TokenRemap<'_>,
    global_tsid_count: usize,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    if weight.is_empty() {
        return weight.clone();
    }
    if weight.is_full() {
        let all_global_tokens = token_remap.all_global_tokens();
        let Some(last_global_tsid) = global_tsid_count.checked_sub(1).map(|count| count as u32) else {
            return Weight::empty();
        };
        return (!all_global_tokens.is_empty())
            .then(|| Weight::from_uniform(0..=last_global_tsid, all_global_tokens))
            .unwrap_or_else(Weight::empty);
    }

    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };
    let mut intervals = SmallVec::<[(u32, u32, Arc<RangeSetBlaze<u32>>); 4]>::new();
    for (start, end, tokens) in entries {
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_remap.map_tokens(token_key, tokens.as_ref(), token_cache);
        if mapped_tokens.is_empty() {
            continue;
        }
        for local_tsid in start..=end {
            let Some(runs) = local_to_global_tsid_runs.get(local_tsid as usize) else {
                continue;
            };
            for &(global_start, global_end) in runs {
                intervals.push((global_start, global_end, Arc::clone(&mapped_tokens)));
            }
        }
    }
    if intervals.is_empty() {
        return Weight::empty();
    }
    intervals.sort_unstable_by_key(|(start, _, _)| *start);

    // A global refinement class has one local class for this source, so these
    // intervals are disjoint. Keep the old generic path as an exact fallback
    // if a caller ever supplies a non-partition map.
    for pair in intervals.windows(2) {
        if pair[0].1 >= pair[1].0 {
            return remap_weight_with_token_remap(
                weight,
                local_to_global_tsids,
                token_remap,
                global_tsid_count,
                token_cache,
            );
        }
    }

    use crate::ds::weight::finalize_weight_map;
    use range_set_blaze::RangeMapBlaze;
    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start = intervals[0].0;
    let mut run_end = intervals[0].1;
    let mut run_tokens = Arc::clone(&intervals[0].2);
    for (start, end, tokens) in intervals.into_iter().skip(1) {
        if start == run_end.saturating_add(1)
            && (Arc::ptr_eq(&run_tokens, &tokens) || run_tokens.as_ref() == tokens.as_ref())
        {
            run_end = end;
        } else {
            map.extend_simple(std::iter::once((run_start..=run_end, run_tokens)));
            run_start = start;
            run_end = end;
            run_tokens = tokens;
        }
    }
    map.extend_simple(std::iter::once((run_start..=run_end, run_tokens)));
    finalize_weight_map(map)
}

fn remap_weight_cached_with_token_remap(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    token_remap: &TokenRemap<'_>,
    global_tsid_count: usize,
    cache: &mut RemapCache<Weight>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    let ptr = Arc::as_ptr(&weight.0) as usize;
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    let remapped = remap_weight_with_token_remap(
        weight,
        local_to_global_tsids,
        token_remap,
        global_tsid_count,
        token_cache,
    );
    cache.insert(ptr, remapped.clone());
    remapped
}

fn remap_weight_with_token_remap(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    token_remap: &TokenRemap<'_>,
    global_tsid_count: usize,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    if weight.is_empty() {
        return weight.clone();
    }

    if weight.is_full() {
        let all_global_tokens = token_remap.all_global_tokens();
        let Some(last_global_tsid) = global_tsid_count.checked_sub(1).map(|count| count as u32) else {
            return Weight::empty();
        };
        if all_global_tokens.is_empty() {
            return Weight::empty();
        }
        // Every global TSID contains exactly one local TSID for this source.
        // Hence local `all` covers the complete global TSID domain even when
        // individual local classes are interleaved in that global ordering.
        return Weight::from_uniform(0..=last_global_tsid, all_global_tokens);
    }

    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };
    use crate::ds::weight::{finalize_weight_map, shared_rangeset};

    let mut tokens_by_global_tsid = Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new();
    for (start, end, tokens) in entries {
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_remap.map_tokens(token_key, tokens.as_ref(), token_cache);

        for local_tsid in start..=end {
            let Some(global_tsids) = local_to_global_tsids.get(local_tsid as usize) else {
                continue;
            };
            for &global_tsid in global_tsids {
                if (global_tsid as usize) < global_tsid_count {
                    tokens_by_global_tsid.push((global_tsid, Arc::clone(&mapped_tokens)));
                }
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

    use range_set_blaze::RangeMapBlaze;
    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start: Option<u32> = None;
    let mut run_end = 0u32;
    let mut run_tokens: Option<Arc<RangeSetBlaze<u32>>> = None;
    for (global_tsid, tokens) in merged_by_global_tsid {
        if let Some(current) = &run_tokens {
            if global_tsid == run_end.wrapping_add(1)
                && (Arc::ptr_eq(current, &tokens) || current.as_ref() == tokens.as_ref())
            {
                run_end = global_tsid;
                continue;
            }
            map.extend_simple(std::iter::once((
                run_start.expect("run start must exist")..=run_end,
                Arc::clone(current),
            )));
        }
        run_start = Some(global_tsid);
        run_end = global_tsid;
        run_tokens = Some(tokens);
    }
    if let Some(tokens) = run_tokens {
        map.extend_simple(std::iter::once((
            run_start.expect("run start must exist")..=run_end,
            tokens,
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

    fn id_map_with_tsid_partition(
        original_to_internal: Vec<u32>,
        representative_original_ids: Vec<u32>,
    ) -> InternalIdMap {
        let num_tsids = representative_original_ids.len() as u32;
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                original_to_internal,
                num_tsids,
                representative_original_ids,
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                Vec::new(),
                0,
            ),
        }
    }

    fn reference_local_to_global_tsid_map(
        local_id_map: &InternalIdMap,
        global_id_map: &InternalIdMap,
    ) -> Vec<Vec<u32>> {
        let mut result = vec![std::collections::BTreeSet::new(); local_id_map.num_tsids() as usize];
        for (original_state, &local_tsid) in local_id_map
            .tokenizer_states
            .original_to_internal
            .iter()
            .enumerate()
        {
            result[local_tsid as usize]
                .insert(global_id_map.tokenizer_states.original_to_internal[original_state]);
        }
        result
            .into_iter()
            .map(|targets| targets.into_iter().collect())
            .collect()
    }

    #[test]
    fn representative_tsid_map_matches_set_reference() {
        let global = id_map_with_tsid_partition(vec![0, 0, 1, 1, 2, 2], vec![0, 2, 4]);
        let local = id_map_with_tsid_partition(vec![0, 0, 1, 1, 0, 0], vec![0, 2]);

        assert_eq!(
            build_local_to_global_tsid_map(&local, &global),
            reference_local_to_global_tsid_map(&local, &global),
        );
    }

    #[test]
    fn contiguous_token_offset_rejects_unmapped_classes() {
        assert_eq!(contiguous_token_offset(&[9, 10, 11]), Some(9));
        assert_eq!(contiguous_token_offset(&[u32::MAX]), None);
        assert_eq!(contiguous_token_offset(&[9, u32::MAX]), None);
    }

    #[test]
    fn affine_token_remap_matches_explicit_token_map() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=2, 5..=5])),
            (1, RangeSetBlaze::from_iter([1..=4])),
        ]);
        let tsid_map = vec![vec![1, 3], vec![2]];
        let explicit = vec![vec![7], vec![8], vec![9], vec![10], vec![11], vec![12]];
        let mut explicit_cache = RemapCache::default();
        let mut explicit_tokens = RemapCache::default();
        let explicit_result = remap_weight_cached_with_token_remap(
            &weight,
            &tsid_map,
            &TokenRemap::Explicit(&explicit),
            4,
            &mut explicit_cache,
            &mut explicit_tokens,
        );
        let mut offset_cache = RemapCache::default();
        let mut offset_tokens = RemapCache::default();
        let offset_result = remap_weight_cached_with_token_remap(
            &weight,
            &tsid_map,
            &TokenRemap::Offset {
                offset: 7,
                local_token_count: 6,
            },
            4,
            &mut offset_cache,
            &mut offset_tokens,
        );
        assert_eq!(explicit_result, offset_result);
    }

    #[test]
    fn interval_remap_matches_generic_and_overlapping_fallback() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([2..=3])),
        ]);
        let token_remap = TokenRemap::Offset {
            offset: 10,
            local_token_count: 4,
        };

        for tsid_map in [vec![vec![2, 4], vec![0, 1, 3]], vec![vec![0], vec![0]]] {
            let runs = build_local_to_global_tsid_runs(&tsid_map);
            let mut generic_tokens = RemapCache::default();
            let generic = remap_weight_with_token_remap(
                &weight,
                &tsid_map,
                &token_remap,
                5,
                &mut generic_tokens,
            );
            let mut interval_tokens = RemapCache::default();
            let interval = remap_weight_with_tsid_runs(
                &weight,
                &tsid_map,
                &runs,
                &token_remap,
                5,
                &mut interval_tokens,
            );
            assert_eq!(interval, generic);
        }
    }

    #[test]
    fn full_weight_remap_covers_uniform_global_tsid_domain() {
        let tsid_map = vec![vec![2, 4], vec![0, 1, 3]];
        let mut cache = RemapCache::default();
        let mut token_cache = RemapCache::default();
        let remapped = remap_weight_cached_with_token_remap(
            &Weight::all(),
            &tsid_map,
            &TokenRemap::Offset {
                offset: 10,
                local_token_count: 3,
            },
            5,
            &mut cache,
            &mut token_cache,
        );
        let expected = Weight::from_uniform(0..=4, RangeSetBlaze::from_iter([10..=12]));
        assert_eq!(remapped, expected);
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
