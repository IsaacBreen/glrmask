use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use rustc_hash::FxHashMap;

use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::{Weight, finalize_weight_map, shared_rangeset};

pub(super) fn reconcile_weight_id_maps(
    left_weights: &mut [&mut Weight],
    left_id_map: &mut InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &mut InternalIdMap,
) {
    let common_id_map = build_common_internal_id_map(&[left_id_map, right_id_map]);

    let left_tsid_map = build_local_to_common_tsid_map(left_id_map, &common_id_map);
    let left_token_map = build_local_to_common_token_map_from_common_classes(left_id_map, &common_id_map);
    let right_tsid_map = build_local_to_common_tsid_map(right_id_map, &common_id_map);
    let right_token_map = build_local_to_common_token_map_from_common_classes(right_id_map, &common_id_map);

    remap_weights_with_maps(
        left_weights,
        &left_tsid_map,
        &left_token_map,
        common_id_map.num_tsids() as usize,
    );
    remap_weights_with_maps(
        right_weights,
        &right_tsid_map,
        &right_token_map,
        common_id_map.num_tsids() as usize,
    );

    *left_id_map = common_id_map.clone();
    *right_id_map = common_id_map;
}

pub(super) fn reconcile_weight_id_maps_into_common(
    left_weights: &mut [&mut Weight],
    left_id_map: &InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &InternalIdMap,
) -> InternalIdMap {
    let common_id_map = build_common_internal_id_map(&[left_id_map, right_id_map]);

    let left_tsid_map = build_local_to_common_tsid_map(left_id_map, &common_id_map);
    let left_token_map = build_local_to_common_token_map_from_common_classes(left_id_map, &common_id_map);
    let right_tsid_map = build_local_to_common_tsid_map(right_id_map, &common_id_map);
    let right_token_map = build_local_to_common_token_map_from_common_classes(right_id_map, &common_id_map);

    remap_weights_with_maps(
        left_weights,
        &left_tsid_map,
        &left_token_map,
        common_id_map.num_tsids() as usize,
    );
    remap_weights_with_maps(
        right_weights,
        &right_tsid_map,
        &right_token_map,
        common_id_map.num_tsids() as usize,
    );

    common_id_map
}

fn build_common_internal_id_map(inputs: &[&InternalIdMap]) -> InternalIdMap {
    let num_tokenizer_states = inputs
        .iter()
        .map(|input| input.tokenizer_states.original_to_internal.len())
        .max()
        .unwrap_or(0);
    let num_original_tokens = inputs
        .iter()
        .map(|input| input.vocab_tokens.original_to_internal.len())
        .max()
        .unwrap_or(0);

    let tokenizer_states = build_common_many_to_one_id_map(
        inputs,
        num_tokenizer_states,
        |input| &input.tokenizer_states,
        false,
    );
    let vocab_tokens = build_common_many_to_one_id_map(
        inputs,
        num_original_tokens,
        |input| &input.vocab_tokens,
        true,
    );

    InternalIdMap {
        tokenizer_states,
        vocab_tokens,
    }
}

fn build_common_many_to_one_id_map(
    inputs: &[&InternalIdMap],
    num_originals: usize,
    project: impl Fn(&InternalIdMap) -> &ManyToOneIdMap,
    allow_unmapped: bool,
) -> ManyToOneIdMap {
    if let [left, right] = inputs {
        return build_common_many_to_one_id_map_pair(
            project(left),
            project(right),
            num_originals,
            allow_unmapped,
        );
    }

    let mut composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut original_to_internal = vec![u32::MAX; num_originals];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut representatives: Vec<u32> = Vec::new();

    for original in 0..num_originals {
        let composite: Vec<u32> = inputs
            .iter()
            .map(|input| {
                project(input)
                    .original_to_internal
                    .get(original)
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        if allow_unmapped && composite.iter().all(|&value| value == u32::MAX) {
            continue;
        }

        let next_id = internal_to_originals.len() as u32;
        let class_id = *composite_to_class.entry(composite).or_insert_with(|| {
            internal_to_originals.push(Vec::new());
            representatives.push(original as u32);
            next_id
        });
        original_to_internal[original] = class_id;
        internal_to_originals[class_id as usize].push(original as u32);
    }

    reorder_common_classes(
        composite_to_class,
        &mut original_to_internal,
        &mut internal_to_originals,
        &mut representatives,
        allow_unmapped,
    );

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

fn build_common_many_to_one_id_map_pair(
    left: &ManyToOneIdMap,
    right: &ManyToOneIdMap,
    num_originals: usize,
    allow_unmapped: bool,
) -> ManyToOneIdMap {
    let mut composite_to_class: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut original_to_internal = vec![u32::MAX; num_originals];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut representatives: Vec<u32> = Vec::new();

    for (original, slot) in original_to_internal.iter_mut().enumerate() {
        let left_internal = left
            .original_to_internal
            .get(original)
            .copied()
            .unwrap_or(u32::MAX);
        let right_internal = right
            .original_to_internal
            .get(original)
            .copied()
            .unwrap_or(u32::MAX);
        if allow_unmapped && left_internal == u32::MAX && right_internal == u32::MAX {
            continue;
        }

        let composite = (left_internal, right_internal);
        let next_id = internal_to_originals.len() as u32;
        let class_id = *composite_to_class.entry(composite).or_insert_with(|| {
            internal_to_originals.push(Vec::new());
            representatives.push(original as u32);
            next_id
        });
        *slot = class_id;
        internal_to_originals[class_id as usize].push(original as u32);
    }

    reorder_common_pair_classes(
        composite_to_class,
        &mut original_to_internal,
        &mut internal_to_originals,
        &mut representatives,
        allow_unmapped,
    );

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

fn reorder_common_pair_classes(
    composite_to_class: FxHashMap<(u32, u32), u32>,
    original_to_internal: &mut [u32],
    internal_to_originals: &mut Vec<Vec<u32>>,
    representatives: &mut Vec<u32>,
    allow_unmapped: bool,
) {
    let num_classes = internal_to_originals.len();
    if num_classes <= 1 {
        return;
    }

    let mut sorted: Vec<((u32, u32), u32)> = composite_to_class.into_iter().collect();
    sorted.sort_unstable_by_key(|(composite, _)| *composite);

    let mut old_to_new = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        old_to_new[*old_id as usize] = new_id as u32;
    }

    for value in original_to_internal.iter_mut() {
        if *value == u32::MAX && allow_unmapped {
            continue;
        }
        *value = old_to_new[*value as usize];
    }

    let mut new_internal_to_originals = vec![Vec::new(); num_classes];
    let mut new_representatives = vec![u32::MAX; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        new_internal_to_originals[new_id] = std::mem::take(&mut internal_to_originals[*old_id as usize]);
        new_representatives[new_id] = representatives[*old_id as usize];
    }
    *internal_to_originals = new_internal_to_originals;
    *representatives = new_representatives;
}

fn reorder_common_classes(
    composite_to_class: HashMap<Vec<u32>, u32>,
    original_to_internal: &mut [u32],
    internal_to_originals: &mut Vec<Vec<u32>>,
    representatives: &mut Vec<u32>,
    allow_unmapped: bool,
) {
    let num_classes = internal_to_originals.len();
    if num_classes <= 1 {
        return;
    }

    let mut sorted: Vec<(Vec<u32>, u32)> = composite_to_class.into_iter().collect();
    sorted.sort_by(|left, right| left.0.cmp(&right.0));

    let mut old_to_new = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        old_to_new[*old_id as usize] = new_id as u32;
    }

    for value in original_to_internal.iter_mut() {
        if *value == u32::MAX && allow_unmapped {
            continue;
        }
        *value = old_to_new[*value as usize];
    }

    let mut new_internal_to_originals = vec![Vec::new(); num_classes];
    let mut new_representatives = vec![u32::MAX; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        new_internal_to_originals[new_id] = std::mem::take(&mut internal_to_originals[*old_id as usize]);
        new_representatives[new_id] = representatives[*old_id as usize];
    }
    *internal_to_originals = new_internal_to_originals;
    *representatives = new_representatives;
}

fn build_local_to_common_tsid_map(
    local_id_map: &InternalIdMap,
    common_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local_id_map.num_tsids() as usize;
    let mut local_to_common = vec![Vec::new(); num_local];

    for (state, &local_tsid) in local_id_map
        .tokenizer_states
        .original_to_internal
        .iter()
        .enumerate()
    {
        if local_tsid == u32::MAX {
            continue;
        }
        let common_tsid = common_id_map
            .tokenizer_states
            .original_to_internal
            .get(state)
            .copied()
            .unwrap_or(u32::MAX);
        if common_tsid == u32::MAX {
            continue;
        }
        local_to_common[local_tsid as usize].push(common_tsid);
    }

    sort_dedup_local_to_common(local_to_common)
}

fn build_local_to_common_token_map_from_common_classes(
    local_id_map: &InternalIdMap,
    common_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local_id_map.num_internal_tokens() as usize;
    let mut local_to_common = vec![Vec::new(); num_local];

    for (common_token, originals) in common_id_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let Some(&representative) = originals.first() else {
            continue;
        };
        let local_token = local_id_map
            .vocab_tokens
            .original_to_internal
            .get(representative as usize)
            .copied()
            .unwrap_or(u32::MAX);
        if local_token == u32::MAX {
            continue;
        }
        if let Some(common_tokens) = local_to_common.get_mut(local_token as usize) {
            common_tokens.push(common_token as u32);
        }
    }

    sort_dedup_local_to_common(local_to_common)
}

fn sort_dedup_local_to_common(mut local_to_common: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    for ids in &mut local_to_common {
        if ids.len() <= 1 {
            continue;
        }
        ids.sort_unstable();
        ids.dedup();
    }
    local_to_common
}

fn remap_weights_with_maps(
    weights: &mut [&mut Weight],
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
) {
    let mut cache = HashMap::<usize, Weight>::new();
    for weight in weights.iter_mut() {
        let remapped = remap_weight_cached_general(
            weight,
            local_to_common_tsids,
            local_to_common_tokens,
            common_tsid_count,
            &mut cache,
        );
        **weight = remapped;
    }
}

fn remap_weight_cached_general(
    weight: &Weight,
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
    cache: &mut HashMap<usize, Weight>,
) -> Weight {
    let ptr = Arc::as_ptr(&weight.0) as usize;
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }

    let remapped = remap_weight_general(
        weight,
        local_to_common_tsids,
        local_to_common_tokens,
        common_tsid_count,
    );
    cache.insert(ptr, remapped.clone());
    remapped
}

fn remap_weight_general(
    weight: &Weight,
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
) -> Weight {
    if weight.is_empty() {
        return weight.clone();
    }

    if weight.is_full() {
        let mut all_common_tokens = RangeSetBlaze::new();
        for common_tokens in local_to_common_tokens {
            for &common_token in common_tokens {
                all_common_tokens.insert(common_token);
            }
        }
        if all_common_tokens.is_empty() {
            return Weight::empty();
        }

        let mut all_common_tsids = BTreeSet::new();
        for common_tsids in local_to_common_tsids {
            for &common_tsid in common_tsids {
                if (common_tsid as usize) < common_tsid_count {
                    all_common_tsids.insert(common_tsid);
                }
            }
        }
        if all_common_tsids.is_empty() {
            return Weight::empty();
        }

        return Weight::from_per_tsid_token_sets(
            all_common_tsids
                .into_iter()
                .map(|common_tsid| (common_tsid, all_common_tokens.clone())),
        );
    }

    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };

    let mut token_cache = HashMap::<usize, Arc<RangeSetBlaze<u32>>>::new();
    let mut tokens_by_common_tsid: Vec<Option<Arc<RangeSetBlaze<u32>>>> = vec![None; common_tsid_count];
    let mut any_set = false;

    for (start, end, tokens) in entries {
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_cache
            .entry(token_key)
            .or_insert_with(|| {
                let mut result = RangeSetBlaze::new();
                for local_token in tokens.iter() {
                    if let Some(common_tokens) = local_to_common_tokens.get(local_token as usize) {
                        for &common_token in common_tokens {
                            result.insert(common_token);
                        }
                    }
                }
                Arc::new(result)
            })
            .clone();

        for local_tsid in start..=end {
            let Some(common_tsids) = local_to_common_tsids.get(local_tsid as usize) else {
                continue;
            };
            for &common_tsid in common_tsids {
                let index = common_tsid as usize;
                if index >= common_tsid_count {
                    continue;
                }
                match &mut tokens_by_common_tsid[index] {
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

    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start: Option<u32> = None;
    let mut run_end = 0u32;
    let mut run_tokens: Option<Arc<RangeSetBlaze<u32>>> = None;

    for (index, slot) in tokens_by_common_tsid.iter().enumerate() {
        let common_tsid = index as u32;
        if let Some(tokens) = slot {
            if let Some(ref current) = run_tokens {
                if Arc::ptr_eq(current, tokens) || current.as_ref() == tokens.as_ref() {
                    run_end = common_tsid;
                    continue;
                }
                map.extend_simple(std::iter::once((
                    run_start.unwrap()..=run_end,
                    Arc::clone(current),
                )));
            }
            run_start = Some(common_tsid);
            run_end = common_tsid;
            run_tokens = Some(Arc::clone(tokens));
        } else if let Some(ref current) = run_tokens {
            map.extend_simple(std::iter::once((
                run_start.unwrap()..=run_end,
                Arc::clone(current),
            )));
            run_start = None;
            run_tokens = None;
        }
    }
    if let Some(tokens) = run_tokens {
        map.extend_simple(std::iter::once((run_start.unwrap()..=run_end, tokens)));
    }

    finalize_weight_map(map)
}
