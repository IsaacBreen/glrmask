use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use rustc_hash::FxHashMap;

use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::{SharedTokenSet, Weight, finalize_weight_map, shared_rangeset};

pub(super) fn reconcile_weight_id_maps(
    left_weights: &mut [&mut Weight],
    left_id_map: &mut InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &mut InternalIdMap,
) {
    if try_reconcile_refinement_fast_path(left_weights, left_id_map, right_weights, right_id_map) {
        return;
    }

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
    if right_weights.is_empty() {
        return left_id_map.clone();
    }
    if left_weights.is_empty() {
        return right_id_map.clone();
    }
    if internal_id_map_refines(left_id_map, right_id_map) {
        remap_weights_into_existing_common(right_weights, right_id_map, left_id_map);
        return left_id_map.clone();
    }
    if internal_id_map_refines(right_id_map, left_id_map) {
        remap_weights_into_existing_common(left_weights, left_id_map, right_id_map);
        return right_id_map.clone();
    }

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

fn try_reconcile_refinement_fast_path(
    left_weights: &mut [&mut Weight],
    left_id_map: &mut InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &mut InternalIdMap,
) -> bool {
    if right_weights.is_empty() {
        *right_id_map = left_id_map.clone();
        return true;
    }
    if left_weights.is_empty() {
        *left_id_map = right_id_map.clone();
        return true;
    }
    if internal_id_map_refines(left_id_map, right_id_map) {
        remap_weights_into_existing_common(right_weights, right_id_map, left_id_map);
        *right_id_map = left_id_map.clone();
        return true;
    }
    if internal_id_map_refines(right_id_map, left_id_map) {
        remap_weights_into_existing_common(left_weights, left_id_map, right_id_map);
        *left_id_map = right_id_map.clone();
        return true;
    }
    false
}

fn remap_weights_into_existing_common(
    weights: &mut [&mut Weight],
    local_id_map: &InternalIdMap,
    common_id_map: &InternalIdMap,
) {
    if weights.is_empty() {
        return;
    }
    let tsid_map = build_local_to_common_tsid_map(local_id_map, common_id_map);
    let token_map = build_local_to_common_token_map_from_common_classes(local_id_map, common_id_map);
    remap_weights_with_maps(
        weights,
        &tsid_map,
        &token_map,
        common_id_map.num_tsids() as usize,
    );
}

fn internal_id_map_refines(finer: &InternalIdMap, coarser: &InternalIdMap) -> bool {
    many_to_one_id_map_refines(&finer.tokenizer_states, &coarser.tokenizer_states, false)
        && many_to_one_id_map_refines(&finer.vocab_tokens, &coarser.vocab_tokens, true)
}

fn many_to_one_id_map_refines(
    finer: &ManyToOneIdMap,
    coarser: &ManyToOneIdMap,
    allow_unmapped: bool,
) -> bool {
    if finer.original_to_internal.len() < coarser.original_to_internal.len() {
        return false;
    }

    let mut coarser_by_finer = vec![None; finer.internal_to_originals.len()];
    for original in 0..finer.original_to_internal.len() {
        let finer_internal = finer.original_to_internal[original];
        let coarser_internal = coarser
            .original_to_internal
            .get(original)
            .copied()
            .unwrap_or(u32::MAX);
        if allow_unmapped && finer_internal == u32::MAX && coarser_internal == u32::MAX {
            continue;
        }
        let Some(slot) = coarser_by_finer.get_mut(finer_internal as usize) else {
            return false;
        };
        match slot {
            Some(previous) if *previous != coarser_internal => return false,
            Some(_) => {}
            None => *slot = Some(coarser_internal),
        }
    }

    true
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

#[derive(Debug)]
struct InjectiveLocalMap {
    destination_by_local: Vec<u32>,
    destination_order_is_monotone: bool,
}

impl InjectiveLocalMap {
    fn from_local_to_common(local_to_common: &[Vec<u32>], common_count: usize) -> Option<Self> {
        let mut destination_by_local = vec![u32::MAX; local_to_common.len()];
        let mut seen_destinations = vec![false; common_count];
        let mut previous_destination = None;
        let mut destination_order_is_monotone = true;

        for (local, destinations) in local_to_common.iter().enumerate() {
            let &destination = match destinations.as_slice() {
                [] => continue,
                [destination] => destination,
                _ => return None,
            };
            let destination_index = destination as usize;
            if destination_index >= common_count || std::mem::replace(&mut seen_destinations[destination_index], true) {
                return None;
            }
            if let Some(previous) = previous_destination
                && destination <= previous
            {
                destination_order_is_monotone = false;
            }
            previous_destination = Some(destination);
            destination_by_local[local] = destination;
        }

        Some(Self {
            destination_by_local,
            destination_order_is_monotone,
        })
    }

    #[inline]
    fn destination(&self, local: u32) -> Option<u32> {
        self.destination_by_local
            .get(local as usize)
            .copied()
            .filter(|&destination| destination != u32::MAX)
    }
}

fn remap_token_set_with_injective_map(
    tokens: &SharedTokenSet,
    token_map: &InjectiveLocalMap,
    cache: &mut HashMap<usize, SharedTokenSet>,
) -> SharedTokenSet {
    let key = Arc::as_ptr(tokens) as usize;
    if let Some(cached) = cache.get(&key) {
        return Arc::clone(cached);
    }

    let mut mapped = RangeSetBlaze::new();
    for local_token in tokens.iter() {
        if let Some(common_token) = token_map.destination(local_token) {
            mapped.insert(common_token);
        }
    }
    // Keep the generic remapper's sharing boundary: it creates a fresh token
    // set per source-weight remap and shares only within that weight. Global
    // interning here changes serialized artifact layout even though masks are
    // equivalent.
    let mapped = Arc::new(mapped);
    cache.insert(key, Arc::clone(&mapped));
    mapped
}

fn remap_weight_with_injective_maps(
    weight: &Weight,
    tsid_map: &InjectiveLocalMap,
    token_map: &InjectiveLocalMap,
    token_cache: &mut HashMap<usize, SharedTokenSet>,
) -> Weight {
    // Preserve the generic path for the special universal representation. It
    // is rare and has different "all mapped IDs" semantics.
    if weight.is_empty() || weight.is_full() {
        return weight.clone();
    }

    let mut entries = Vec::<(u32, SharedTokenSet)>::new();
    for (local_range, tokens) in weight.0.range_values() {
        let mapped_tokens = remap_token_set_with_injective_map(tokens, token_map, token_cache);
        if mapped_tokens.is_empty() {
            continue;
        }
        for local_tsid in *local_range.start()..=*local_range.end() {
            if let Some(common_tsid) = tsid_map.destination(local_tsid) {
                entries.push((common_tsid, Arc::clone(&mapped_tokens)));
            }
        }
    }

    if entries.is_empty() {
        return Weight::empty();
    }
    if !tsid_map.destination_order_is_monotone {
        entries.sort_unstable_by_key(|(common_tsid, _)| *common_tsid);
    }

    // Match the general remapper's canonical construction order exactly. The
    // fast path avoids the `common_tsid_count`-sized scratch vector, not the
    // final RangeMap / interning boundary.
    let mut map = RangeMapBlaze::<u32, SharedTokenSet>::new();
    let mut run_start = entries[0].0;
    let mut run_end = entries[0].0;
    let mut run_tokens = Arc::clone(&entries[0].1);
    for (common_tsid, tokens) in entries.into_iter().skip(1) {
        if common_tsid == run_end + 1
            && (Arc::ptr_eq(&run_tokens, &tokens) || run_tokens.as_ref() == tokens.as_ref())
        {
            run_end = common_tsid;
            continue;
        }
        map.extend_simple(std::iter::once((run_start..=run_end, run_tokens)));
        run_start = common_tsid;
        run_end = common_tsid;
        run_tokens = tokens;
    }
    map.extend_simple(std::iter::once((run_start..=run_end, run_tokens)));
    finalize_weight_map(map)
}

fn remap_weights_with_maps(
    weights: &mut [&mut Weight],
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
) {
    let tsid_map = InjectiveLocalMap::from_local_to_common(local_to_common_tsids, common_tsid_count);
    let token_map = InjectiveLocalMap::from_local_to_common(
        local_to_common_tokens,
        local_to_common_tokens
            .iter()
            .flatten()
            .copied()
            .max()
            .map_or(0, |maximum| maximum as usize + 1),
    );
    let mut cache = HashMap::<usize, Weight>::new();

    for weight in weights.iter_mut() {
        let ptr = Arc::as_ptr(&weight.0) as usize;
        let remapped = if let Some(cached) = cache.get(&ptr) {
            cached.clone()
        } else {
            let remapped = match (&tsid_map, &token_map) {
                (Some(tsid_map), Some(token_map)) if !weight.is_full() => {
                    let mut token_cache = HashMap::<usize, SharedTokenSet>::new();
                    remap_weight_with_injective_maps(weight, tsid_map, token_map, &mut token_cache)
                }
                _ => remap_weight_general(
                    weight,
                    local_to_common_tsids,
                    local_to_common_tokens,
                    common_tsid_count,
                ),
            };
            cache.insert(ptr, remapped.clone());
            remapped
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn map(original_to_internal: Vec<u32>, num_internal: u32) -> ManyToOneIdMap {
        ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            original_to_internal,
            num_internal,
        )
    }

    fn id_map(
        states: Vec<u32>,
        num_states: u32,
        tokens: Vec<u32>,
        num_tokens: u32,
    ) -> InternalIdMap {
        InternalIdMap {
            tokenizer_states: map(states, num_states),
            vocab_tokens: map(tokens, num_tokens),
        }
    }

    fn entries_key(weight: &Weight) -> Vec<(u32, u32, Vec<(u32, u32)>)> {
        weight
            .0
            .range_values()
            .map(|(range, tokens)| {
                (
                    *range.start(),
                    *range.end(),
                    tokens
                        .ranges()
                        .map(|range| (*range.start(), *range.end()))
                        .collect(),
                )
            })
            .collect()
    }

    fn id_map_key(id_map: &InternalIdMap) -> (Vec<u32>, Vec<Vec<u32>>, Vec<u32>, Vec<Vec<u32>>) {
        (
            id_map.tokenizer_states.original_to_internal.clone(),
            id_map.tokenizer_states.internal_to_originals.clone(),
            id_map.vocab_tokens.original_to_internal.clone(),
            id_map.vocab_tokens.internal_to_originals.clone(),
        )
    }

    fn reconcile_generic_for_test(
        left_weight: &mut Weight,
        left_id_map: &mut InternalIdMap,
        right_weight: &mut Weight,
        right_id_map: &mut InternalIdMap,
    ) {
        let common_id_map = build_common_internal_id_map(&[left_id_map, right_id_map]);
        let left_tsid_map = build_local_to_common_tsid_map(left_id_map, &common_id_map);
        let left_token_map =
            build_local_to_common_token_map_from_common_classes(left_id_map, &common_id_map);
        let right_tsid_map = build_local_to_common_tsid_map(right_id_map, &common_id_map);
        let right_token_map =
            build_local_to_common_token_map_from_common_classes(right_id_map, &common_id_map);

        remap_weights_with_maps(
            &mut [left_weight],
            &left_tsid_map,
            &left_token_map,
            common_id_map.num_tsids() as usize,
        );
        remap_weights_with_maps(
            &mut [right_weight],
            &right_tsid_map,
            &right_token_map,
            common_id_map.num_tsids() as usize,
        );

        *left_id_map = common_id_map.clone();
        *right_id_map = common_id_map;
    }

    #[test]
    fn injective_reconcile_remap_matches_general_for_reordered_ids() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([3..=3])),
            (2, RangeSetBlaze::from_iter([0..=1])),
            (3, RangeSetBlaze::from_iter([3..=3])),
        ]);
        let tsid_map = vec![vec![2], vec![0], vec![3], vec![1]];
        let token_map = vec![vec![3], vec![1], vec![0], vec![2]];

        let general = remap_weight_general(&weight, &tsid_map, &token_map, 4);
        let fast = remap_weight_with_injective_maps(
            &weight,
            &InjectiveLocalMap::from_local_to_common(&tsid_map, 4).expect("injective tsids"),
            &InjectiveLocalMap::from_local_to_common(&token_map, 4).expect("injective tokens"),
            &mut HashMap::new(),
        );

        assert_eq!(entries_key(&fast), entries_key(&general));
    }

    #[test]
    fn refinement_reconcile_matches_generic_with_non_empty_coarser_weights() {
        let left_id_map = id_map(
            vec![0, 0, 1, 1, 2, 2],
            3,
            vec![0, 0, 1, 1, 2, 2],
            3,
        );
        let right_id_map = id_map(vec![0, 0, 0, 0, 1, 1], 2, vec![0, 0, 0, 0, 0, 0], 1);
        assert!(internal_id_map_refines(&left_id_map, &right_id_map));

        let left_weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (2, RangeSetBlaze::from_iter([2..=2])),
        ]);
        let right_weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=0])),
            (1, RangeSetBlaze::from_iter([0..=0])),
        ]);

        let mut fast_left_id_map = left_id_map.clone();
        let mut fast_right_id_map = right_id_map.clone();
        let mut fast_left_weight = left_weight.clone();
        let mut fast_right_weight = right_weight.clone();
        reconcile_weight_id_maps(
            &mut [&mut fast_left_weight],
            &mut fast_left_id_map,
            &mut [&mut fast_right_weight],
            &mut fast_right_id_map,
        );

        let mut generic_left_id_map = left_id_map;
        let mut generic_right_id_map = right_id_map;
        let mut generic_left_weight = left_weight;
        let mut generic_right_weight = right_weight;
        reconcile_generic_for_test(
            &mut generic_left_weight,
            &mut generic_left_id_map,
            &mut generic_right_weight,
            &mut generic_right_id_map,
        );

        assert_eq!(id_map_key(&fast_left_id_map), id_map_key(&generic_left_id_map));
        assert_eq!(id_map_key(&fast_right_id_map), id_map_key(&generic_right_id_map));
        assert_eq!(entries_key(&fast_left_weight), entries_key(&generic_left_weight));
        assert_eq!(entries_key(&fast_right_weight), entries_key(&generic_right_weight));
    }

    #[test]
    fn refinement_reconcile_preserves_finer_weights_when_other_side_is_empty() {
        let left_id_map = id_map(
            vec![0, 1, 2, 3],
            4,
            vec![0, 1, 2, 3, 4, 5],
            6,
        );
        let right_id_map = id_map(vec![0, 0, 0, 0], 1, vec![0, 0, 0, 0, 0, 0], 1);
        assert!(internal_id_map_refines(&left_id_map, &right_id_map));

        let left_weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=2])),
            (3, RangeSetBlaze::from_iter([4..=5])),
        ]);
        let mut fast_left_id_map = left_id_map.clone();
        let mut fast_right_id_map = right_id_map;
        let mut fast_left_weight = left_weight.clone();
        reconcile_weight_id_maps(
            &mut [&mut fast_left_weight],
            &mut fast_left_id_map,
            &mut [],
            &mut fast_right_id_map,
        );

        assert_eq!(id_map_key(&fast_left_id_map), id_map_key(&left_id_map));
        assert_eq!(id_map_key(&fast_right_id_map), id_map_key(&left_id_map));
        assert_eq!(entries_key(&fast_left_weight), entries_key(&left_weight));
    }

    #[test]
    fn empty_side_reconcile_adopts_non_empty_side_after_domain_compaction() {
        let left_id_map = id_map(
            vec![0, 0, 1, 1],
            2,
            vec![0, 0, 1, 1, 2, 2],
            3,
        );
        let compact_empty_id_map = id_map(vec![0], 1, vec![0], 1);
        assert!(!internal_id_map_refines(&left_id_map, &compact_empty_id_map));

        let left_weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([2..=2])),
        ]);
        let mut fast_left_id_map = left_id_map.clone();
        let mut fast_right_id_map = compact_empty_id_map;
        let mut fast_left_weight = left_weight.clone();
        reconcile_weight_id_maps(
            &mut [&mut fast_left_weight],
            &mut fast_left_id_map,
            &mut [],
            &mut fast_right_id_map,
        );

        assert_eq!(id_map_key(&fast_left_id_map), id_map_key(&left_id_map));
        assert_eq!(id_map_key(&fast_right_id_map), id_map_key(&left_id_map));
        assert_eq!(entries_key(&fast_left_weight), entries_key(&left_weight));
    }
}
