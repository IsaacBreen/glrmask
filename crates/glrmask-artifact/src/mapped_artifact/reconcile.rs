use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::{CheckSortedDisjointMap, RangeMapBlaze, RangeSetBlaze};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::equiv_types::{InternalIdMap, ManyToOneIdMap};
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

    remap_weights_into_existing_common(left_weights, left_id_map, &common_id_map);
    remap_weights_into_existing_common(right_weights, right_id_map, &common_id_map);

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

    reconcile_weight_id_maps_into_forced_common(
        left_weights,
        left_id_map,
        right_weights,
        right_id_map,
    )
}

pub(super) fn reconcile_weight_id_maps_into_forced_common(
    left_weights: &mut [&mut Weight],
    left_id_map: &InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &InternalIdMap,
) -> InternalIdMap {
    let profiling = std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some();
    let total_started_at = profiling.then(Instant::now);
    let build_started_at = profiling.then(Instant::now);
    let common_id_map = build_common_internal_id_map(&[left_id_map, right_id_map]);
    let build_ms = build_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let parallel_remap = std::env::var("GLRMASK_PARALLEL_FORCED_COMMON_REMAP")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let ((left_ms, ()), (right_ms, ())) = if parallel_remap {
        rayon::join(
            || {
                let started = profiling.then(Instant::now);
                remap_weights_into_existing_common(left_weights, left_id_map, &common_id_map);
                (
                    started
                        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0),
                    (),
                )
            },
            || {
                let started = profiling.then(Instant::now);
                remap_weights_into_existing_common(right_weights, right_id_map, &common_id_map);
                (
                    started
                        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0),
                    (),
                )
            },
        )
    } else {
        let left_started = profiling.then(Instant::now);
        remap_weights_into_existing_common(left_weights, left_id_map, &common_id_map);
        let left_ms = left_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let right_started = profiling.then(Instant::now);
        remap_weights_into_existing_common(right_weights, right_id_map, &common_id_map);
        let right_ms = right_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        ((left_ms, ()), (right_ms, ()))
    };

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][forced_common_reconcile] left_weights={} right_weights={} left_tsids={} right_tsids={} common_tsids={} left_tokens={} right_tokens={} common_tokens={} build_ms={:.3} left_remap_ms={:.3} right_remap_ms={:.3} total_ms={:.3}",
            left_weights.len(),
            right_weights.len(),
            left_id_map.num_tsids(),
            right_id_map.num_tsids(),
            common_id_map.num_tsids(),
            left_id_map.num_internal_tokens(),
            right_id_map.num_internal_tokens(),
            common_id_map.num_internal_tokens(),
            build_ms,
            left_ms,
            right_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

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

pub(super) fn remap_weights_into_existing_common(
    weights: &mut [&mut Weight],
    local_id_map: &InternalIdMap,
    common_id_map: &InternalIdMap,
) {
    if weights.is_empty() {
        return;
    }
    remap_weights_with_axes(
        weights,
        TokenAxisMap::Refinement { local: &local_id_map.tokenizer_states, common: &common_id_map.tokenizer_states },
        TokenAxisMap::Refinement { local: &local_id_map.vocab_tokens, common: &common_id_map.vocab_tokens },
        common_id_map.num_tsids() as usize,
        common_id_map.num_internal_tokens() as usize,
    );
}

pub(super) fn internal_id_map_refines(finer: &InternalIdMap, coarser: &InternalIdMap) -> bool {
    many_to_one_id_map_refines(&finer.tokenizer_states, &coarser.tokenizer_states, false)
        && many_to_one_id_map_refines(&finer.vocab_tokens, &coarser.vocab_tokens, true)
}

fn many_to_one_id_map_refines(
    finer: &ManyToOneIdMap,
    coarser: &ManyToOneIdMap,
    allow_unmapped: bool,
) -> bool {
    if finer.original_to_internal.len() < coarser.original_to_internal.len() {
        // Vocabulary maps are partial: an omitted trailing ID and an explicit
        // UNMAPPED entry have identical meaning. A prepared common partition
        // may trim such padding while a later artifact retains vocabulary size.
        // Do not accept the omission of any mapped ID, or of tokenizer states.
        if !allow_unmapped || coarser.original_to_internal[finer.original_to_internal.len()..]
            .iter().any(|&internal| internal != u32::MAX)
        {
            return false;
        }
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

pub(super) fn build_common_internal_id_map(inputs: &[&InternalIdMap]) -> InternalIdMap {
    build_common_internal_id_map_impl(inputs, false)
}

pub(super) fn build_common_internal_id_map_preserving_unmapped_tokenizer(
    inputs: &[&InternalIdMap],
) -> InternalIdMap {
    build_common_internal_id_map_impl(inputs, true)
}

fn build_common_internal_id_map_impl(
    inputs: &[&InternalIdMap],
    allow_unmapped_tokenizer: bool,
) -> InternalIdMap {
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

    let parallel_dimensions = std::env::var("GLRMASK_PARALLEL_COMMON_ID_DIMENSIONS")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let (tokenizer_states, vocab_tokens) = if parallel_dimensions {
        rayon::join(
            || {
                build_common_many_to_one_id_map(
                    inputs,
                    num_tokenizer_states,
                    |input| &input.tokenizer_states,
                    allow_unmapped_tokenizer,
                )
            },
            || {
                build_common_many_to_one_id_map(
                    inputs,
                    num_original_tokens,
                    |input| &input.vocab_tokens,
                    true,
                )
            },
        )
    } else {
        (
            build_common_many_to_one_id_map(
                inputs,
                num_tokenizer_states,
                |input| &input.tokenizer_states,
                allow_unmapped_tokenizer,
            ),
            build_common_many_to_one_id_map(
                inputs,
                num_original_tokens,
                |input| &input.vocab_tokens,
                true,
            ),
        )
    };

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
    let maps = inputs.iter().map(|input| project(input)).collect::<Vec<_>>();
    refine_observation_classes(&maps, num_originals, allow_unmapped, true)
}

/// Exact common partition of observations in the same original coordinate.
/// Lexicographic numbering preserves input-class locality; first-seen order
/// avoids sorting when a consumer has no such preference.
pub fn refine_observation_classes(
    inputs: &[&ManyToOneIdMap],
    num_originals: usize,
    allow_unmapped: bool,
    lexicographic: bool,
) -> ManyToOneIdMap {
    if lexicographic && let [left, right] = inputs {
        return build_common_many_to_one_id_map_pair(
            left, right, num_originals, allow_unmapped,
        );
    }
    if allow_unmapped && lexicographic && inputs.len() > 2 {
        return refine_sparse_memberships(inputs, num_originals, true, false).0;
    }

    let (mut result, signatures) = partition_observation_domain(
        (0..num_originals).map(|original| original as u32), inputs.len(),
        |original, key| {
            for (value, input) in key.iter_mut().zip(inputs) {
                *value = input.original_to_internal.get(original as usize)
                    .copied().unwrap_or(u32::MAX);
            }
        },
        Some(num_originals),
        if allow_unmapped { Unobserved::Omit } else { Unobserved::Merge },
    );
    if lexicographic {
        reorder_common_classes(signatures, &mut result);
    }
    result
}

#[derive(Clone, Copy)]
enum Unobserved {
    Omit,
    Merge,
    Singleton,
}

/// Partition an explicit ordered original domain by the same observation
/// signature used for dense artifact reconciliation.  An omitted original
/// is not equivalent to an unobserved member of the requested domain.
fn partition_observation_domain(
    originals: impl IntoIterator<Item = u32>,
    source_count: usize,
    mut observe: impl FnMut(u32, &mut [u32]),
    dense_len: Option<usize>,
    unobserved: Unobserved,
) -> (ManyToOneIdMap, FxHashMap<Box<[u32]>, u32>) {
    let mut signatures = FxHashMap::<Box<[u32]>, u32>::default();
    let mut key = vec![u32::MAX; source_count];
    let mut result = ManyToOneIdMap {
        original_to_internal: vec![u32::MAX; dense_len.unwrap_or(0)],
        internal_to_originals: Vec::new(),
        representative_original_ids: Vec::new(),
    };
    for original in originals {
        observe(original, &mut key);
        let absent = key.iter().all(|&value| value == u32::MAX);
        if absent && matches!(unobserved, Unobserved::Omit) {
            continue;
        }
        let singleton = absent && matches!(unobserved, Unobserved::Singleton);
        let existing = if singleton { None } else { signatures.get(key.as_slice()).copied() };
        let class = existing.unwrap_or_else(|| {
            let class = result.internal_to_originals.len() as u32;
            if !singleton {
                signatures.insert(key.clone().into_boxed_slice(), class);
            }
            result.internal_to_originals.push(Vec::new());
            result.representative_original_ids.push(original);
            class
        });
        if dense_len.is_some() {
            result.original_to_internal[original as usize] = class;
        }
        result.internal_to_originals[class as usize].push(original);
    }
    (result, signatures)
}

/// First-original ordering on an explicit sparse vocabulary domain.
/// Group-only sources are indexed only by actual memberships. Group-only
/// output merges the all-absent signature; conservative dense publication
/// can instead retain each unobserved original as a separate class.
pub fn refine_vocab_observations(
    inputs: &[ManyToOneIdMap],
    originals: impl IntoIterator<Item = u32>,
    dense_len: Option<usize>,
    merge_unobserved: bool,
) -> ManyToOneIdMap {
    let sparse = inputs.iter().map(|input| {
        (merge_unobserved || input.original_to_internal.is_empty()).then(|| {
            let mut membership = FxHashMap::default();
            for (class, members) in input.internal_to_originals.iter().enumerate() {
                for &original in members {
                    membership.insert(original, class as u32);
                }
            }
            membership
        })
    }).collect::<Vec<_>>();
    partition_observation_domain(
        originals, inputs.len(),
        |original, key| {
            for ((value, input), sparse) in key.iter_mut().zip(inputs).zip(&sparse) {
                *value = match sparse {
                    Some(membership) => membership.get(&original).copied(),
                    None => input.original_to_internal.get(original as usize).copied(),
                }.unwrap_or(u32::MAX);
            }
        },
        dense_len,
        if merge_unobserved { Unobserved::Merge } else { Unobserved::Singleton },
    ).0
}
/// A persistent signature prefix, interned once for each distinct
/// membership extension. An original carries only its current head ID.
struct Membership {
    previous: u32,
    source: u32,
    local_class: u32,
    size: usize,
    representative: u32,
    global: u32,
}

/// Refine sparse observations by actual inverse memberships, not a dense
/// original-by-source product. Each source extends a signature at most
/// once per original. Distinct signature prefixes share their history.
///
/// Source-local order puts single-owner classes first (contiguous token
/// offsets), followed by overlapping classes in first-original order.
/// Lexicographic order instead retains the ordinary common-map convention.
/// The optional fanout is a derived projection of the same membership
/// chains, not a second reconstruction of the partition.
pub fn refine_sparse_memberships(
    inputs: &[&ManyToOneIdMap],
    original_count: usize,
    lexicographic: bool,
    build_fanout: bool,
) -> (ManyToOneIdMap, Option<Vec<Vec<Vec<u32>>>>) {
    const ABSENT: u32 = u32::MAX;
    let mut heads = vec![ABSENT; original_count];
    let mut nodes = Vec::<Membership>::new();
    let mut extensions = FxHashMap::<u32, u32>::default();
    for (source, input) in inputs.iter().enumerate() {
        let source = u32::try_from(source).expect("too many observation sources");
        for (local_class, originals) in input.internal_to_originals.iter().enumerate() {
            let Some(&first) = originals.first() else { continue };
            let local_class = u32::try_from(local_class).expect("too many observation classes");
            let first_previous = heads[first as usize];
            let mut extend = |previous| {
                let id = u32::try_from(nodes.len()).expect("observation signature arena overflow");
                assert_ne!(id, ABSENT, "observation signature arena exhausted");
                nodes.push(Membership {
                    previous, source, local_class, size: 0,
                    representative: ABSENT, global: ABSENT,
                });
                id
            };
            let first_head = extend(first_previous);
            extensions.clear();
            for &original in originals {
                let head = &mut heads[original as usize];
                *head = if *head == first_previous {
                    first_head
                } else {
                    *extensions.entry(*head).or_insert_with(|| extend(*head))
                };
            }
        }
    }

    // Only signatures present at the end are classes. Earlier prefixes
    // survive solely as compact witnesses for their descendants' fanout.
    for (original, &head) in heads.iter().enumerate() {
        if head != ABSENT {
            let node = &mut nodes[head as usize];
            node.size += 1;
            if node.representative == ABSENT {
                node.representative = original as u32;
            }
        }
    }

    let order = if lexicographic {
        let mut order = nodes.iter().enumerate()
            .filter_map(|(id, node)| (node.size != 0).then_some(id as u32))
            .collect::<Vec<_>>();
        order.sort_unstable_by(|&left, &right| {
            let left = nodes[left as usize].representative as usize;
            let right = nodes[right as usize].representative as usize;
            for input in inputs {
                let left = input.original_to_internal.get(left).copied().unwrap_or(ABSENT);
                let right = input.original_to_internal.get(right).copied().unwrap_or(ABSENT);
                let order = left.cmp(&right);
                if !order.is_eq() { return order }
            }
            std::cmp::Ordering::Equal
        });
        order
    } else {
        // Single-owner nodes were created in source/local-class order.
        let mut order = nodes.iter().enumerate()
            .filter_map(|(id, node)| (node.size != 0 && node.previous == ABSENT).then_some(id as u32))
            .collect::<Vec<_>>();
        for (original, &head) in heads.iter().enumerate() {
            if head != ABSENT {
                let node = &nodes[head as usize];
                if node.previous != ABSENT && node.representative as usize == original {
                    order.push(head);
                }
            }
        }
        order
    };

    let mut local_to_global = build_fanout.then(|| inputs.iter()
        .map(|input| vec![Vec::<u32>::new(); input.internal_to_originals.len()])
        .collect::<Vec<_>>());
    let mut groups = Vec::with_capacity(order.len());
    let mut representatives = Vec::with_capacity(order.len());
    for head in order {
        let node = &mut nodes[head as usize];
        let global = groups.len() as u32;
        node.global = global;
        groups.push(Vec::with_capacity(node.size));
        representatives.push(node.representative);
        if let Some(fanout) = &mut local_to_global {
            let mut membership = head;
            while membership != ABSENT {
                let node = &nodes[membership as usize];
                fanout[node.source as usize][node.local_class as usize].push(global);
                membership = node.previous;
            }
        }
    }
    for (original, head) in heads.iter_mut().enumerate() {
        if *head != ABSENT {
            *head = nodes[*head as usize].global;
            groups[*head as usize].push(original as u32);
        }
    }

    (ManyToOneIdMap {
        original_to_internal: heads,
        internal_to_originals: groups,
        representative_original_ids: representatives,
    }, local_to_global)
}

fn build_common_many_to_one_id_map_pair(
    left: &ManyToOneIdMap,
    right: &ManyToOneIdMap,
    num_originals: usize,
    allow_unmapped: bool,
) -> ManyToOneIdMap {
    if !allow_unmapped {
        return build_common_many_to_one_id_map_pair_hashed(left, right, num_originals, false);
    }
    let direct = build_common_many_to_one_id_map_pair_direct_sparse(left, right, num_originals);
    if std::env::var_os("GLRMASK_VALIDATE_DIRECT_SPARSE_COMMON_PAIR_MAP").is_some() {
        let reference = build_common_many_to_one_id_map_pair_hashed(left, right, num_originals, true);
        assert_eq!(direct.original_to_internal, reference.original_to_internal);
        assert_eq!(direct.internal_to_originals, reference.internal_to_originals);
        assert_eq!(direct.representative_original_ids, reference.representative_original_ids);
    }
    direct
}

fn build_common_many_to_one_id_map_pair_direct_sparse(
    left: &ManyToOneIdMap,
    right: &ManyToOneIdMap,
    num_originals: usize,
) -> ManyToOneIdMap {
    let right_classes = right.internal_to_originals.len();
    let mut original_to_internal = vec![u32::MAX; num_originals];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<u32>::new();

    // Composite pair order is lexicographic `(left, right)`, with the unmapped
    // sentinel sorting after every real class. Emit exactly that order directly.
    let mut by_right = (0..right_classes).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
    let mut touched_right = Vec::<usize>::new();
    let mut left_only = Vec::<u32>::new();

    let mut emit = |mut originals: Vec<u32>| {
        if originals.is_empty() {
            return;
        }
        // Most maps are already canonical and this branch is just one linear
        // check. Some independently/parallel-built artifacts can arrive with
        // an equivalent class whose inverse list is not in strict ID order.
        // The common-map relation is set-valued, so canonicalize that rare
        // input locally instead of making the direct sparse fast path depend
        // on incidental inverse-list construction order.
        if !originals.windows(2).all(|pair| pair[0] < pair[1]) {
            originals.sort_unstable();
            originals.dedup();
        }
        let class = internal_to_originals.len() as u32;
        for &original in &originals {
            original_to_internal[original as usize] = class;
        }
        representatives.push(originals[0]);
        internal_to_originals.push(originals);
    };

    for left_originals in &left.internal_to_originals {
        touched_right.clear();
        left_only.clear();
        for &original in left_originals {
            let right_internal = right
                .original_to_internal
                .get(original as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if right_internal == u32::MAX {
                left_only.push(original);
            } else {
                let slot = &mut by_right[right_internal as usize];
                if slot.is_empty() {
                    touched_right.push(right_internal as usize);
                }
                slot.push(original);
            }
        }
        touched_right.sort_unstable();
        for right_class in touched_right.drain(..) {
            emit(std::mem::take(&mut by_right[right_class]));
        }
        emit(std::mem::take(&mut left_only));
    }

    // `(MAX, right)` classes follow every `(left, *)` class.
    for right_originals in &right.internal_to_originals {
        let mut right_only = Vec::new();
        for &original in right_originals {
            let left_internal = left
                .original_to_internal
                .get(original as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if left_internal == u32::MAX {
                right_only.push(original);
            }
        }
        emit(right_only);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

fn build_common_many_to_one_id_map_pair_hashed(
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

    let mut result = ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    };
    reorder_common_classes(composite_to_class, &mut result);
    result
}

fn reorder_common_classes<K: Ord>(
    composite_to_class: impl IntoIterator<Item = (K, u32)>,
    result: &mut ManyToOneIdMap,
) {
    let count = result.internal_to_originals.len();
    if count <= 1 {
        return;
    }
    let mut sorted = composite_to_class.into_iter().collect::<Vec<_>>();
    sorted.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let mut perm = vec![0u32; count];
    for (new_id, (_, old_id)) in sorted.into_iter().enumerate() {
        perm[old_id as usize] = new_id as u32;
    }
    result.relabel_classes(&perm, count);
}

/// Project a local observation coordinate onto an exact refinement. One
/// representative per nonempty common class suffices; destinations are
/// already unique and ordered by common class ID.
pub fn local_to_refined_classes(
    local: &ManyToOneIdMap,
    common: &ManyToOneIdMap,
) -> Vec<Vec<u32>> {
    let mut fanout = vec![Vec::new(); local.internal_to_originals.len()];
    for (destination, originals) in common.internal_to_originals.iter().enumerate() {
        let Some(&representative) = originals.first() else { continue };
        let local_class = local.original_to_internal.get(representative as usize)
            .copied().unwrap_or(u32::MAX);
        #[cfg(debug_assertions)]
        for &original in originals {
            debug_assert_eq!(local.original_to_internal.get(original as usize)
                .copied().unwrap_or(u32::MAX), local_class,
                "destination must refine the local observation classes");
        }
        if local_class != u32::MAX {
            fanout[local_class as usize].push(destination as u32);
        }
    }
    fanout
}

#[cfg(test)]
fn build_local_to_common_tsid_map(
    local: &InternalIdMap,
    common: &InternalIdMap,
) -> Vec<Vec<u32>> {
    local_to_refined_classes(&local.tokenizer_states, &common.tokenizer_states)
}

#[cfg(test)]
fn build_local_to_common_token_map_from_common_classes(
    local: &InternalIdMap,
    common: &InternalIdMap,
) -> Vec<Vec<u32>> {
    local_to_refined_classes(&local.vocab_tokens, &common.vocab_tokens)
}

/// The maps used by the fast remappers are injective or disjoint expansions,
/// so sorted destination runs cannot overlap. Coalesce the same adjacent runs
/// as CompactRangeBuilder, then build the tree from the ordered stream once.
/// The final interning boundary and identifier numbering remain unchanged.
fn weight_from_sorted_disjoint_runs(
    runs: impl IntoIterator<Item = (u32, u32, SharedTokenSet)>,
) -> Weight {
    let runs = runs.into_iter();
    // The ordinary builder is cheaper than an extra buffer for tiny weights.
    if runs.size_hint().1.is_some_and(|maximum| maximum <= 16) {
        return Weight::from_tsid_runs_shared(runs);
    }
    let mut ranges: Vec<(std::ops::RangeInclusive<u32>, SharedTokenSet)> = Vec::new();
    for (start, end, tokens) in runs {
        if tokens.is_empty() {
            continue;
        }
        if let Some((previous, previous_tokens)) = ranges.last_mut()
            && previous.end().checked_add(1) == Some(start)
            && (Arc::ptr_eq(previous_tokens, &tokens)
                || previous_tokens.as_ref() == tokens.as_ref())
        {
            let first = *previous.start();
            *previous = first..=end;
        } else {
            ranges.push((start..=end, tokens));
        }
    }
    let map = RangeMapBlaze::from_sorted_disjoint_map(CheckSortedDisjointMap::new(
        ranges.iter().map(|(range, tokens)| (range.clone(), tokens)),
    ));
    finalize_weight_map(map)
}

/// A token-coordinate image can be supplied without allocating one vector
/// per token for the common disjoint-partition offset case.
#[derive(Clone, Copy)]
pub enum TokenAxisMap<'a> {
    Explicit(&'a [Vec<u32>]),
    /// Proven injection, with u32::MAX denoting an absent local class.
    Injection(&'a [u32]),
    /// Destination classes refine the source partition (including absent originals).
    Refinement { local: &'a ManyToOneIdMap, common: &'a ManyToOneIdMap },
    Offset { offset: u32, local_token_count: u32 },
}

/// One prepared image of a local coordinate. Disjoint images allow a weight
/// to be built directly from runs; overlapping images require token union.
enum AxisImage<'a> {
    Affine { offset: u32, count: usize },
    Contiguous { first: Vec<u32>, last: Vec<u32> },
    Singleton { destinations: Cow<'a, [u32]>, monotone: bool },
    Disjoint { offsets: Vec<usize>, runs: Vec<(u32, u32)>, monotone: bool },
    Overlapping { fanout: &'a [Vec<u32>], destination_count: usize },
}

impl<'a> AxisImage<'a> {
    fn explicit(fanout: &'a [Vec<u32>], destination_count: usize) -> Self {
        if let Some(offset) = fanout.first().and_then(|row| row.first()).copied() {
            if fanout.iter().enumerate().all(|(local, row)| {
                offset.checked_add(local as u32).is_some_and(|target| {
                    (target as usize) < destination_count && row.as_slice() == [target]
                })
            }) {
                return Self::Affine { offset, count: fanout.len() };
            }
        }

        let mut seen = vec![false; destination_count];
        let mut offsets = Vec::with_capacity(fanout.len() + 1);
        let mut runs = Vec::<(u32, u32)>::new();
        let mut monotone = true;
        let mut adjacent = true;
        let mut singleton = true;
        let mut previous: Option<u32> = None;
        for row in fanout {
            offsets.push(runs.len());
            singleton &= row.len() <= 1;
            let row_start = runs.len();
            for &target in row {
                if target as usize >= destination_count
                    || std::mem::replace(&mut seen[target as usize], true)
                {
                    return Self::Overlapping { fanout, destination_count };
                }
                if let Some(prev) = previous {
                    monotone &= prev < target;
                    adjacent &= prev.checked_add(1) == Some(target);
                }
                previous = Some(target);
                if runs.len() > row_start {
                    let (_, end) = runs.last_mut().unwrap();
                    if end.checked_add(1) == Some(target) {
                        *end = target;
                        continue;
                    }
                }
                runs.push((target, target));
            }
        }
        offsets.push(runs.len());
        if adjacent && monotone {
            let mut first = vec![u32::MAX; fanout.len()];
            let mut last = vec![u32::MAX; fanout.len()];
            let mut next = u32::MAX;
            for local in (0..fanout.len()).rev() {
                if offsets[local] != offsets[local + 1] {
                    next = runs[offsets[local]].0;
                }
                first[local] = next;
            }
            let mut prev = u32::MAX;
            for local in 0..fanout.len() {
                if offsets[local] != offsets[local + 1] {
                    prev = runs[offsets[local + 1] - 1].1;
                }
                last[local] = prev;
            }
            Self::Contiguous { first, last }
        } else if singleton {
            Self::Singleton {
                destinations: fanout.iter().map(|row| {
                    row.first().copied().unwrap_or(u32::MAX)
                }).collect::<Vec<_>>().into(),
                monotone,
            }
        } else {
            Self::Disjoint { offsets, runs, monotone }
        }
    }

    /// Construct the disjoint image directly from the common partition.
    /// A common class belongs to at most one local class. Visit destination
    /// IDs in order, counting maximal runs rather than expanding memberships.
    fn refinement(local: &ManyToOneIdMap, common: &ManyToOneIdMap) -> Self {
        let n = local.internal_to_originals.len();
        let sparse = local.original_to_internal.is_empty().then(|| {
            local.internal_to_originals.iter().enumerate().flat_map(|(class, originals)|
                originals.iter().map(move |&original| (original, class as u32))
            ).collect::<FxHashMap<_, _>>()
        });
        let local_class = |original: u32| {
            if let Some(sparse) = &sparse {
                sparse.get(&original).copied().unwrap_or(u32::MAX)
            } else {
                local.original_to_internal.get(original as usize).copied().unwrap_or(u32::MAX)
            }
        };
        let membership = |originals: &[u32]| {
            let Some(&representative) = originals.first() else { return u32::MAX };
            let class = local_class(representative);
            #[cfg(debug_assertions)]
            for &original in originals {
                debug_assert_eq!(local_class(original), class,
                    "destination must refine the local observation classes");
            }
            class
        };
        let mut first = vec![u32::MAX; n];
        let mut last = vec![u32::MAX; n];
        let mut counts = vec![0usize; n];
        let mut previous = None::<(usize, u32)>;
        let mut monotone = true;
        let mut adjacent = true;
        for (destination, originals) in common.internal_to_originals.iter().enumerate() {
            let class = membership(originals);
            if class == u32::MAX { continue; }
            let class = class as usize;
            let destination = destination as u32;
            if let Some((prev_class, prev_dest)) = previous {
                monotone &= prev_class <= class;
                adjacent &= prev_dest.checked_add(1) == Some(destination);
            }
            previous = Some((class, destination));
            if first[class] == u32::MAX {
                first[class] = destination;
                counts[class] = 1;
            } else if last[class].checked_add(1) != Some(destination) {
                counts[class] += 1;
            }
            last[class] = destination;
        }
        let singleton = first == last;
        if singleton {
            if let Some(&offset) = first.first() {
                if offset != u32::MAX && first.iter().enumerate().all(|(i, &target)|
                    offset.checked_add(i as u32) == Some(target))
                {
                    return Self::Affine { offset, count: n };
                }
            }
        }
        if adjacent && monotone {
            let mut next = u32::MAX;
            for start in first.iter_mut().rev() {
                if *start != u32::MAX { next = *start; }
                *start = next;
            }
            let mut prev = u32::MAX;
            for end in &mut last {
                if *end != u32::MAX { prev = *end; }
                *end = prev;
            }
            return Self::Contiguous { first, last };
        }
        if singleton {
            return Self::Singleton { destinations: Cow::Owned(first), monotone };
        }
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0usize);
        for count in counts {
            offsets.push(offsets.last().unwrap() + count);
        }
        let mut runs = vec![(u32::MAX, u32::MAX); *offsets.last().unwrap()];
        let mut cursor = offsets[..n].to_vec();
        for (destination, originals) in common.internal_to_originals.iter().enumerate() {
            let class = membership(originals);
            if class == u32::MAX { continue; }
            let class = class as usize;
            let destination = destination as u32;
            let pos = cursor[class];
            if pos > offsets[class] && runs[pos - 1].1.checked_add(1) == Some(destination) {
                runs[pos - 1].1 = destination;
            } else {
                runs[pos] = (destination, destination);
                cursor[class] += 1;
            }
        }
        Self::Disjoint { offsets, runs, monotone }
    }

    fn tokens(map: TokenAxisMap<'a>, destination_count: usize) -> Self {
        match map {
            TokenAxisMap::Explicit(fanout) => Self::explicit(fanout, destination_count),
            TokenAxisMap::Refinement { local, common } => {
                assert_eq!(common.internal_to_originals.len(), destination_count);
                Self::refinement(local, common)
            }
            TokenAxisMap::Injection(destinations) => {
                if let Some(&offset) = destinations.first() {
                    if offset != u32::MAX && destinations.iter().enumerate().all(|(local, &target)| {
                        (target as usize) < destination_count && offset.checked_add(local as u32) == Some(target)
                    }) {
                        return Self::Affine { offset, count: destinations.len() };
                    }
                }
                let mut previous = None;
                let mut monotone = true;
                for &target in destinations.iter().filter(|&&target| target != u32::MAX) {
                    assert!((target as usize) < destination_count, "token injection exceeds destination");
                    monotone &= previous.is_none_or(|last| last < target);
                    previous = Some(target);
                }
                Self::Singleton { destinations: Cow::Borrowed(destinations), monotone }
            }
            TokenAxisMap::Offset { offset, local_token_count } => {
                let count = (local_token_count as usize)
                    .min(destination_count.saturating_sub(offset as usize));
                Self::Affine { offset, count }
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Affine { count, .. } => *count,
            Self::Contiguous { first, .. } => first.len(),
            Self::Singleton { destinations, .. } => destinations.len(),
            Self::Disjoint { offsets, .. } => offsets.len() - 1,
            Self::Overlapping { fanout, .. } => fanout.len(),
        }
    }

    fn identity(&self, destination_count: usize) -> bool {
        matches!(self, Self::Affine { offset: 0, count } if *count == destination_count)
    }

    fn monotone(&self) -> bool {
        match self {
            Self::Affine { .. } | Self::Contiguous { .. } => true,
            Self::Singleton { monotone, .. } | Self::Disjoint { monotone, .. } => *monotone,
            Self::Overlapping { .. } => false,
        }
    }

    fn is_disjoint(&self) -> bool {
        !matches!(self, Self::Overlapping { .. })
    }

    #[inline]
    fn visit_runs(&self, start: u32, end: u32, mut visit: impl FnMut(u32, u32)) {
        let start = start as usize;
        let end = (end as usize).min(self.len().saturating_sub(1));
        if self.len() == 0 || start > end {
            return;
        }
        match self {
            Self::Affine { offset, .. } => visit(*offset + start as u32, *offset + end as u32),
            Self::Contiguous { first, last } => {
                let (a, b) = (first[start], last[end]);
                if a != u32::MAX && b != u32::MAX && a <= b {
                    visit(a, b);
                }
            }
            Self::Singleton { destinations, .. } => {
                for &target in &destinations[start..=end] {
                    if target != u32::MAX {
                        visit(target, target);
                    }
                }
            }
            Self::Disjoint { offsets, runs, .. } => {
                for &(a, b) in &runs[offsets[start]..offsets[end + 1]] {
                    visit(a, b);
                }
            }
            Self::Overlapping { fanout, destination_count } => {
                for row in &fanout[start..=end] {
                    for &target in row {
                        if (target as usize) < *destination_count {
                            visit(target, target);
                        }
                    }
                }
            }
        }
    }

    fn image(&self, ranges: impl IntoIterator<Item = (u32, u32)>) -> RangeSetBlaze<u32> {
        let mut mapped = Vec::new();
        for (start, end) in ranges {
            self.visit_runs(start, end, |a, b| mapped.push(a..=b));
        }
        RangeSetBlaze::from_iter(mapped)
    }

    fn full_image(&self) -> RangeSetBlaze<u32> {
        self.image(std::iter::once((0, u32::MAX)))
    }
}

fn remap_token_set(
    tokens: &SharedTokenSet,
    image: &AxisImage<'_>,
    range_cache: Option<&FxHashMap<(u32, u32), RangeSetBlaze<u32>>>,
) -> SharedTokenSet {
    let result = if let Some(cache) = range_cache {
        RangeSetBlaze::from_iter(tokens.ranges().flat_map(|r| {
            cache[&(*r.start(), *r.end())].ranges()
        }))
    } else {
        image.image(tokens.ranges().map(|r| (*r.start(), *r.end())))
    };
    shared_rangeset(result)
}

const PARALLEL_UNIQUE_WEIGHT_THRESHOLD: usize = 128;

fn map_batch<T: Sync, R: Send>(values: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    if values.len() >= PARALLEL_UNIQUE_WEIGHT_THRESHOLD && rayon::current_num_threads() > 1 {
        values.par_iter().map(f).collect()
    } else {
        values.iter().map(f).collect()
    }
}

/// Lift a single rectangular relation through both coordinate images.
/// Nonoverlapping state images need no union scratch. For overlap, collect
/// the exact union at each destination before building the sorted runs.
fn remap_relation(
    weight: &Weight,
    states: &AxisImage<'_>,
    tokens: &AxisImage<'_>,
    common_tsid_count: usize,
    identity_states: bool,
    mapped_sets: Option<&FxHashMap<usize, SharedTokenSet>>,
) -> Weight {
    if weight.is_empty() {
        return weight.clone();
    }
    if weight.is_full() {
        let all_tokens = shared_rangeset(tokens.full_image());
        if all_tokens.is_empty() {
            return Weight::empty();
        }
        return weight_from_sorted_disjoint_runs(states.full_image().ranges().map(|r| {
            (*r.start(), *r.end(), Arc::clone(&all_tokens))
        }));
    }

    let mapped = |source: &SharedTokenSet| {
        mapped_sets.map_or_else(|| remap_token_set(source, tokens, None), |sets| {
            Arc::clone(&sets[&(Arc::as_ptr(source) as usize)])
        })
    };
    if identity_states {
        return weight.remap_token_sets_preserving_tsid_ranges(mapped);
    }
    if states.is_disjoint() {
        let mut runs = Vec::new();
        for (start, end, source) in weight.range_entries() {
            let set = mapped(source);
            if !set.is_empty() {
                states.visit_runs(start, end, |a, b| runs.push((a, b, Arc::clone(&set))));
            }
        }
        if !states.monotone() {
            runs.sort_unstable_by_key(|(start, _, _)| *start);
        }
        return weight_from_sorted_disjoint_runs(runs);
    }

    let mut rows = vec![None::<SharedTokenSet>; common_tsid_count];
    for (start, end, source) in weight.range_entries() {
        let set = mapped(source);
        if set.is_empty() {
            continue;
        }
        states.visit_runs(start, end, |a, b| {
            for row in &mut rows[a as usize..=b as usize] {
                match row {
                    Some(existing) if !Arc::ptr_eq(existing, &set) && **existing != *set => {
                        *existing = shared_rangeset(existing.as_ref() | set.as_ref());
                    }
                    None => *row = Some(Arc::clone(&set)),
                    _ => {}
                }
            }
        });
    }
    weight_from_sorted_disjoint_runs(rows.into_iter().enumerate().filter_map(|(q, set)| {
        set.map(|set| (q as u32, q as u32, set))
    }))
}

pub fn remap_weights_with_maps(
    weights: &mut [&mut Weight],
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
    common_token_count: usize,
) {
    remap_weights_with_token_axis(
        weights, local_to_common_tsids, TokenAxisMap::Explicit(local_to_common_tokens),
        common_tsid_count, common_token_count,
    );
}

pub fn remap_weights_with_token_axis(
    weights: &mut [&mut Weight],
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: TokenAxisMap<'_>,
    common_tsid_count: usize,
    common_token_count: usize,
) {
    remap_weights_with_axes(
        weights, TokenAxisMap::Explicit(local_to_common_tsids), local_to_common_tokens,
        common_tsid_count, common_token_count,
    );
}

/// Lift both dimensions directly from their certified coordinate images.
pub fn remap_weights_with_axes(
    weights: &mut [&mut Weight],
    local_to_common_tsids: TokenAxisMap<'_>,
    local_to_common_tokens: TokenAxisMap<'_>,
    common_tsid_count: usize,
    common_token_count: usize,
) {
    if weights.is_empty() {
        return;
    }
    let profiling = std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some();
    let started = profiling.then(Instant::now);
    let states = AxisImage::tokens(local_to_common_tsids, common_tsid_count);
    let tokens = AxisImage::tokens(local_to_common_tokens, common_token_count);
    let identity_states = states.identity(common_tsid_count);
    let identity_tokens = tokens.identity(common_token_count);
    if identity_states && identity_tokens {
        return;
    }

    // Both source objects stay alive until publication, so pointer keys are
    // exact identities. Deduplicate the work across the entire artifact.
    let mut by_weight = FxHashMap::<usize, usize>::default();
    let mut unique = Vec::<Weight>::new();
    let indices = weights.iter().map(|weight| {
        *by_weight.entry(weight.ptr_key()).or_insert_with(|| {
            let index = unique.len();
            unique.push((**weight).clone());
            index
        })
    }).collect::<Vec<_>>();
    let mut by_set = FxHashMap::<usize, SharedTokenSet>::default();
    for weight in &unique {
        for (_, _, set) in weight.range_entries() {
            by_set.entry(Arc::as_ptr(set) as usize).or_insert_with(|| Arc::clone(set));
        }
    }
    let sources = by_set.into_iter().collect::<Vec<_>>();
    let range_cache = if matches!(tokens, AxisImage::Disjoint { .. } | AxisImage::Singleton { .. })
        && sources.len() >= PARALLEL_UNIQUE_WEIGHT_THRESHOLD
    {
        let total_tokens = sources.iter().map(|(_, set)| set.len() as usize).sum::<usize>();
        let total_ranges = sources.iter().map(|(_, set)| set.ranges().count()).sum::<usize>();
        if total_tokens >= 8192 && total_ranges.saturating_mul(4) < total_tokens {
            let ranges = sources.iter().flat_map(|(_, set)| {
                set.ranges().map(|r| (*r.start(), *r.end()))
            }).collect::<FxHashSet<_>>().into_iter().collect::<Vec<_>>();
            Some(map_batch(&ranges, |&(a, b)| {
                ((a, b), tokens.image(std::iter::once((a, b))))
            }).into_iter().collect::<FxHashMap<_, _>>())
        } else {
            None
        }
    } else {
        None
    };
    let mapped_sets = map_batch(&sources, |(key, set)| {
        let image = if identity_tokens {
            Arc::clone(set)
        } else {
            remap_token_set(set, &tokens, range_cache.as_ref())
        };
        (*key, image)
    }).into_iter().collect::<FxHashMap<_, _>>();
    let remapped = map_batch(&unique, |weight| {
        remap_relation(weight, &states, &tokens, common_tsid_count, identity_states, Some(&mapped_sets))
    });

    if std::env::var_os("GLRMASK_VALIDATE_GLOBAL_TOKEN_REMAP_CACHE").is_some() {
        for (source, result) in unique.iter().zip(&remapped) {
            assert_eq!(*result, remap_relation(
                source, &states, &tokens, common_tsid_count, identity_states, None,
            ), "global token remap cache changed a weight");
        }
    }
    for (weight, index) in weights.iter_mut().zip(indices) {
        **weight = remapped[index].clone();
    }
    if let Some(started) = started {
        eprintln!(
            "[glrmask/profile][token_run_reconcile] weights={} unique_weights={} unique_token_sets={} identity_tsids={} identity_tokens={} total_ms={:.3}",
            weights.len(), unique.len(), mapped_sets.len(), identity_states, identity_tokens,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
}
#[cfg(test)]
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
// Tests intentionally build one-range sets; clippy's suggested rewrite would
// replace the range element with the individual range members.
#[allow(clippy::single_range_in_vec_init)]
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
            .raw_range_values()
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

    #[test]
    fn direct_partition_image_matches_expanded_memberships() {
        let local = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            vec![1, 0, u32::MAX, 1, 0, 3, 3, 1], 5,
        );
        let common = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            vec![0, 1, 2, 3, 4, 5, 5, 6], 7,
        );
        let explicit = local_to_refined_classes(&local, &common);
        let mut sparse = local.clone();
        sparse.original_to_internal.clear();
        for source in [&local, &sparse] {
            let image = AxisImage::refinement(source, &common);
            for start in 0..5 {
                for end in start..5 {
                    let expected = RangeSetBlaze::from_iter(
                        explicit[start..=end].iter().flatten().copied(),
                    );
                    assert_eq!(image.image([(start as u32, end as u32)]), expected);
                }
            }
        }
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
            common_id_map.num_internal_tokens() as usize,
        );
        remap_weights_with_maps(
            &mut [right_weight],
            &right_tsid_map,
            &right_token_map,
            common_id_map.num_tsids() as usize,
            common_id_map.num_internal_tokens() as usize,
        );

        *left_id_map = common_id_map.clone();
        *right_id_map = common_id_map;
    }

    #[test]
    fn sparse_unmapped_pair_common_map_matches_hashed_pair_order() {
        for (left, right, num_originals, allow_unmapped) in [
            (
                map(vec![0, 0, 1, 1, 2, 2], 3),
                map(vec![0, 1, 0, 1, 2, 2], 3),
                6,
                false,
            ),
            (
                map(vec![0, u32::MAX, 1, 1, u32::MAX, 2], 3),
                map(vec![u32::MAX, 0, 0, 1, u32::MAX, 1], 2),
                6,
                true,
            ),
        ] {
            let hashed = build_common_many_to_one_id_map_pair_hashed(
                &left,
                &right,
                num_originals,
                allow_unmapped,
            );
            if allow_unmapped {
                let sparse = build_common_many_to_one_id_map_pair_direct_sparse(
                    &left, &right, num_originals,
                );
                let refined = refine_sparse_memberships(&[&left, &right], num_originals, true, true).0;
                assert_eq!(refined.original_to_internal, hashed.original_to_internal);
                assert_eq!(sparse.original_to_internal, hashed.original_to_internal);
                assert_eq!(sparse.internal_to_originals, hashed.internal_to_originals);
                assert_eq!(
                    sparse.representative_original_ids,
                    hashed.representative_original_ids,
                );
            }
        }
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
        let mut fast = weight.clone();
        remap_weights_with_maps(&mut [&mut fast], &tsid_map, &token_map, 4, 4);

        assert_eq!(entries_key(&fast), entries_key(&general));
    }

    #[test]
    fn bulk_sorted_weight_builder_matches_incremental_builder() {
        let mut seed = 0x5a17_6943u32;
        for count in [0, 1, 2, 15, 16, 17, 32, 128, 257] {
            for _ in 0..16 {
                let mut start = 0u32;
                let mut runs = Vec::new();
                for i in 0..count {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let end = start + (seed >> 8) % 4;
                    let tokens = if i % 11 == 0 {
                        RangeSetBlaze::new()
                    } else {
                        // Equal values in independent Arcs must coalesce too.
                        RangeSetBlaze::from_iter([((i / 3) % 7) as u32..=((i / 3) % 7 + 2) as u32])
                    };
                    runs.push((start, end, Arc::new(tokens)));
                    start = end + 1 + (seed >> 16) % 3;
                }
                let reference = Weight::from_tsid_ranges_shared(runs.clone());
                let candidate = weight_from_sorted_disjoint_runs(runs);
                assert_eq!(entries_key(&candidate), entries_key(&reference));
                assert_eq!(candidate, reference);
            }
        }
        let tokens = Arc::new(RangeSetBlaze::from_iter([1..=3]));
        let runs = (0..32).map(|i| {
            let point = u32::MAX - 64 + i * 2;
            (point, point, Arc::clone(&tokens))
        }).collect::<Vec<_>>();
        assert_eq!(weight_from_sorted_disjoint_runs(runs.clone()),
            Weight::from_tsid_ranges_shared(runs));
    }

    #[test]
    fn shared_token_precomputation_matches_uncached_projection() {
        let weight = Weight::from_tsid_ranges_shared([
            (0, 1, Arc::new(RangeSetBlaze::from_iter([0..=1]))),
            (2, 3, Arc::new(RangeSetBlaze::from_iter([1..=1, 3..=3]))),
        ]);
        let tokens = vec![vec![3], vec![1], vec![0], vec![2]];
        for tsids in [
            vec![vec![2], vec![0], vec![3], vec![1]],
            vec![vec![3, 4], vec![], vec![0, 1, 2], vec![5]],
            vec![vec![0], vec![0], vec![1], vec![2]],
        ] {
            let count = tsids.iter().flatten().copied().max().unwrap() as usize + 1;
            let mut actual = weight.clone();
            remap_weights_with_maps(&mut [&mut actual], &tsids, &tokens, count, 4);
            assert_eq!(actual, remap_weight_general(&weight, &tsids, &tokens, count));
        }
    }

    #[test]
    fn contiguous_disjoint_tsid_range_remap_matches_general() {
        let tokens_a = Arc::new(RangeSetBlaze::from_iter([0..=1]));
        let tokens_b = Arc::new(RangeSetBlaze::from_iter([2..=3]));
        let weight = Weight::from_tsid_ranges_shared([
            (0, 1, Arc::clone(&tokens_a)),
            (3, 3, Arc::clone(&tokens_b)),
        ]);
        let tsid_map = vec![vec![0, 1], vec![2, 3], vec![], vec![4, 5]];
        let token_map = (0..4).map(|token| vec![token]).collect::<Vec<_>>();
        assert!(matches!(AxisImage::explicit(&tsid_map, 6), AxisImage::Contiguous { .. }));
        let general = remap_weight_general(&weight, &tsid_map, &token_map, 6);
        let mut fast = weight.clone();
        remap_weights_with_maps(&mut [&mut fast], &tsid_map, &token_map, 6, 4);

        assert_eq!(entries_key(&fast), entries_key(&general));
        assert_eq!(
            entries_key(&fast)
                .into_iter()
                .map(|(start, end, _)| (start, end))
                .collect::<Vec<_>>(),
            vec![(0, 3), (4, 5)],
        );
    }

    #[test]
    fn identity_tsid_remap_matches_general_with_split_token_class() {
        let weight = Weight::from_tsid_ranges_shared([
            (
                0,
                2,
                Arc::new(RangeSetBlaze::from_iter([0..=1, 3..=3])),
            ),
            (5, 9, Arc::new(RangeSetBlaze::from_iter([2..=2]))),
        ]);
        let tsid_map = (0..10).map(|tsid| vec![tsid]).collect::<Vec<_>>();
        let token_map = vec![vec![0], vec![1, 2, 3], vec![4], vec![5]];
        assert!(AxisImage::explicit(&tsid_map, 10).identity(10));
        let general = remap_weight_general(&weight, &tsid_map, &token_map, 10);
        let mut fast = weight.clone();
        remap_weights_with_maps(&mut [&mut fast], &tsid_map, &token_map, 10, 6);

        assert_eq!(entries_key(&fast), entries_key(&general));
        assert_eq!(
            entries_key(&fast)
                .into_iter()
                .map(|(start, end, _)| (start, end))
                .collect::<Vec<_>>(),
            vec![(0, 2), (5, 9)],
            "identity TSID remapping must preserve source ranges",
        );
    }

    #[test]
    fn disjoint_run_token_remap_matches_general_for_split_reordered_ids() {
        let tokens = Arc::new(RangeSetBlaze::from_iter([0..=2, 4..=4]));
        let token_map = vec![vec![4, 5], vec![0, 1], vec![2, 3], vec![], vec![7, 8, 9]];
        let run_map = AxisImage::explicit(&token_map, 10);
        assert!(!run_map.destination_order_is_monotone);

        let fast = remap_token_set(&tokens, &run_map, None);
        let general = shared_rangeset(RangeSetBlaze::from_iter(tokens.iter().flat_map(|id| token_map[id as usize].iter().copied())));

        assert_eq!(fast.as_ref(), general.as_ref());
        assert_eq!(
            fast.ranges()
                .map(|range| (*range.start(), *range.end()))
                .collect::<Vec<_>>(),
            vec![(0, 5), (7, 9)],
        );
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

    #[test]
    fn vocabulary_refinement_ignores_only_unmapped_trailing_padding() {
        let shorter = map(vec![0, 1], 2);
        let padded = map(vec![0, 0, u32::MAX, u32::MAX], 1);
        assert!(many_to_one_id_map_refines(&shorter, &padded, true));
        assert!(!many_to_one_id_map_refines(&shorter, &padded, false),
            "tokenizer-state domain omissions are not vocabulary padding");
        let missing_real_id = map(vec![0, 0, 0, u32::MAX], 1);
        assert!(!many_to_one_id_map_refines(&shorter, &missing_real_id, true));
        let merged = map(vec![0, 0], 1);
        let split = map(vec![0, 1, u32::MAX], 2);
        assert!(!many_to_one_id_map_refines(&merged, &split, true),
            "padding equivalence must not erase a real partition distinction");
    }

    #[test]
    fn precomputed_refinement_with_unmapped_tail_preserves_original_coordinates() {
        let u = u32::MAX;
        let local = id_map(vec![0,1,1,0], 2, vec![1,0,1,1,u,2,2,u,u,u,u], 3);
        let common = id_map(vec![2,0,0,1], 3, vec![3,4,0,2,u,5,1,u,u,u], 6);
        assert!(internal_id_map_refines(&common, &local));
        let original = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([2..=2])),
        ]);
        fn original_pairs(weight: &Weight, map: &InternalIdMap) -> BTreeSet<(u32,u32)> {
            let mut out = BTreeSet::new();
            for (states, tokens) in weight.raw_range_values() {
                for internal_state in states {
                    for internal_token in tokens.iter() {
                        for &state in &map.tokenizer_states.internal_to_originals[internal_state as usize] {
                            for &token in &map.vocab_tokens.internal_to_originals[internal_token as usize] {
                                out.insert((state,token));
                            }
                        }
                    }
                }
            }
            out
        }
        let expected = original_pairs(&original, &local);
        let mut remapped = original.clone();
        remap_weights_into_existing_common(&mut [&mut remapped], &local, &common);
        assert_eq!(original_pairs(&remapped, &common), expected);
        assert!(!expected.iter().any(|&(_,id)| id==10), "padding never acquires a weight");
    }

}
