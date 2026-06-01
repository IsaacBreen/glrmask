use super::*;
pub(super) fn order_token_groups(
    weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    let sketch_layout =
        sketch_layout_from_mapped_weight_token_sets(weights, &initial_perm, num_groups);
    let exact_profile_layout =
        exact_profile_layout_from_mapped_weight_token_sets(weights, &initial_perm, num_groups);
    let layout = best_token_layout_by_range_count(
        weights,
        &initial_perm,
        [sketch_layout, exact_profile_layout],
    );
    if layout.is_empty() {
        return initial_perm;
    }
    compose_group_layout(initial_perm, &layout)
}

pub(super) fn order_token_groups_exact_profile(
    weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    let layout =
        exact_profile_layout_from_mapped_weight_token_sets(weights, &initial_perm, num_groups);
    if layout.is_empty() {
        return initial_perm;
    }
    compose_group_layout(initial_perm, &layout)
}

pub(super) fn order_token_groups_sketch(
    weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    let layout = sketch_layout_from_mapped_weight_token_sets(weights, &initial_perm, num_groups);
    if layout.is_empty() {
        return initial_perm;
    }
    compose_group_layout(initial_perm, &layout)
}

pub(super) fn order_tsid_groups(
    token_compacted_weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
    num_tokens: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    let quotient_weights = apply_permutations_to_weight_set(
        token_compacted_weights,
        &initial_perm,
        &identity_perm(num_tokens),
    );
    if quotient_weights.is_empty() {
        return initial_perm;
    }

    let sketch_layout = sketch_layout_from_tsid_equal_values(&quotient_weights, num_groups);
    let exact_profile_layout = exact_profile_layout_from_tsid_equal_values(&quotient_weights, num_groups);
    let layout = best_tsid_layout_by_range_count(
        &quotient_weights,
        &initial_perm,
        num_tokens,
        [sketch_layout, exact_profile_layout],
    );
    compose_group_layout(initial_perm, &layout)
}

pub(super) fn order_tsid_groups_exact_profile(
    token_compacted_weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
    num_tokens: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    let quotient_weights = apply_permutations_to_weight_set(
        token_compacted_weights,
        &initial_perm,
        &identity_perm(num_tokens),
    );
    if quotient_weights.is_empty() {
        return initial_perm;
    }

    let layout = exact_profile_layout_from_tsid_equal_values(&quotient_weights, num_groups);
    compose_group_layout(initial_perm, &layout)
}

pub(super) fn order_tsid_groups_sketch(
    token_compacted_weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
    num_tokens: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    let quotient_weights = apply_permutations_to_weight_set(
        token_compacted_weights,
        &initial_perm,
        &identity_perm(num_tokens),
    );
    if quotient_weights.is_empty() {
        return initial_perm;
    }

    let layout = sketch_layout_from_tsid_equal_values(&quotient_weights, num_groups);
    compose_group_layout(initial_perm, &layout)
}

fn best_token_layout_by_range_count(
    weights: &[Weight],
    initial_perm: &[u32],
    layouts: [Vec<usize>; 2],
) -> Vec<usize> {
    layouts
        .into_iter()
        .filter(|layout| !layout.is_empty())
        .min_by_key(|layout| {
            let perm = compose_group_layout(initial_perm.to_vec(), layout);
            token_ranges_after_perm(weights, &perm)
        })
        .unwrap_or_default()
}

fn best_tsid_layout_by_range_count(
    quotient_weights: &[Weight],
    initial_perm: &[u32],
    num_tokens: usize,
    layouts: [Vec<usize>; 2],
) -> Vec<usize> {
    layouts
        .into_iter()
        .filter(|layout| !layout.is_empty())
        .min_by_key(|layout| {
            let perm = compose_group_layout(initial_perm.to_vec(), layout);
            apply_permutations_to_weight_set(
                quotient_weights,
                &perm,
                &identity_perm(num_tokens),
            )
            .iter()
            .map(Weight::num_ranges)
            .sum::<usize>()
        })
        .unwrap_or_default()
}

fn token_ranges_after_perm(weights: &[Weight], token_perm: &[u32]) -> usize {
    let token_runs = permutation_runs(token_perm);
    let mut seen_ptrs = HashSet::new();
    let mut seen_mapped_sets = HashSet::<Vec<(u32, u32)>>::new();
    let mut ranges = 0usize;

    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
            let ptr = Arc::as_ptr(token_set) as usize;
            if !seen_ptrs.insert(ptr) {
                continue;
            }
            let mapped_ranges = mapped_rangeset_key_with_runs(token_set, &token_runs);
            if seen_mapped_sets.insert(mapped_ranges.clone()) {
                ranges += mapped_ranges.len();
            }
        }
    }

    ranges
}

fn sketch_layout_from_mapped_weight_token_sets(
    weights: &[Weight],
    token_perm: &[u32],
    num_groups: usize,
) -> Vec<usize> {
    let token_runs = permutation_runs(token_perm);
    let mut seen_ptrs = HashSet::new();
    let mut seen_mapped_sets = HashSet::<Vec<(u32, u32)>>::new();
    let mut sketches = vec![[u64::MAX; DEFAULT_LAYOUT_SKETCH_WORDS]; num_groups];
    let mut degrees = vec![0usize; num_groups];
    let mut context = 0u64;

    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
            let ptr = Arc::as_ptr(token_set) as usize;
            if !seen_ptrs.insert(ptr) {
                continue;
            }

            let mapped_ranges = mapped_rangeset_key_with_runs(token_set, &token_runs);
            if mapped_ranges.is_empty() || !seen_mapped_sets.insert(mapped_ranges.clone()) {
                continue;
            }

            for &(start, end) in &mapped_ranges {
                let start = (start as usize).min(num_groups);
                let end = (end as usize).min(num_groups.saturating_sub(1));
                if start > end {
                    continue;
                }
                for member in start..=end {
                    update_membership_sketch(&mut sketches[member], context);
                    degrees[member] += 1;
                }
            }
            context += 1;
        }
    }

    if context == 0 {
        Vec::new()
    } else {
        sketch_layout_from_group_sketches(sketches, degrees)
    }
}

fn exact_profile_layout_from_mapped_weight_token_sets(
    weights: &[Weight],
    token_perm: &[u32],
    num_groups: usize,
) -> Vec<usize> {
    let token_runs = permutation_runs(token_perm);
    let mut seen_ptrs = HashSet::new();
    let mut seen_mapped_sets = HashSet::<Vec<(u32, u32)>>::new();
    let mut profiles = vec![Vec::<u32>::new(); num_groups];
    let mut context = 0u32;

    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
            let ptr = Arc::as_ptr(token_set) as usize;
            if !seen_ptrs.insert(ptr) {
                continue;
            }

            let mapped_ranges = mapped_rangeset_key_with_runs(token_set, &token_runs);
            if mapped_ranges.is_empty() || !seen_mapped_sets.insert(mapped_ranges.clone()) {
                continue;
            }

            for &(start, end) in &mapped_ranges {
                let start = (start as usize).min(num_groups);
                let end = (end as usize).min(num_groups.saturating_sub(1));
                if start > end {
                    continue;
                }
                for profile in &mut profiles[start..=end] {
                    profile.push(context);
                }
            }
            context += 1;
        }
    }

    if context == 0 {
        Vec::new()
    } else {
        exact_profile_layout_from_group_profiles(profiles)
    }
}

fn mapped_rangeset_key_with_runs(
    set: &RangeSetBlaze<u32>,
    runs: &[PermRun],
) -> Vec<(u32, u32)> {
    let mut mapped = Vec::new();
    for range in set.ranges() {
        mapped.extend(
            overlapping_perm_runs(runs, *range.start(), *range.end())
                .iter()
                .map(|run| run.mapped),
        );
    }
    mapped.sort_unstable();
    mapped.dedup();

    let mut ranges = Vec::new();
    let Some((&first, rest)) = mapped.split_first() else {
        return ranges;
    };
    let mut start = first;
    let mut end = first;
    for &member in rest {
        if member == end + 1 {
            end = member;
        } else {
            ranges.push((start, end));
            start = member;
            end = member;
        }
    }
    ranges.push((start, end));
    ranges
}

fn exact_profile_layout_from_tsid_equal_values(weights: &[Weight], num_groups: usize) -> Vec<usize> {
    let mut profiles = vec![Vec::<u32>::new(); num_groups];
    let mut context = 0u32;

    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }

        let mut contexts_by_token_set = HashMap::<Vec<(u32, u32)>, u32>::new();
        for (tsid_range, token_set) in weight.0.range_values() {
            let token_set_context = *contexts_by_token_set
                .entry(rangeset_key(token_set))
                .or_insert_with(|| {
                    let current = context;
                    context += 1;
                    current
                });
            let start = (*tsid_range.start() as usize).min(num_groups);
            let end = (*tsid_range.end() as usize).min(num_groups.saturating_sub(1));
            if start > end {
                continue;
            }
            for profile in &mut profiles[start..=end] {
                profile.push(token_set_context);
            }
        }
    }

    exact_profile_layout_from_group_profiles(profiles)
}

fn sketch_layout_from_tsid_equal_values(weights: &[Weight], num_groups: usize) -> Vec<usize> {
    let mut sketches = vec![[u64::MAX; DEFAULT_LAYOUT_SKETCH_WORDS]; num_groups];
    let mut degrees = vec![0usize; num_groups];
    let mut context = 0u64;

    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }

        let mut contexts_by_token_set = HashMap::<Vec<(u32, u32)>, u64>::new();
        for (tsid_range, token_set) in weight.0.range_values() {
            let token_set_context = *contexts_by_token_set
                .entry(rangeset_key(token_set))
                .or_insert_with(|| {
                    let current = context;
                    context += 1;
                    current
                });
            let start = (*tsid_range.start() as usize).min(num_groups);
            let end = (*tsid_range.end() as usize).min(num_groups.saturating_sub(1));
            if start > end {
                continue;
            }
            for tsid in start..=end {
                update_membership_sketch(&mut sketches[tsid], token_set_context);
                degrees[tsid] += 1;
            }
        }
    }

    sketch_layout_from_group_sketches(sketches, degrees)
}

fn exact_profile_layout_from_group_profiles(profiles: Vec<Vec<u32>>) -> Vec<usize> {
    let mut layout: Vec<usize> = (0..profiles.len()).collect();
    layout.sort_by(|&left, &right| {
        profiles[left]
            .cmp(&profiles[right])
            .then_with(|| profiles[right].len().cmp(&profiles[left].len()))
            .then(left.cmp(&right))
    });
    layout
}

fn sketch_layout_from_group_sketches(
    sketches: Vec<[u64; DEFAULT_LAYOUT_SKETCH_WORDS]>,
    degrees: Vec<usize>,
) -> Vec<usize> {
    let mut layout: Vec<usize> = (0..sketches.len()).collect();
    layout.sort_by(|&left, &right| {
        sketches[left]
            .cmp(&sketches[right])
            .then_with(|| degrees[right].cmp(&degrees[left]))
            .then(left.cmp(&right))
    });
    layout
}

fn update_membership_sketch(sketch: &mut [u64; DEFAULT_LAYOUT_SKETCH_WORDS], context: u64) {
    for (idx, slot) in sketch.iter_mut().enumerate() {
        let hash = mix64(context ^ ((idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
        if hash < *slot {
            *slot = hash;
        }
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
