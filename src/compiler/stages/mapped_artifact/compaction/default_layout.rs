use super::*;
use super::exact_layout::rangeset_members_below;

pub(super) fn order_token_groups(
    weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    let token_sets = collect_token_sets_after_permutation(weights, &initial_perm);
    if token_sets.is_empty() {
        return initial_perm;
    }

    let layout = sketch_layout_from_token_sets(&token_sets, num_groups);
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

    let layout = sketch_layout_from_tsid_equal_values(&quotient_weights, num_groups);
    compose_group_layout(initial_perm, &layout)
}

fn sketch_layout_from_token_sets(
    token_sets: &[RangeSetBlaze<u32>],
    num_groups: usize,
) -> Vec<usize> {
    let mut sketches = vec![[u64::MAX; DEFAULT_LAYOUT_SKETCH_WORDS]; num_groups];
    let mut degrees = vec![0usize; num_groups];

    for (context, token_set) in token_sets.iter().enumerate() {
        for member in rangeset_members_below(token_set, num_groups) {
            update_membership_sketch(&mut sketches[member], context as u64);
            degrees[member] += 1;
        }
    }

    sketch_layout_from_group_sketches(sketches, degrees)
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
            for tsid in start..=end {
                update_membership_sketch(&mut sketches[tsid], token_set_context);
                degrees[tsid] += 1;
            }
        }
    }

    sketch_layout_from_group_sketches(sketches, degrees)
}

fn update_membership_sketch(sketch: &mut [u64; DEFAULT_LAYOUT_SKETCH_WORDS], context: u64) {
    for (idx, slot) in sketch.iter_mut().enumerate() {
        let hash = mix64(context ^ ((idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
        if hash < *slot {
            *slot = hash;
        }
    }
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

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
