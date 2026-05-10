use super::*;
use super::almost_optimal_layout::{
    almost_optimal_adjacency_layout,
    bounded_greedy_adjacency_layout,
    greedy_adjacency_layout,
    improve_layout_2opt,
    improve_layout_reinsert,
    polish_layout_bounded,
    path_score,
    remaining_path_score_upper_bound,
    top_neighbors_by_vertex,
    top_weighted_vertices,
    weighted_degrees,
    SplitMix64,
};

pub(super) fn order_token_groups_globally_exact(
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

    let pair_weights = build_token_cooccurrence_pair_weights(&token_sets, num_groups);
    let layout = exact_layout_from_pair_weights_or_panic(&pair_weights, num_groups, "token");
    compose_group_layout(initial_perm, &layout)
}

pub(super) fn order_tsid_groups_globally_exact(
    token_compacted_weights: &[Weight],
    initial_perm: Vec<u32>,
    num_groups: usize,
    num_tokens: usize,
) -> Vec<u32> {
    if num_groups < 2 {
        return initial_perm;
    }

    // Rebuild through the TSID quotient before measuring the TSID objective.
    // This is important for exactness: two previously-distinct weights may
    // become the same interned weight after semantic TSID merging, and the
    // objective counts that final interned representative only once.
    let quotient_weights = apply_permutations_to_weight_set(
        token_compacted_weights,
        &initial_perm,
        &identity_perm(num_tokens),
    );
    if quotient_weights.is_empty() {
        return initial_perm;
    }

    let pair_weights = build_tsid_equal_value_pair_weights(&quotient_weights, num_groups);
    let layout = exact_layout_from_pair_weights_or_panic(&pair_weights, num_groups, "TSID");
    compose_group_layout(initial_perm, &layout)
}

pub(super) fn build_token_cooccurrence_pair_weights(
    token_sets: &[RangeSetBlaze<u32>],
    num_groups: usize,
) -> Vec<usize> {
    let mut pair_weights = vec![0usize; num_groups * num_groups];
    for token_set in token_sets {
        let mut members = rangeset_members_below(token_set, num_groups);
        members.sort_unstable();
        members.dedup();
        add_unit_clique_pair_weights(&mut pair_weights, num_groups, &members);
    }
    pair_weights
}

pub(super) fn build_tsid_equal_value_pair_weights(weights: &[Weight], num_groups: usize) -> Vec<usize> {
    let mut pair_weights = vec![0usize; num_groups * num_groups];
    if num_groups == 0 {
        return pair_weights;
    }

    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }

        let mut by_token_set = HashMap::<Vec<(u32, u32)>, Vec<usize>>::new();
        for (tsid_range, token_set) in weight.0.range_values() {
            let members = by_token_set.entry(rangeset_key(token_set)).or_default();
            let start = (*tsid_range.start() as usize).min(num_groups);
            let end = (*tsid_range.end() as usize).min(num_groups.saturating_sub(1));
            if start <= end {
                members.extend(start..=end);
            }
        }

        for members in by_token_set.values_mut() {
            members.sort_unstable();
            members.dedup();
            add_unit_clique_pair_weights(&mut pair_weights, num_groups, members);
        }
    }

    pair_weights
}

pub(super) fn rangeset_members_below(set: &RangeSetBlaze<u32>, upper_exclusive: usize) -> Vec<usize> {
    let mut members = Vec::new();
    if upper_exclusive == 0 {
        return members;
    }
    for range in set.ranges() {
        let start = *range.start() as usize;
        let end = (*range.end() as usize).min(upper_exclusive.saturating_sub(1));
        if start <= end {
            members.extend(start..=end);
        }
    }
    members
}

fn add_unit_clique_pair_weights(
    pair_weights: &mut [usize],
    num_groups: usize,
    members: &[usize],
) {
    for left_idx in 0..members.len() {
        let left = members[left_idx];
        if left >= num_groups {
            continue;
        }
        for &right in &members[left_idx + 1..] {
            if right >= num_groups || right == left {
                continue;
            }
            pair_weights[left * num_groups + right] += 1;
            pair_weights[right * num_groups + left] += 1;
        }
    }
}

pub(super) fn exact_layout_from_pair_weights_or_panic(
    pair_weights: &[usize],
    num_groups: usize,
    dimension_name: &str,
) -> Vec<usize> {
    debug_assert_eq!(pair_weights.len(), num_groups * num_groups);
    if num_groups < 2 {
        return (0..num_groups).collect();
    }

    let components = positive_pair_weight_components(pair_weights, num_groups);
    let max_component_groups = globally_exact_component_max_groups();
    let mut layout = Vec::with_capacity(num_groups);

    for component in components {
        if component.len() <= 1 {
            layout.extend(component);
            continue;
        }

        let local_weights = project_pair_weights(pair_weights, num_groups, &component);
        let local_layout = if almost_optimal_compaction_enabled() {
            eprintln!(
                "[glrmask/profile][almost_optimal_compaction] dimension={dimension_name} component_groups={} passes={} using=iterated_local_search",
                component.len(),
                almost_optimal_passes(),
            );
            almost_optimal_adjacency_layout(&local_weights, component.len())
        } else if component.len() <= max_component_groups {
            exact_max_adjacency_layout(&local_weights, component.len())
        } else {
            eprintln!(
                "[glrmask/profile][globally_exact_compaction] dimension={dimension_name} component_groups={} dp_limit={} using=branch_and_bound_exact",
                component.len(),
                max_component_groups,
            );
            exact_max_adjacency_layout_branch_and_bound(&local_weights, component.len())
        };
        layout.extend(local_layout.into_iter().map(|local| component[local]));
    }

    layout
}

fn fast_default_layout_from_pair_weights(pair_weights: &[usize], num_groups: usize) -> Vec<usize> {
    debug_assert_eq!(pair_weights.len(), num_groups * num_groups);
    if num_groups < 2 {
        return (0..num_groups).collect();
    }

    if legacy_exact_adjacency_proxy_enabled() && num_groups <= EXACT_LAYOUT_MAX_GROUPS {
        return exact_max_adjacency_layout(pair_weights, num_groups);
    }

    let components = positive_pair_weight_components(pair_weights, num_groups);
    let mut layout = Vec::with_capacity(num_groups);
    for component in components {
        if component.len() <= 1 {
            layout.extend(component);
            continue;
        }

        let local_weights = project_pair_weights(pair_weights, num_groups, &component);
        let local_layout = fast_default_component_layout(&local_weights, component.len());
        layout.extend(local_layout.into_iter().map(|local| component[local]));
    }
    layout
}

fn fast_default_component_layout(adjacency: &[usize], num_groups: usize) -> Vec<usize> {
    debug_assert_eq!(adjacency.len(), num_groups * num_groups);
    if num_groups < 2 {
        return (0..num_groups).collect();
    }

    let weighted_degree = weighted_degrees(adjacency, num_groups);
    let degree_order = top_weighted_vertices(&weighted_degree, num_groups);
    let top_neighbors = top_neighbors_by_vertex(adjacency, &weighted_degree, num_groups);
    let start_limit = if num_groups > LARGE_ALMOST_OPTIMAL_COMPONENT_GROUPS {
        8.min(num_groups)
    } else {
        24.min(num_groups)
    };

    let mut best_layout = Vec::new();
    let mut best_score = 0usize;
    let mut rng = SplitMix64::new(almost_optimal_seed() ^ ((num_groups as u64) << 33) ^ 0xd1b5_4a32_d192_ed03);

    for &start in degree_order.iter().take(start_limit) {
        let mut candidate = bounded_greedy_adjacency_layout(
            adjacency,
            &weighted_degree,
            num_groups,
            &top_neighbors,
            &degree_order,
            start,
            None,
            &mut rng,
        );
        polish_layout_bounded(adjacency, &mut candidate, num_groups);
        let score = path_score(adjacency, &candidate, num_groups);
        if best_layout.is_empty()
            || score > best_score
            || (score == best_score && candidate.as_slice() < best_layout.as_slice())
        {
            best_score = score;
            best_layout = candidate;
        }
    }

    best_layout
}

fn positive_pair_weight_components(pair_weights: &[usize], num_groups: usize) -> Vec<Vec<usize>> {
    let mut visited = vec![false; num_groups];
    let mut components = Vec::new();

    for start in 0..num_groups {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();

        while let Some(node) = stack.pop() {
            component.push(node);
            for next in 0..num_groups {
                if !visited[next] && pair_weights[node * num_groups + next] > 0 {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components.sort_by(|left, right| {
        left.first()
            .copied()
            .unwrap_or(usize::MAX)
            .cmp(&right.first().copied().unwrap_or(usize::MAX))
            .then(left.len().cmp(&right.len()))
    });
    components
}

fn project_pair_weights(
    pair_weights: &[usize],
    num_groups: usize,
    component: &[usize],
) -> Vec<usize> {
    let mut projected = vec![0usize; component.len() * component.len()];
    for (local_left, &global_left) in component.iter().enumerate() {
        for (local_right, &global_right) in component.iter().enumerate() {
            projected[local_left * component.len() + local_right] =
                pair_weights[global_left * num_groups + global_right];
        }
    }
    projected
}

fn exact_max_adjacency_layout(adjacency: &[usize], num_groups: usize) -> Vec<usize> {
    debug_assert!(num_groups <= EXACT_LAYOUT_MAX_GROUPS);
    if num_groups < 2 {
        return (0..num_groups).collect();
    }

    let states = 1usize << num_groups;
    let mut best = vec![0usize; states * num_groups];
    let mut reachable = vec![false; states * num_groups];
    let mut parent = vec![usize::MAX; states * num_groups];

    for group in 0..num_groups {
        reachable[(1usize << group) * num_groups + group] = true;
    }

    for mask in 1usize..states {
        for last in 0..num_groups {
            let state_idx = mask * num_groups + last;
            if !reachable[state_idx] {
                continue;
            }
            let current = best[state_idx];
            for next in 0..num_groups {
                let bit = 1usize << next;
                if mask & bit != 0 {
                    continue;
                }
                let next_mask = mask | bit;
                let next_score = current + adjacency[last * num_groups + next];
                let next_idx = next_mask * num_groups + next;
                if !reachable[next_idx]
                    || next_score > best[next_idx]
                    || (next_score == best[next_idx] && last < parent[next_idx])
                {
                    reachable[next_idx] = true;
                    best[next_idx] = next_score;
                    parent[next_idx] = last;
                }
            }
        }
    }

    let full_mask = states - 1;
    let mut last = (0..num_groups)
        .max_by_key(|&group| (best[full_mask * num_groups + group], usize::MAX - group))
        .unwrap();
    let mut mask = full_mask;
    let mut reversed = Vec::with_capacity(num_groups);

    loop {
        reversed.push(last);
        let prev = parent[mask * num_groups + last];
        if prev == usize::MAX {
            break;
        }
        mask &= !(1usize << last);
        last = prev;
    }

    reversed.reverse();
    reversed
}

fn exact_max_adjacency_layout_branch_and_bound(
    adjacency: &[usize],
    num_groups: usize,
) -> Vec<usize> {
    debug_assert_eq!(adjacency.len(), num_groups * num_groups);
    if num_groups < 2 {
        return (0..num_groups).collect();
    }

    let mut weighted_degree = vec![0usize; num_groups];
    for left in 0..num_groups {
        for right in 0..num_groups {
            let weight = adjacency[left * num_groups + right];
            weighted_degree[left] += weight;
        }
    }

    let mut best_layout = greedy_adjacency_layout(adjacency, &weighted_degree, num_groups);
    improve_layout_2opt(adjacency, &mut best_layout, num_groups);
    improve_layout_reinsert(adjacency, &mut best_layout, num_groups);
    let mut best_score = path_score(adjacency, &best_layout, num_groups);
    let initial_upper_bound =
        remaining_path_score_upper_bound(adjacency, num_groups, None, &vec![false; num_groups]);
    eprintln!(
        "[glrmask/profile][globally_exact_compaction_bnb] groups={num_groups} incumbent_score={best_score} initial_upper_bound={initial_upper_bound}"
    );
    if best_score == initial_upper_bound {
        eprintln!(
            "[glrmask/profile][globally_exact_compaction_bnb] groups={num_groups} proven=upper_bound_tight"
        );
        return best_layout;
    }

    let mut used = vec![false; num_groups];
    let mut path = Vec::with_capacity(num_groups);
    let mut starts: Vec<usize> = (0..num_groups).collect();
    starts.sort_by_key(|&group| (usize::MAX - weighted_degree[group], group));

    for start in starts {
        used[start] = true;
        path.push(start);
        exact_layout_branch_and_bound_dfs(
            adjacency,
            num_groups,
            &weighted_degree,
            &mut used,
            &mut path,
            0,
            &mut best_score,
            &mut best_layout,
        );
        path.pop();
        used[start] = false;
    }

    best_layout
}

fn exact_layout_branch_and_bound_dfs(
    adjacency: &[usize],
    num_groups: usize,
    weighted_degree: &[usize],
    used: &mut [bool],
    path: &mut Vec<usize>,
    score: usize,
    best_score: &mut usize,
    best_layout: &mut Vec<usize>,
) {
    let remaining = num_groups - path.len();
    if remaining == 0 {
        if score > *best_score || (score == *best_score && path.as_slice() < best_layout.as_slice()) {
            *best_score = score;
            best_layout.clear();
            best_layout.extend_from_slice(path);
        }
        return;
    }

    let optimistic = score.saturating_add(remaining_path_score_upper_bound(
        adjacency,
        num_groups,
        path.last().copied(),
        used,
    ));
    if optimistic < *best_score {
        return;
    }

    let last = *path.last().unwrap();
    let mut candidates: Vec<usize> = (0..num_groups).filter(|&group| !used[group]).collect();
    candidates.sort_by_key(|&candidate| {
        (
            usize::MAX - adjacency[last * num_groups + candidate],
            usize::MAX - weighted_degree[candidate],
            candidate,
        )
    });

    for next in candidates {
        used[next] = true;
        path.push(next);
        exact_layout_branch_and_bound_dfs(
            adjacency,
            num_groups,
            weighted_degree,
            used,
            path,
            score + adjacency[last * num_groups + next],
            best_score,
            best_layout,
        );
        path.pop();
        used[next] = false;
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    fn singleton_set(token: u32) -> RangeSetBlaze<u32> {
        RangeSetBlaze::from_iter(std::iter::once(token..=token))
    }

    fn set_from_bits(bits: usize, width: usize) -> RangeSetBlaze<u32> {
        RangeSetBlaze::from_iter(
            (0..width)
                .filter(move |bit| bits & (1usize << bit) != 0)
                .map(|bit| bit as u32..=bit as u32),
        )
    }

    fn all_permutations(n: usize) -> Vec<Vec<usize>> {
        fn rec(pos: usize, values: &mut [usize], out: &mut Vec<Vec<usize>>) {
            if pos == values.len() {
                out.push(values.to_vec());
                return;
            }
            for idx in pos..values.len() {
                values.swap(pos, idx);
                rec(pos + 1, values, out);
                values.swap(pos, idx);
            }
        }

        let mut values: Vec<usize> = (0..n).collect();
        let mut out = Vec::new();
        rec(0, &mut values, &mut out);
        out
    }

    fn path_score(pair_weights: &[usize], num_groups: usize, layout: &[usize]) -> usize {
        layout
            .windows(2)
            .map(|pair| pair_weights[pair[0] * num_groups + pair[1]])
            .sum()
    }

    fn brute_force_best_path_score(pair_weights: &[usize], num_groups: usize) -> usize {
        all_permutations(num_groups)
            .into_iter()
            .map(|layout| path_score(pair_weights, num_groups, &layout))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn exact_token_layout_matches_bruteforce_for_all_four_element_set_families() {
        // Exhaustive over every family of subsets on a four-element token
        // universe.  The exact objective for a family of token sets is the
        // constant total cardinality minus the Hamiltonian path score generated
        // by pair co-occurrence weights, so matching the brute-force path score
        // verifies the token-layout optimizer for this entire tiny universe.
        let num_groups = 4;
        let nonempty_subsets: Vec<_> = (1usize..(1usize << num_groups))
            .map(|bits| set_from_bits(bits, num_groups))
            .collect();

        for family_bits in 0usize..(1usize << nonempty_subsets.len()) {
            let family: Vec<_> = nonempty_subsets
                .iter()
                .enumerate()
                .filter_map(|(idx, set)| {
                    (family_bits & (1usize << idx) != 0).then(|| set.clone())
                })
                .collect();
            let pair_weights = build_token_cooccurrence_pair_weights(&family, num_groups);
            let exact_layout = exact_layout_from_pair_weights_or_panic(
                &pair_weights,
                num_groups,
                "test-token",
            );
            let exact_score = path_score(&pair_weights, num_groups, &exact_layout);
            let brute_score = brute_force_best_path_score(&pair_weights, num_groups);
            assert_eq!(exact_score, brute_score, "family_bits={family_bits:#x}");
        }
    }

    #[test]
    fn exact_tsid_layout_matches_bruteforce_for_all_single_weight_four_tsid_labelings() {
        // Exhaustive over every single-weight map from four TSIDs to
        // {empty, token-set-A, token-set-B}.  This directly validates the
        // outer RangeMap objective transformation used by globally exact mode.
        let num_groups = 4;
        for mut code in 0usize..3usize.pow(num_groups as u32) {
            let mut entries = Vec::new();
            let mut labels = Vec::new();
            for tsid in 0..num_groups {
                let label = code % 3;
                code /= 3;
                labels.push(label);
                match label {
                    0 => {}
                    1 => entries.push((tsid as u32, singleton_set(11))),
                    2 => entries.push((tsid as u32, singleton_set(17))),
                    _ => unreachable!(),
                }
            }

            let weight = Weight::from_per_tsid_token_sets(entries);
            let pair_weights = build_tsid_equal_value_pair_weights(&[weight], num_groups);
            let exact_layout = exact_layout_from_pair_weights_or_panic(
                &pair_weights,
                num_groups,
                "test-tsid",
            );
            let exact_score = path_score(&pair_weights, num_groups, &exact_layout);
            let brute_score = brute_force_best_path_score(&pair_weights, num_groups);
            assert_eq!(exact_score, brute_score, "labels={labels:?}");
        }
    }

    #[test]
    fn exact_layout_decomposes_zero_weight_components_without_losing_score() {
        let num_groups = 6;
        let mut pair_weights = vec![0usize; num_groups * num_groups];
        // Component {0, 1, 2}
        pair_weights[0 * num_groups + 1] = 7;
        pair_weights[1 * num_groups + 0] = 7;
        pair_weights[1 * num_groups + 2] = 5;
        pair_weights[2 * num_groups + 1] = 5;
        // Component {3, 4}; node 5 is an isolated singleton component.
        pair_weights[3 * num_groups + 4] = 9;
        pair_weights[4 * num_groups + 3] = 9;

        let exact_layout = exact_layout_from_pair_weights_or_panic(
            &pair_weights,
            num_groups,
            "test-components",
        );
        let exact_score = path_score(&pair_weights, num_groups, &exact_layout);
        let brute_score = brute_force_best_path_score(&pair_weights, num_groups);
        assert_eq!(exact_score, brute_score);
    }
}
