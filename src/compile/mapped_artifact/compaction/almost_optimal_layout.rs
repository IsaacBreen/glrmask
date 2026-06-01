use super::*;

pub(super) fn almost_optimal_adjacency_layout(adjacency: &[usize], num_groups: usize) -> Vec<usize> {
    debug_assert_eq!(adjacency.len(), num_groups * num_groups);
    if num_groups < 2 {
        return (0..num_groups).collect();
    }

    let weighted_degree = weighted_degrees(adjacency, num_groups);

    if num_groups > LARGE_ALMOST_OPTIMAL_COMPONENT_GROUPS {
        return almost_optimal_large_adjacency_layout(adjacency, num_groups, &weighted_degree);
    }

    let mut rng = SplitMix64::new(almost_optimal_seed() ^ ((num_groups as u64) << 32));
    let mut best_layout = greedy_adjacency_layout(adjacency, &weighted_degree, num_groups);
    polish_layout(adjacency, &mut best_layout, num_groups);
    let mut best_score = path_score(adjacency, &best_layout, num_groups);
    let upper_bound =
        remaining_path_score_upper_bound(adjacency, num_groups, None, &vec![false; num_groups]);

    if best_score == upper_bound {
        eprintln!(
            "[glrmask/profile][almost_optimal_compaction] groups={num_groups} score={best_score} upper_bound={upper_bound} proven=upper_bound_tight"
        );
        return best_layout;
    }

    let passes = almost_optimal_passes();
    for pass in 0..passes {
        let mut candidate = if pass % 4 == 0 {
            randomized_greedy_adjacency_layout(adjacency, &weighted_degree, num_groups, &mut rng)
        } else {
            best_layout.clone()
        };
        perturb_layout(&mut candidate, &mut rng, pass);
        polish_layout(adjacency, &mut candidate, num_groups);
        let score = path_score(adjacency, &candidate, num_groups);
        if score > best_score || (score == best_score && candidate.as_slice() < best_layout.as_slice()) {
            best_score = score;
            best_layout = candidate;
            eprintln!(
                "[glrmask/profile][almost_optimal_compaction] groups={num_groups} pass={pass} improved_score={best_score} upper_bound={upper_bound} gap={}",
                upper_bound.saturating_sub(best_score),
            );
            if best_score == upper_bound {
                eprintln!(
                    "[glrmask/profile][almost_optimal_compaction] groups={num_groups} score={best_score} upper_bound={upper_bound} proven=upper_bound_tight"
                );
                break;
            }
        }
    }

    eprintln!(
        "[glrmask/profile][almost_optimal_compaction] groups={num_groups} final_score={best_score} upper_bound={upper_bound} gap={}",
        upper_bound.saturating_sub(best_score),
    );
    best_layout
}

pub(super) fn weighted_degrees(adjacency: &[usize], num_groups: usize) -> Vec<usize> {
    let mut weighted_degree = vec![0usize; num_groups];
    for left in 0..num_groups {
        for right in 0..num_groups {
            weighted_degree[left] += adjacency[left * num_groups + right];
        }
    }
    weighted_degree
}

pub(super) fn remaining_path_score_upper_bound(
    adjacency: &[usize],
    num_groups: usize,
    last: Option<usize>,
    used: &[bool],
) -> usize {
    let unused_count = used.iter().filter(|&&is_used| !is_used).count();
    if unused_count == 0 {
        return 0;
    }

    let mut degree_capacity_sum = 0usize;
    let mut endpoint_loss_candidates = Vec::with_capacity(unused_count);

    if let Some(last) = last {
        let best_from_last = (0..num_groups)
            .filter(|&candidate| !used[candidate])
            .map(|candidate| adjacency[last * num_groups + candidate])
            .max()
            .unwrap_or(0);
        degree_capacity_sum += best_from_last;
    }

    for vertex in 0..num_groups {
        if used[vertex] {
            continue;
        }

        let mut best = 0usize;
        let mut second = 0usize;
        for other in 0..num_groups {
            if other == vertex {
                continue;
            }
            if used[other] && Some(other) != last {
                continue;
            }
            let weight = adjacency[vertex * num_groups + other];
            if weight >= best {
                second = best;
                best = weight;
            } else if weight > second {
                second = weight;
            }
        }
        degree_capacity_sum += best + second;
        endpoint_loss_candidates.push(second);
    }

    endpoint_loss_candidates.sort_unstable();
    let endpoint_losses_needed = if last.is_some() { 1 } else { 2 };
    let endpoint_loss: usize = endpoint_loss_candidates
        .iter()
        .take(endpoint_losses_needed.min(endpoint_loss_candidates.len()))
        .sum();

    degree_capacity_sum.saturating_sub(endpoint_loss) / 2
}

fn almost_optimal_large_adjacency_layout(
    adjacency: &[usize],
    num_groups: usize,
    weighted_degree: &[usize],
) -> Vec<usize> {
    let mut rng = SplitMix64::new(almost_optimal_seed() ^ ((num_groups as u64) << 32));
    let top_neighbors = top_neighbors_by_vertex(adjacency, weighted_degree, num_groups);
    let degree_order = top_weighted_vertices(weighted_degree, num_groups);
    let starts = LARGE_ALMOST_OPTIMAL_GREEDY_STARTS.min(num_groups);
    let mut best_layout = Vec::new();
    let mut best_score = 0usize;

    for &start in degree_order.iter().take(starts) {
        let mut candidate = bounded_greedy_adjacency_layout(
            adjacency,
            weighted_degree,
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

    let upper_bound =
        remaining_path_score_upper_bound(adjacency, num_groups, None, &vec![false; num_groups]);
    if best_score == upper_bound {
        eprintln!(
            "[glrmask/profile][almost_optimal_compaction] groups={num_groups} score={best_score} upper_bound={upper_bound} proven=upper_bound_tight"
        );
        return best_layout;
    }

    let passes = almost_optimal_passes();
    for pass in 0..passes {
        let start = degree_order[rng.gen_usize(starts)];
        let random_window = if pass % 2 == 0 {
            Some(LARGE_ALMOST_OPTIMAL_RANDOM_WINDOW)
        } else {
            None
        };
        let mut candidate = bounded_greedy_adjacency_layout(
            adjacency,
            weighted_degree,
            num_groups,
            &top_neighbors,
            &degree_order,
            start,
            random_window,
            &mut rng,
        );
        perturb_layout(&mut candidate, &mut rng, pass);
        polish_layout_bounded(adjacency, &mut candidate, num_groups);
        let score = path_score(adjacency, &candidate, num_groups);
        if score > best_score || (score == best_score && candidate.as_slice() < best_layout.as_slice()) {
            best_score = score;
            best_layout = candidate;
            eprintln!(
                "[glrmask/profile][almost_optimal_compaction] groups={num_groups} pass={pass} improved_score={best_score} upper_bound={upper_bound} gap={}",
                upper_bound.saturating_sub(best_score),
            );
            if best_score == upper_bound {
                eprintln!(
                    "[glrmask/profile][almost_optimal_compaction] groups={num_groups} score={best_score} upper_bound={upper_bound} proven=upper_bound_tight"
                );
                break;
            }
        }
    }

    eprintln!(
        "[glrmask/profile][almost_optimal_compaction] groups={num_groups} final_score={best_score} upper_bound={upper_bound} gap={}",
        upper_bound.saturating_sub(best_score),
    );
    best_layout
}

#[derive(Clone, Debug)]
pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn gen_usize(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper_exclusive
    }
}

pub(super) fn greedy_adjacency_layout(
    adjacency: &[usize],
    weighted_degree: &[usize],
    num_groups: usize,
) -> Vec<usize> {
    let starts = if num_groups > LARGE_ALMOST_OPTIMAL_COMPONENT_GROUPS {
        top_weighted_vertices(weighted_degree, LARGE_ALMOST_OPTIMAL_GREEDY_STARTS.min(num_groups))
    } else {
        (0..num_groups).collect()
    };
    let mut best_layout = Vec::new();
    let mut best_score = 0usize;

    for start in starts {
        let mut used = vec![false; num_groups];
        let mut layout = Vec::with_capacity(num_groups);
        used[start] = true;
        layout.push(start);

        while layout.len() < num_groups {
            let last = *layout.last().unwrap();
            let next = (0..num_groups)
                .filter(|&group| !used[group])
                .max_by_key(|&group| {
                    (
                        adjacency[last * num_groups + group],
                        weighted_degree[group],
                        usize::MAX - group,
                    )
                })
                .unwrap();
            used[next] = true;
            layout.push(next);
        }

        let score = path_score(adjacency, &layout, num_groups);
        if best_layout.is_empty()
            || score > best_score
            || (score == best_score && layout.as_slice() < best_layout.as_slice())
        {
            best_score = score;
            best_layout = layout;
        }
    }

    best_layout
}

fn randomized_greedy_adjacency_layout(
    adjacency: &[usize],
    weighted_degree: &[usize],
    num_groups: usize,
    rng: &mut SplitMix64,
) -> Vec<usize> {
    let start_pool = top_weighted_vertices(weighted_degree, 32.min(num_groups));
    let start = start_pool[rng.gen_usize(start_pool.len())];
    let mut used = vec![false; num_groups];
    let mut layout = Vec::with_capacity(num_groups);
    used[start] = true;
    layout.push(start);

    while layout.len() < num_groups {
        let last = *layout.last().unwrap();
        let mut candidates: Vec<usize> = (0..num_groups).filter(|&group| !used[group]).collect();
        candidates.sort_by_key(|&group| {
            (
                usize::MAX - adjacency[last * num_groups + group],
                usize::MAX - weighted_degree[group],
                group,
            )
        });
        let window = 8.min(candidates.len());
        let next = candidates[rng.gen_usize(window)];
        used[next] = true;
        layout.push(next);
    }

    layout
}

pub(super) fn top_neighbors_by_vertex(
    adjacency: &[usize],
    weighted_degree: &[usize],
    num_groups: usize,
) -> Vec<Vec<usize>> {
    let limit = LARGE_ALMOST_OPTIMAL_NEIGHBORS.min(num_groups.saturating_sub(1)).max(1);
    (0..num_groups)
        .map(|left| {
            let mut neighbors: Vec<usize> = (0..num_groups)
                .filter(|&right| right != left && adjacency[left * num_groups + right] > 0)
                .collect();
            if neighbors.len() > limit * 4 {
                neighbors.select_nth_unstable_by_key(limit, |&right| {
                    (
                        usize::MAX - adjacency[left * num_groups + right],
                        usize::MAX - weighted_degree[right],
                        right,
                    )
                });
                neighbors.truncate(limit);
            }
            neighbors.sort_by_key(|&right| {
                (
                    usize::MAX - adjacency[left * num_groups + right],
                    usize::MAX - weighted_degree[right],
                    right,
                )
            });
            neighbors.truncate(limit);
            neighbors
        })
        .collect()
}

pub(super) fn bounded_greedy_adjacency_layout(
    adjacency: &[usize],
    weighted_degree: &[usize],
    num_groups: usize,
    top_neighbors: &[Vec<usize>],
    degree_order: &[usize],
    start: usize,
    random_window: Option<usize>,
    rng: &mut SplitMix64,
) -> Vec<usize> {
    let mut used = vec![false; num_groups];
    let mut layout = Vec::with_capacity(num_groups);
    let mut degree_cursor = 0usize;
    used[start] = true;
    layout.push(start);

    while layout.len() < num_groups {
        let last = *layout.last().unwrap();
        let mut candidates = [usize::MAX; LARGE_ALMOST_OPTIMAL_RANDOM_WINDOW];
        let mut candidate_len = 0usize;
        let mut best = None;

        for &candidate in &top_neighbors[last] {
            if used[candidate] {
                continue;
            }
            if let Some(window) = random_window {
                if candidate_len < window.min(candidates.len()) {
                    candidates[candidate_len] = candidate;
                    candidate_len += 1;
                    continue;
                }
            }
            best = Some(candidate);
            break;
        }

        let next = if candidate_len > 0 {
            candidates[rng.gen_usize(candidate_len)]
        } else if let Some(best) = best {
            best
        } else {
            while degree_cursor < degree_order.len() && used[degree_order[degree_cursor]] {
                degree_cursor += 1;
            }
            if degree_cursor < degree_order.len() {
                degree_order[degree_cursor]
            } else {
                (0..num_groups)
                    .filter(|&group| !used[group])
                    .max_by_key(|&group| (weighted_degree[group], usize::MAX - group))
                    .unwrap()
            }
        };

        debug_assert!(!used[next]);
        used[next] = true;
        layout.push(next);
    }

    debug_assert_eq!(layout.len(), num_groups);
    layout
}

pub(super) fn top_weighted_vertices(weighted_degree: &[usize], limit: usize) -> Vec<usize> {
    let mut vertices: Vec<usize> = (0..weighted_degree.len()).collect();
    vertices.sort_by_key(|&vertex| (usize::MAX - weighted_degree[vertex], vertex));
    vertices.truncate(limit.max(1));
    vertices
}

fn perturb_layout(layout: &mut Vec<usize>, rng: &mut SplitMix64, pass: usize) {
    if layout.len() < 4 {
        return;
    }

    let moves = 1 + (pass % 7);
    for _ in 0..moves {
        match rng.gen_usize(3) {
            0 => {
                let left = rng.gen_usize(layout.len());
                let right = rng.gen_usize(layout.len());
                if left != right {
                    layout.swap(left, right);
                }
            }
            1 => {
                let mut left = rng.gen_usize(layout.len());
                let mut right = rng.gen_usize(layout.len());
                if left > right {
                    std::mem::swap(&mut left, &mut right);
                }
                if right > left {
                    layout[left..=right].reverse();
                }
            }
            _ => {
                let from = rng.gen_usize(layout.len());
                let value = layout.remove(from);
                let to = rng.gen_usize(layout.len() + 1);
                layout.insert(to, value);
            }
        }
    }
}

fn polish_layout(adjacency: &[usize], layout: &mut Vec<usize>, num_groups: usize) {
    if num_groups > LARGE_ALMOST_OPTIMAL_COMPONENT_GROUPS {
        polish_layout_bounded(adjacency, layout, num_groups);
        return;
    }

    loop {
        let before = path_score(adjacency, layout, num_groups);
        improve_layout_2opt(adjacency, layout, num_groups);
        improve_layout_reinsert(adjacency, layout, num_groups);
        let after = path_score(adjacency, layout, num_groups);
        if after == before {
            break;
        }
    }
}

pub(super) fn polish_layout_bounded(adjacency: &[usize], layout: &mut [usize], num_groups: usize) {
    for _ in 0..4 {
        let mut improved = false;
        improved |= improve_layout_bounded_2opt(adjacency, layout, num_groups);
        improved |= improve_layout_adjacent_swaps(adjacency, layout, num_groups);
        if !improved {
            break;
        }
    }
}

fn improve_layout_bounded_2opt(
    adjacency: &[usize],
    layout: &mut [usize],
    num_groups: usize,
) -> bool {
    if layout.len() < 4 {
        return false;
    }

    let mut improved_any = false;
    for left_edge in 0..layout.len() - 2 {
        let a = layout[left_edge];
        let b = layout[left_edge + 1];
        let right_limit = (left_edge + LARGE_ALMOST_OPTIMAL_2OPT_WINDOW).min(layout.len() - 2);
        let mut best_gain = 0usize;
        let mut best_right_edge = None;

        for right_edge in left_edge + 2..=right_limit {
            let c = layout[right_edge];
            let d = layout[right_edge + 1];
            let old = adjacency[a * num_groups + b] + adjacency[c * num_groups + d];
            let new = adjacency[a * num_groups + c] + adjacency[b * num_groups + d];
            if new > old {
                let gain = new - old;
                if gain > best_gain {
                    best_gain = gain;
                    best_right_edge = Some(right_edge);
                }
            }
        }

        if let Some(right_edge) = best_right_edge {
            layout[left_edge + 1..=right_edge].reverse();
            improved_any = true;
        }
    }

    improved_any
}

fn improve_layout_adjacent_swaps(
    adjacency: &[usize],
    layout: &mut [usize],
    num_groups: usize,
) -> bool {
    if layout.len() < 2 {
        return false;
    }

    let mut improved_any = false;
    for index in 0..layout.len() - 1 {
        let old = local_path_score_around(adjacency, layout, num_groups, index, index + 1);
        layout.swap(index, index + 1);
        let new = local_path_score_around(adjacency, layout, num_groups, index, index + 1);
        if new > old {
            improved_any = true;
        } else {
            layout.swap(index, index + 1);
        }
    }
    improved_any
}

fn local_path_score_around(
    adjacency: &[usize],
    layout: &[usize],
    num_groups: usize,
    left: usize,
    right: usize,
) -> usize {
    let start = left.saturating_sub(1);
    let end = (right + 1).min(layout.len().saturating_sub(1));
    (start..end)
        .map(|index| adjacency[layout[index] * num_groups + layout[index + 1]])
        .sum()
}

pub(super) fn improve_layout_2opt(adjacency: &[usize], layout: &mut [usize], num_groups: usize) {
    if layout.len() < 4 {
        return;
    }

    loop {
        let mut improved = false;
        for left_edge in 0..layout.len() - 2 {
            let a = layout[left_edge];
            let b = layout[left_edge + 1];
            for right_edge in left_edge + 2..layout.len() - 1 {
                let c = layout[right_edge];
                let d = layout[right_edge + 1];
                let old = adjacency[a * num_groups + b] + adjacency[c * num_groups + d];
                let new = adjacency[a * num_groups + c] + adjacency[b * num_groups + d];
                if new > old {
                    layout[left_edge + 1..=right_edge].reverse();
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
        if !improved {
            break;
        }
    }
}

pub(super) fn improve_layout_reinsert(adjacency: &[usize], layout: &mut Vec<usize>, num_groups: usize) {
    if layout.len() < 3 {
        return;
    }

    loop {
        let mut best_gain = 0isize;
        let mut best_from = 0usize;
        let mut best_to = 0usize;

        for from in 0..layout.len() {
            let removed = layout[from];
            let remove_loss = incident_path_score_at(adjacency, layout, num_groups, from);
            let close_gain = if from > 0 && from + 1 < layout.len() {
                adjacency[layout[from - 1] * num_groups + layout[from + 1]]
            } else {
                0
            };
            let base_gain = close_gain as isize - remove_loss as isize;

            let reduced_len = layout.len() - 1;
            for to in 0..=reduced_len {
                let insert_gain = if to == 0 {
                    let right = reduced_layout_at(layout, from, 0);
                    adjacency[removed * num_groups + right] as isize
                } else if to == reduced_len {
                    let left = reduced_layout_at(layout, from, reduced_len - 1);
                    adjacency[left * num_groups + removed] as isize
                } else {
                    let left = reduced_layout_at(layout, from, to - 1);
                    let right = reduced_layout_at(layout, from, to);
                    adjacency[left * num_groups + removed] as isize
                        + adjacency[removed * num_groups + right] as isize
                        - adjacency[left * num_groups + right] as isize
                };
                let gain = base_gain + insert_gain;
                if gain > best_gain {
                    best_gain = gain;
                    best_from = from;
                    best_to = to;
                }
            }
        }

        if best_gain <= 0 {
            break;
        }

        let value = layout.remove(best_from);
        layout.insert(best_to, value);
    }
}

fn reduced_layout_at(layout: &[usize], removed: usize, index: usize) -> usize {
    if index < removed {
        layout[index]
    } else {
        layout[index + 1]
    }
}

fn incident_path_score_at(
    adjacency: &[usize],
    layout: &[usize],
    num_groups: usize,
    index: usize,
) -> usize {
    let mut score = 0usize;
    if index > 0 {
        score += adjacency[layout[index - 1] * num_groups + layout[index]];
    }
    if index + 1 < layout.len() {
        score += adjacency[layout[index] * num_groups + layout[index + 1]];
    }
    score
}

pub(super) fn path_score(adjacency: &[usize], layout: &[usize], num_groups: usize) -> usize {
    layout
        .windows(2)
        .map(|pair| adjacency[pair[0] * num_groups + pair[1]])
        .sum()
}
