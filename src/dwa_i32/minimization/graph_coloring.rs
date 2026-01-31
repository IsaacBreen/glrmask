//! Graph coloring algorithms for DWA minimization.
//!
//! This module provides algorithms for solving graph coloring problems that arise
//! during DWA minimization. The key insight is that merging states in a DWA can be
//! viewed as a graph coloring problem: states are nodes, incompatible states have edges,
//! and the goal is to find the minimum number of colors (merged states).

use std::collections::BTreeSet;
use std::cell::Cell;

thread_local! {
    static CURRENT_HEIGHT: Cell<Option<usize>> = Cell::new(None);
}

pub fn set_exact_coloring_height(height: Option<usize>) {
    CURRENT_HEIGHT.with(|h| h.set(height));
}

/// Greedy graph coloring - fast O(n*m) but not necessarily optimal.
/// 
/// The algorithm processes nodes in order of decreasing degree (high-degree nodes first),
/// assigning each node the smallest color not used by its neighbors.
/// 
/// # Arguments
/// * `adj` - Adjacency list representation of the incompatibility graph.
///           adj[i] contains the indices of all nodes that are incompatible with node i.
/// 
/// # Returns
/// A vector of colors, one for each node. Nodes with the same color can be merged.
pub fn solve_greedy_coloring(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    let n = adj.len();
    if n == 0 { return vec![]; }

    let start = std::time::Instant::now();
    let mut colors = vec![usize::MAX; n];
    
    // Sort by degree (high degree nodes first) - this heuristic often gives better results
    let mut nodes: Vec<usize> = (0..n).collect();
    nodes.sort_by_key(|&i| std::cmp::Reverse(adj[i].len()));

    for &u in &nodes {
        // Find smallest color not used by neighbors
        let neighbor_colors: BTreeSet<usize> = 
            adj[u].iter().filter_map(|&v| {
                if colors[v] != usize::MAX { Some(colors[v]) } else { None }
            }).collect();
        
        let mut c = 0;
        while neighbor_colors.contains(&c) {
            c += 1;
        }
        colors[u] = c;
    }
    
    let num_colors = colors.iter().max().map(|&c| c + 1).unwrap_or(0);
    if n >= 100 {
        crate::debug!(5, "Greedy graph coloring: {} nodes → {} colors in {:?}", 
            n, num_colors, start.elapsed());
    }
    
    colors
}

fn has_clique_of_size(adj: &Vec<Vec<usize>>, target: usize) -> bool {
    let n = adj.len();
    if target <= 1 {
        return n > 0;
    }
    if n == 0 {
        return false;
    }

    let words = (n + 63) / 64;
    let mut adj_bits = vec![vec![0u64; words]; n];
    for u in 0..n {
        for &v in &adj[u] {
            let word = v / 64;
            let bit = v % 64;
            adj_bits[u][word] |= 1u64 << bit;
        }
    }

    fn popcount(set: &[u64]) -> usize {
        set.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn intersect(a: &[u64], b: &[u64]) -> Vec<u64> {
        a.iter().zip(b.iter()).map(|(x, y)| x & y).collect()
    }

    fn difference(a: &[u64], b: &[u64]) -> Vec<u64> {
        a.iter().zip(b.iter()).map(|(x, y)| x & !y).collect()
    }

    fn clear_bit(set: &mut [u64], idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        set[word] &= !(1u64 << bit);
    }

    fn iter_bits(set: &[u64]) -> Vec<usize> {
        let mut indices = Vec::new();
        for (word_idx, &word) in set.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                indices.push(word_idx * 64 + bit);
                w &= w - 1;
            }
        }
        indices
    }

    fn intersection_count(a: &[u64], b: &[u64]) -> usize {
        a.iter().zip(b.iter()).map(|(x, y)| (x & y).count_ones() as usize).sum()
    }

    fn search(
        size: usize,
        mut p: Vec<u64>,
        adj_bits: &Vec<Vec<u64>>,
        target: usize,
    ) -> bool {
        if size >= target {
            return true;
        }

        if size + popcount(&p) < target {
            return false;
        }

        let mut pivot = None;
        let mut pivot_deg = 0usize;
        for v in iter_bits(&p) {
            let deg = intersection_count(&adj_bits[v], &p);
            if deg > pivot_deg {
                pivot_deg = deg;
                pivot = Some(v);
            }
        }
        let pivot = pivot.unwrap_or_else(|| iter_bits(&p).first().copied().unwrap_or(0));
        let mut candidates = difference(&p, &adj_bits[pivot]);

        for v in iter_bits(&candidates) {
            let new_p = intersect(&p, &adj_bits[v]);
            if search(size + 1, new_p, adj_bits, target) {
                return true;
            }
            clear_bit(&mut p, v);
            clear_bit(&mut candidates, v);

            if size + popcount(&p) < target {
                return false;
            }
        }
        false
    }

    let mut all = vec![!0u64; words];
    let extra_bits = n % 64;
    if extra_bits != 0 {
        all[words - 1] &= (1u64 << extra_bits) - 1;
    }

    search(0, all, &adj_bits, target)
}

/// Exact graph coloring solver - finds the OPTIMAL (minimum) number of colors.
pub fn solve_exact_graph_coloring(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    solve_exact_graph_coloring_with_stats(adj).0
}

/// Exact graph coloring solver with stats (greedy upper bound + clique check).
///
/// **CRITICAL**: This function MUST be exact. Do NOT add fallbacks to greedy
/// algorithms or heuristics. If performance is a concern, use solve_greedy_coloring()
/// instead, but NEVER compromise the exactness of this function.
///
/// For performance-sensitive contexts, use FastMinimize which intentionally
/// uses greedy methods. ExactMinimize is for when optimality is required.
///
/// Uses backtracking with pruning. The algorithm explores colorings in order,
/// pruning branches that can't improve on the current best solution.
///
/// **WARNING**: This has worst-case exponential time complexity O(k^n) where k is
/// the chromatic number and n is the number of nodes.
///
/// # Arguments
/// * `adj` - Adjacency list representation of the incompatibility graph.
///
/// # Returns
/// Returns (colors, greedy_upper_bound, clique_found).
pub fn solve_exact_graph_coloring_with_stats(adj: &Vec<Vec<usize>>) -> (Vec<usize>, usize, bool) {
    let n = adj.len();
    if n == 0 { return (vec![], 0, true); }

    let height_opt = CURRENT_HEIGHT.with(|h| h.get());
    if let Some(height) = height_opt {
        eprintln!("Height {}: starting", height);
    }

    let start = std::time::Instant::now();

    let degrees: Vec<usize> = adj.iter().map(|v| v.len()).collect();

    fn select_vertex(
        colors: &[usize],
        saturation: &[usize],
        degrees: &[usize],
    ) -> Option<usize> {
        let mut best: Option<usize> = None;
        for i in 0..colors.len() {
            if colors[i] != usize::MAX {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) => {
                    let sat_i = saturation[i];
                    let sat_b = saturation[b];
                    if sat_i > sat_b || (sat_i == sat_b && degrees[i] > degrees[b]) {
                        best = Some(i);
                    }
                }
            }
        }
        best
    }

    fn solve_dsatur_greedy(adj: &Vec<Vec<usize>>, degrees: &[usize]) -> Vec<usize> {
        let n = adj.len();
        let mut colors = vec![usize::MAX; n];
        let mut saturation = vec![0usize; n];
        let mut neighbor_color_flags: Vec<Vec<bool>> = vec![Vec::new(); n];
        let mut max_colors = 0usize;

        for _ in 0..n {
            let Some(u) = select_vertex(&colors, &saturation, degrees) else { break; };

            // Find the smallest available color
            let mut c = 0usize;
            while c < max_colors {
                if !neighbor_color_flags[u][c] {
                    break;
                }
                c += 1;
            }

            if c == max_colors {
                max_colors += 1;
                for flags in neighbor_color_flags.iter_mut() {
                    flags.push(false);
                }
            }

            colors[u] = c;
            for &v in &adj[u] {
                if !neighbor_color_flags[v][c] {
                    neighbor_color_flags[v][c] = true;
                    saturation[v] += 1;
                }
            }
        }

        colors
    }

    fn solve_smallest_last_greedy(adj: &Vec<Vec<usize>>) -> Vec<usize> {
        let n = adj.len();
        let mut degrees: Vec<usize> = adj.iter().map(|v| v.len()).collect();
        let mut remaining = vec![true; n];
        let mut order = Vec::with_capacity(n);

        for _ in 0..n {
            let mut min_deg = usize::MAX;
            let mut min_v = None;
            for i in 0..n {
                if remaining[i] && degrees[i] < min_deg {
                    min_deg = degrees[i];
                    min_v = Some(i);
                }
            }
            let v = min_v.unwrap();
            remaining[v] = false;
            order.push(v);
            for &u in &adj[v] {
                if remaining[u] {
                    degrees[u] -= 1;
                }
            }
        }

        order.reverse();
        let mut colors = vec![usize::MAX; n];
        let mut used = vec![false; n];
        for &v in &order {
            used.fill(false);
            for &u in &adj[v] {
                let c = colors[u];
                if c != usize::MAX {
                    used[c] = true;
                }
            }
            let mut c = 0usize;
            while c < n && used[c] {
                c += 1;
            }
            colors[v] = c;
        }

        colors
    }

    // Upper bound from greedy colorings (degree-ordered and DSATUR-ordered)
    let greedy_start = std::time::Instant::now();
    if let Some(height) = height_opt {
        eprintln!("Height {}: starting greedy", height);
    }
    let greedy_colors = solve_greedy_coloring(adj);
    let dsatur_colors = solve_dsatur_greedy(adj, &degrees);
    let smallest_last_colors = solve_smallest_last_greedy(adj);
    let greedy_num = greedy_colors.iter().max().map(|&c| c + 1).unwrap_or(0);
    let dsatur_num = dsatur_colors.iter().max().map(|&c| c + 1).unwrap_or(0);
    let smallest_last_num = smallest_last_colors.iter().max().map(|&c| c + 1).unwrap_or(0);
    let mut best_coloring = if dsatur_num <= greedy_num && dsatur_num <= smallest_last_num {
        dsatur_colors
    } else if smallest_last_num <= greedy_num {
        smallest_last_colors
    } else {
        greedy_colors
    };
    let mut best_num = greedy_num.min(dsatur_num).min(smallest_last_num);
    let greedy_time = greedy_start.elapsed();
    if let Some(height) = height_opt {
        eprintln!(
            "Height {}: greedy done in {:?}, greedy_ub={}, dsatur_ub={}, smallest_last_ub={}, best_ub={}",
            height,
            greedy_time,
            greedy_num,
            dsatur_num,
            smallest_last_num,
            best_num,
        );
    }

    // Exact lower bound from maximum clique size
        // Exact early-exit: if a clique of size == upper bound exists, greedy is optimal
        let max_clique_ub = std::env::var("DWA_MAX_CLIQUE_UB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(25);
        let mut clique_checked = false;
        let mut clique_found = false;
        let mut clique_time = std::time::Duration::ZERO;
        if best_num <= max_clique_ub {
            clique_checked = true;
            if let Some(height) = height_opt {
                eprintln!("Height {}: starting clique check", height);
            }
            let clique_start = std::time::Instant::now();
            clique_found = has_clique_of_size(adj, best_num);
            clique_time = clique_start.elapsed();
            if let Some(height) = height_opt {
                eprintln!(
                    "Height {}: clique check done in {:?}, found={}",
                    height,
                    clique_time,
                    clique_found,
                );
            }
        } else if let Some(height) = height_opt {
            eprintln!(
                "Height {}: skipping clique check (greedy_ub {} > max_clique_ub {})",
                height,
                best_num,
                max_clique_ub,
            );
        }
    if std::env::var("DWA_TRACE_HEIGHTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        eprintln!(
                "TRACE: exact bounds nodes={} clique_found={} greedy_ub={} dsatur_ub={} smallest_last_ub={}",
            n,
                clique_found,
            greedy_num,
            dsatur_num,
            smallest_last_num,
        );
    }
    if let Some(height) = height_opt {
        let edge_count = adj.iter().map(|v| v.len()).sum::<usize>() / 2;
        eprintln!(
            "Height {}: nodes={}, edges={}, greedy_ub={}, clique_found={}",
            height,
            n,
            edge_count,
            best_num,
            clique_found,
        );
        if clique_checked {
            eprintln!(
                "Height {}: clique check took {:?}, found={}",
                height,
                clique_time,
                clique_found,
            );
        }
    }
        if clique_found {
        let elapsed = start.elapsed();
        if elapsed.as_millis() > 10 {
            crate::debug!(5, "Exact graph coloring: {} nodes → {} colors in {:?} (clique-bound)",
                n, best_num, elapsed);
        }
            return (best_coloring, best_num, true);
    }

        let dsatur_start = std::time::Instant::now();
        if let Some(height) = height_opt {
            eprintln!("Height {}: entering DSATUR solver", height);
        }

    let mut colors = vec![usize::MAX; n];
    let mut saturation = vec![0usize; n];
    let mut neighbor_color_counts = vec![vec![0u32; best_num.max(1)]; n];

    fn assign_color(
        u: usize,
        color: usize,
        adj: &Vec<Vec<usize>>,
        colors: &mut Vec<usize>,
        saturation: &mut Vec<usize>,
        neighbor_color_counts: &mut Vec<Vec<u32>>,
    ) {
        colors[u] = color;
        for &v in &adj[u] {
            let counts = &mut neighbor_color_counts[v][color];
            *counts += 1;
            if *counts == 1 {
                saturation[v] += 1;
            }
        }
    }

    fn unassign_color(
        u: usize,
        color: usize,
        adj: &Vec<Vec<usize>>,
        colors: &mut Vec<usize>,
        saturation: &mut Vec<usize>,
        neighbor_color_counts: &mut Vec<Vec<u32>>,
    ) {
        for &v in &adj[u] {
            let counts = &mut neighbor_color_counts[v][color];
            *counts -= 1;
            if *counts == 0 {
                saturation[v] -= 1;
            }
        }
        colors[u] = usize::MAX;
    }

    fn dsatur_search(
        colored_count: usize,
        num_colors_used: usize,
        adj: &Vec<Vec<usize>>,
        degrees: &[usize],
        colors: &mut Vec<usize>,
        saturation: &mut Vec<usize>,
        neighbor_color_counts: &mut Vec<Vec<u32>>,
        best_num: &mut usize,
        best_coloring: &mut Vec<usize>,
    ) {
        if colored_count == colors.len() {
            if num_colors_used < *best_num {
                *best_num = num_colors_used;
                *best_coloring = colors.clone();
            }
            return;
        }

        if num_colors_used >= *best_num {
            return;
        }

        let Some(u) = select_vertex(colors, saturation, degrees) else { return; };

        // Try existing colors first
        for c in 0..num_colors_used {
            if neighbor_color_counts[u][c] == 0 {
                assign_color(u, c, adj, colors, saturation, neighbor_color_counts);
                dsatur_search(
                    colored_count + 1,
                    num_colors_used,
                    adj,
                    degrees,
                    colors,
                    saturation,
                    neighbor_color_counts,
                    best_num,
                    best_coloring,
                );
                unassign_color(u, c, adj, colors, saturation, neighbor_color_counts);
            }
        }

        // Try a new color if it could still beat the best
        if num_colors_used + 1 < *best_num {
            let new_color = num_colors_used;
            assign_color(u, new_color, adj, colors, saturation, neighbor_color_counts);
            dsatur_search(
                colored_count + 1,
                num_colors_used + 1,
                adj,
                degrees,
                colors,
                saturation,
                neighbor_color_counts,
                best_num,
                best_coloring,
            );
            unassign_color(u, new_color, adj, colors, saturation, neighbor_color_counts);
        }
    }

    dsatur_search(
        0,
        0,
        adj,
        &degrees,
        &mut colors,
        &mut saturation,
        &mut neighbor_color_counts,
        &mut best_num,
        &mut best_coloring,
    );

    let dsatur_time = dsatur_start.elapsed();
    if let Some(height) = height_opt {
        eprintln!(
            "Height {}: DSATUR done in {:?}, colors={}",
            height,
            dsatur_time,
            best_num,
        );
    }

    let elapsed = start.elapsed();
    if elapsed.as_millis() > 10 {
        crate::debug!(5, "Exact graph coloring: {} nodes → {} colors in {:?}",
            n, best_num, elapsed);
    }

    (best_coloring, best_num, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_greedy_empty() {
        let adj: Vec<Vec<usize>> = vec![];
        let colors = solve_greedy_coloring(&adj);
        assert!(colors.is_empty());
    }
    
    #[test]
    fn test_greedy_single() {
        let adj = vec![vec![]];
        let colors = solve_greedy_coloring(&adj);
        assert_eq!(colors, vec![0]);
    }
    
    #[test]
    fn test_greedy_two_incompatible() {
        // Two nodes connected by an edge
        let adj = vec![vec![1], vec![0]];
        let colors = solve_greedy_coloring(&adj);
        assert!(colors[0] != colors[1]);
        assert_eq!(*colors.iter().max().unwrap(), 1); // Uses exactly 2 colors
    }
    
    #[test]
    fn test_greedy_two_compatible() {
        // Two nodes with no edge
        let adj = vec![vec![], vec![]];
        let colors = solve_greedy_coloring(&adj);
        assert_eq!(colors[0], colors[1]); // Same color since compatible
    }
    
    #[test]
    fn test_greedy_triangle() {
        // Complete graph on 3 nodes - needs 3 colors
        let adj = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let colors = solve_greedy_coloring(&adj);
        assert!(colors[0] != colors[1]);
        assert!(colors[1] != colors[2]);
        assert!(colors[0] != colors[2]);
    }
    
    #[test]
    fn test_exact_empty() {
        let adj: Vec<Vec<usize>> = vec![];
        let colors = solve_exact_graph_coloring(&adj);
        assert!(colors.is_empty());
    }
    
    #[test]
    fn test_exact_single() {
        let adj = vec![vec![]];
        let colors = solve_exact_graph_coloring(&adj);
        assert_eq!(colors, vec![0]);
    }
    
    #[test]
    fn test_exact_two_incompatible() {
        let adj = vec![vec![1], vec![0]];
        let colors = solve_exact_graph_coloring(&adj);
        assert!(colors[0] != colors[1]);
        assert_eq!(*colors.iter().max().unwrap(), 1);
    }
    
    #[test]
    fn test_exact_triangle() {
        let adj = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let colors = solve_exact_graph_coloring(&adj);
        assert!(colors[0] != colors[1]);
        assert!(colors[1] != colors[2]);
        assert!(colors[0] != colors[2]);
        assert_eq!(*colors.iter().max().unwrap(), 2);
    }
    
    #[test]
    fn test_exact_bipartite() {
        // Bipartite graph: nodes 0,1 on one side, 2,3 on other
        // Edges: 0-2, 0-3, 1-2, 1-3
        let adj = vec![
            vec![2, 3],    // 0
            vec![2, 3],    // 1
            vec![0, 1],    // 2
            vec![0, 1],    // 3
        ];
        let colors = solve_exact_graph_coloring(&adj);
        // Bipartite graph needs exactly 2 colors
        assert_eq!(colors[0], colors[1]); // Same side
        assert_eq!(colors[2], colors[3]); // Same side
        assert!(colors[0] != colors[2]);  // Different sides
        assert_eq!(*colors.iter().max().unwrap(), 1);
    }
}
