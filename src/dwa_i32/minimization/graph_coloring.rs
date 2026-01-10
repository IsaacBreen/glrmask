//! Graph coloring algorithms for DWA minimization.
//!
//! This module provides algorithms for solving graph coloring problems that arise
//! during DWA minimization. The key insight is that merging states in a DWA can be
//! viewed as a graph coloring problem: states are nodes, incompatible states have edges,
//! and the goal is to find the minimum number of colors (merged states).

use std::collections::BTreeSet;

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
    
    colors
}

/// Exact graph coloring solver - finds the optimal (minimum) number of colors.
/// 
/// Uses backtracking with pruning. The algorithm explores colorings in order,
/// pruning branches that can't improve on the current best solution.
/// 
/// **WARNING**: This has worst-case exponential time complexity O(k^n) where k is
/// the chromatic number and n is the number of nodes. Should only be used for
/// small graphs (< 30 nodes typically).
/// 
/// # Arguments
/// * `adj` - Adjacency list representation of the incompatibility graph.
/// 
/// # Returns
/// A vector of colors, one for each node, using the minimum possible number of colors.
pub fn solve_exact_graph_coloring(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    let n = adj.len();
    if n == 0 { return vec![]; }
    
    // For graphs with more than 30 nodes, fall back to greedy coloring
    // The exact solver has worst-case exponential time complexity
    // Reduced from 50 to 30 because even 45 nodes can cause 4+ second blowup on dense graphs
    if n > 30 {
        return solve_greedy_coloring(adj);
    }

    let mut colors = vec![usize::MAX; n];
    let mut best_coloring = vec![0; n];
    let mut min_colors_found = n + 1;

    // Sort by degree (high degree nodes first) for better pruning
    let mut nodes: Vec<usize> = (0..n).collect();
    nodes.sort_by_key(|&i| std::cmp::Reverse(adj[i].len()));

    fn solve(
        idx: usize,
        current_max_color: usize,
        nodes: &[usize],
        adj: &Vec<Vec<usize>>,
        colors: &mut Vec<usize>,
        min_colors_found: &mut usize,
        best_coloring: &mut Vec<usize>
    ) {
        // Prune: if we've already used as many colors as the best solution, stop
        if current_max_color >= *min_colors_found {
            return;
        }
        
        // Base case: all nodes colored
        if idx == nodes.len() {
            *min_colors_found = current_max_color;
            *best_coloring = colors.clone();
            return;
        }
        
        let u = nodes[idx];
        
        // Try each possible color (0 to current_max_color, which allows one new color)
        for c in 0..=current_max_color {
            // Check if this color conflicts with any neighbor
            let mut conflict = false;
            for &v in &adj[u] {
                if colors[v] == c {
                    conflict = true;
                    break;
                }
            }
            
            if !conflict {
                colors[u] = c;
                let next_max = std::cmp::max(current_max_color, c + 1);
                solve(idx + 1, next_max, nodes, adj, colors, min_colors_found, best_coloring);
                colors[u] = usize::MAX;
            }
        }
    }

    solve(0, 0, &nodes, adj, &mut colors, &mut min_colors_found, &mut best_coloring);
    best_coloring
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
