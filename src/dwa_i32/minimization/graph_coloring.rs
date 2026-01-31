//! Graph coloring algorithms for DWA minimization.
//!
//! This module provides algorithms for solving graph coloring problems that arise
//! during DWA minimization. The key insight is that merging states in a DWA can be
//! viewed as a graph coloring problem: states are nodes, incompatible states have edges,
//! and the goal is to find the minimum number of colors (merged states).

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};
use std::os::raw::c_int;

use cadical::{Solver as CadicalSolver, Timeout as CadicalTimeout};
use varisat::{ExtendFormula, Solver, Var};

thread_local! {
    static CURRENT_HEIGHT: Cell<Option<usize>> = Cell::new(None);
}

extern "C" {
    fn colpack_color_graph(
        row_offsets: *const c_int,
        col_indices: *const c_int,
        num_vertices: c_int,
        num_edges: c_int,
        out_colors: *mut c_int,
        out_color_count: *mut c_int,
    ) -> c_int;
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

fn normalize_colors(colors: Vec<usize>) -> Vec<usize> {
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    colors
        .into_iter()
        .map(|c| {
            *mapping.entry(c).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        })
        .collect()
}

fn is_valid_coloring(adj: &Vec<Vec<usize>>, colors: &[usize]) -> bool {
    for (u, neighbors) in adj.iter().enumerate() {
        let c_u = colors[u];
        for &v in neighbors {
            if c_u == colors[v] {
                return false;
            }
        }
    }
    true
}

/// ColPack-based graph coloring (greedy heuristic via ColPack).
pub fn solve_colpack_coloring(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    let n = adj.len();
    if n == 0 {
        return vec![];
    }

    let mut row_offsets: Vec<c_int> = Vec::with_capacity(n + 1);
    let mut col_indices: Vec<c_int> = Vec::new();
    row_offsets.push(0);
    for neighbors in adj {
        for &v in neighbors {
            col_indices.push(v as c_int);
        }
        row_offsets.push(col_indices.len() as c_int);
    }

    let mut colors_out = vec![0 as c_int; n];
    let mut color_count: c_int = 0;
    let rc = unsafe {
        colpack_color_graph(
            row_offsets.as_ptr(),
            col_indices.as_ptr(),
            n as c_int,
            col_indices.len() as c_int,
            colors_out.as_mut_ptr(),
            &mut color_count as *mut c_int,
        )
    };

    if rc != 0 {
        eprintln!("ColPack coloring failed (rc={}), falling back to greedy", rc);
        return solve_greedy_coloring(adj);
    }

    let colors: Vec<usize> = colors_out.into_iter().map(|c| c as usize).collect();
    let colors = normalize_colors(colors);
    if !is_valid_coloring(adj, &colors) {
        eprintln!("ColPack produced invalid coloring, falling back to greedy");
        return solve_greedy_coloring(adj);
    }
    colors
}

enum ColpackVerification {
    VerifiedOptimal,
    FoundBetter(Vec<usize>),
    TimedOut,
}

fn verify_colpack_optimality(
    adj: &Vec<Vec<usize>>,
    k: usize,
) -> ColpackVerification {
    if k <= 1 {
        return ColpackVerification::VerifiedOptimal;
    }

    let (mut solver, vars) = build_cadical_coloring_solver(adj, k - 1);
    solver.set_callbacks(Some(CadicalTimeout::new(1.0)));
    let result = solver.solve();
    match result {
        Some(true) => ColpackVerification::FoundBetter(extract_cadical_coloring(&solver, &vars, k - 1)),
        Some(false) => ColpackVerification::VerifiedOptimal,
        None => ColpackVerification::TimedOut,
    }
}

/// ColPack coloring with bounded SAT verification (1s per height).
pub fn solve_colpack_with_verification(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    let mut colors = solve_colpack_coloring(adj);
    let k = colors.iter().max().map(|&c| c + 1).unwrap_or(0);
    if k <= 1 {
        return colors;
    }

    let height_opt = CURRENT_HEIGHT.with(|h| h.get());
    let verify_start = std::time::Instant::now();
    let verification = verify_colpack_optimality(adj, k);
    let verify_time = verify_start.elapsed();

    match verification {
        ColpackVerification::VerifiedOptimal => {
            if let Some(height) = height_opt {
                eprintln!(
                    "Height {}: ColPack verified optimal (k={}) in {:?}",
                    height,
                    k,
                    verify_time,
                );
            } else {
                eprintln!("ColPack verified optimal (k={}) in {:?}", k, verify_time);
            }
        }
        ColpackVerification::FoundBetter(better) => {
            if let Some(height) = height_opt {
                eprintln!(
                    "Height {}: ColPack not optimal; SAT found k={} in {:?}",
                    height,
                    k - 1,
                    verify_time,
                );
            } else {
                eprintln!("ColPack not optimal; SAT found k={} in {:?}", k - 1, verify_time);
            }
            colors = better;
        }
        ColpackVerification::TimedOut => {
            if let Some(height) = height_opt {
                eprintln!(
                    "Height {}: ColPack verification timed out (k={}) after {:?}",
                    height,
                    k - 1,
                    verify_time,
                );
            } else {
                eprintln!("ColPack verification timed out (k={}) after {:?}", k - 1, verify_time);
            }
        }
    }

    colors
}

/// Exact graph coloring solver (SAT) - finds the OPTIMAL (minimum) number of colors.
pub fn solve_exact_graph_coloring(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    solve_exact_graph_coloring_with_stats(adj).0
}

/// Exact graph coloring solver (SAT) using CaDiCaL.
pub fn solve_exact_graph_coloring_cadical(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    solve_exact_graph_coloring_with_stats_cadical(adj).0
}

pub fn solve_exact_graph_coloring_with_stats(adj: &Vec<Vec<usize>>) -> (Vec<usize>, usize) {
    solve_exact_graph_coloring_with_stats_impl(adj, solve_sat_exact_varisat)
}

pub fn solve_exact_graph_coloring_with_stats_cadical(adj: &Vec<Vec<usize>>) -> (Vec<usize>, usize) {
    solve_exact_graph_coloring_with_stats_impl(adj, solve_sat_exact_cadical)
}

/// Exact graph coloring solver with stats (greedy upper bound + SAT search).
///
/// **CRITICAL**: This function MUST be exact. Do NOT add fallbacks to greedy
/// algorithms or heuristics. If performance is a concern, use solve_greedy_coloring()
/// instead, but NEVER compromise the exactness of this function.
///
/// For performance-sensitive contexts, use FastMinimize which intentionally
/// uses greedy methods. SatMinimize/DsaturMinimize are for when optimality is required.
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
/// Returns (colors, greedy_upper_bound).
fn solve_exact_graph_coloring_with_stats_impl<F>(
    adj: &Vec<Vec<usize>>,
    solve_sat_exact: F,
) -> (Vec<usize>, usize)
where
    F: FnOnce(&Vec<Vec<usize>>, usize, Option<usize>) -> (Vec<usize>, usize),
{
    let n = adj.len();
    if n == 0 { return (vec![], 0); }

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

    fn has_clique_of_size_bounded(
        adj: &Vec<Vec<usize>>,
        target: usize,
        time_limit: std::time::Duration,
    ) -> bool {
        let n = adj.len();
        if target <= 1 {
            return n >= target;
        }
        if target > n {
            return false;
        }

        let start = std::time::Instant::now();
        let mut adj_matrix = vec![vec![false; n]; n];
        for u in 0..n {
            for &v in &adj[u] {
                adj_matrix[u][v] = true;
            }
        }

        let mut vertices: Vec<usize> = (0..n).collect();
        vertices.sort_by_key(|&v| std::cmp::Reverse(adj[v].len()));

        fn dfs(
            clique: &mut Vec<usize>,
            candidates: &[usize],
            target: usize,
            adj: &Vec<Vec<bool>>,
            start: std::time::Instant,
            time_limit: std::time::Duration,
        ) -> bool {
            if start.elapsed() >= time_limit {
                return false;
            }
            if clique.len() == target {
                return true;
            }
            if clique.len() + candidates.len() < target {
                return false;
            }

            for (idx, &v) in candidates.iter().enumerate() {
                if start.elapsed() >= time_limit {
                    return false;
                }
                let mut next_candidates = Vec::new();
                for &u in candidates.iter().skip(idx + 1) {
                    if adj[v][u] {
                        next_candidates.push(u);
                    }
                }
                clique.push(v);
                if dfs(clique, &next_candidates, target, adj, start, time_limit) {
                    return true;
                }
                clique.pop();
            }

            false
        }

        let mut clique = Vec::with_capacity(target);
        dfs(
            &mut clique,
            &vertices,
            target,
            &adj_matrix,
            start,
            time_limit,
        )
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

    if std::env::var("DWA_TRACE_HEIGHTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        eprintln!(
            "TRACE: exact bounds nodes={} greedy_ub={} dsatur_ub={} smallest_last_ub={}",
            n,
            greedy_num,
            dsatur_num,
            smallest_last_num,
        );
    }
    if let Some(height) = height_opt {
        let edge_count = adj.iter().map(|v| v.len()).sum::<usize>() / 2;
        eprintln!(
            "Height {}: nodes={}, edges={}, greedy_ub={}",
            height,
            n,
            edge_count,
            best_num,
        );
    }

    if best_num <= 1 {
        return (best_coloring, best_num);
    }

    let clique_timeout_ms = std::env::var("DWA_CLIQUE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30000);
    let clique_start = std::time::Instant::now();
    if let Some(height) = height_opt {
        eprintln!(
            "Height {}: clique check start target={} budget={}ms",
            height,
            best_num,
            clique_timeout_ms,
        );
    }
    let found_clique = has_clique_of_size_bounded(
        adj,
        best_num,
        std::time::Duration::from_millis(clique_timeout_ms),
    );
    if let Some(height) = height_opt {
        eprintln!(
            "Height {}: clique check {} in {:?}",
            height,
            if found_clique { "found" } else { "not found" },
            clique_start.elapsed(),
        );
    }
    if found_clique {
        return (best_coloring, best_num);
    }

    let sat_start = std::time::Instant::now();
    if let Some(height) = height_opt {
        eprintln!(
            "Height {}: entering SAT solver (upper_bound={})",
            height,
            best_num,
        );
    }

    let (best_coloring, best_num) = solve_sat_exact(adj, best_num, height_opt);

    if let Some(height) = height_opt {
        eprintln!(
            "Height {}: SAT solver done in {:?}, colors={}",
            height,
            sat_start.elapsed(),
            best_num,
        );
    }

    let elapsed = start.elapsed();
    if elapsed.as_millis() > 10 {
        crate::debug!(5, "Exact graph coloring: {} nodes → {} colors in {:?}",
            n, best_num, elapsed);
    }

    (best_coloring, best_num)
}

fn solve_sat_exact_varisat(
    adj: &Vec<Vec<usize>>,
    max_colors: usize,
    height_opt: Option<usize>,
) -> (Vec<usize>, usize) {
    let n = adj.len();
    let mut solver = Solver::new();
    let mut vars: Vec<Vec<Var>> = Vec::with_capacity(n);
    for _ in 0..n {
        vars.push(Vec::with_capacity(max_colors));
    }
    for v in 0..n {
        for _ in 0..max_colors {
            vars[v].push(solver.new_var());
        }
    }

    for v in 0..n {
        for c in (v + 1)..max_colors {
            solver.add_clause(&[vars[v][c].negative()]);
        }
    }

    for v in 0..n {
        let clause: Vec<_> = vars[v].iter().map(|&var| var.positive()).collect();
        solver.add_clause(&clause);
    }

    for v in 0..n {
        for c1 in 0..max_colors {
            for c2 in (c1 + 1)..max_colors {
                solver.add_clause(&[vars[v][c1].negative(), vars[v][c2].negative()]);
            }
        }
    }

    for u in 0..n {
        for &v in &adj[u] {
            if v > u {
                for c in 0..max_colors {
                    solver.add_clause(&[vars[u][c].negative(), vars[v][c].negative()]);
                }
            }
        }
    }

    if n > 0 && max_colors > 0 {
        solver.add_clause(&[vars[0][0].positive()]);
    }

    fn solve_k(solver: &mut Solver, vars: &Vec<Vec<Var>>, k: usize) -> Option<Vec<usize>> {
        let n = vars.len();
        if n == 0 {
            return Some(vec![]);
        }
        let max_colors = vars[0].len();
        let mut assumptions = Vec::with_capacity(n * (max_colors - k));
        for v in 0..n {
            for c in k..max_colors {
                assumptions.push(vars[v][c].negative());
            }
        }
        solver.assume(&assumptions);
        let sat = solver.solve().expect("SAT solver failed");
        if !sat {
            return None;
        }
        let model = solver.model().expect("SAT model missing");
        let mut assignment = vec![false; n * max_colors];
        for lit in model {
            assignment[lit.index()] = lit.is_positive();
        }
        let mut colors = vec![0usize; n];
        for v in 0..n {
            let mut assigned = None;
            for c in 0..k {
                let var = vars[v][c];
                if assignment[var.index()] {
                    assigned = Some(c);
                    break;
                }
            }
            let color = assigned.expect("SAT model missing vertex color");
            colors[v] = color;
        }
        Some(colors)
    }

    let mut k = max_colors;
    let start_k = std::time::Instant::now();
    if let Some(height) = height_opt {
        eprintln!("Height {}: SAT attempting k={}", height, k);
    } else {
        eprintln!("SAT attempting k={}", k);
    }
    let mut best_coloring = solve_k(&mut solver, &vars, k)
        .expect("SAT solver returned UNSAT at upper bound");
    let start_k_ms = start_k.elapsed().as_millis();
    if let Some(height) = height_opt {
        eprintln!("Height {}: SAT k={}: SAT in {} ms", height, k, start_k_ms);
    } else {
        eprintln!("SAT k={}: SAT in {} ms", k, start_k_ms);
    }

    while k > 1 {
        let candidate = k - 1;
        let step_start = std::time::Instant::now();
        if let Some(height) = height_opt {
            eprintln!("Height {}: SAT attempting k={}", height, candidate);
        } else {
            eprintln!("SAT attempting k={}", candidate);
        }
        let sat_colors = solve_k(&mut solver, &vars, candidate);
        let step_time_ms = step_start.elapsed().as_millis();
        let step_status = if sat_colors.is_some() { "SAT" } else { "UNSAT" };
        if let Some(height) = height_opt {
            eprintln!(
                "Height {}: SAT k={}: {} in {} ms",
                height,
                candidate,
                step_status,
                step_time_ms,
            );
        } else {
            eprintln!("SAT k={}: {} in {} ms", candidate, step_status, step_time_ms);
        }
        if let Some(colors) = sat_colors {
            k = candidate;
            best_coloring = colors;
        } else {
            break;
        }
    }

    if let Some(height) = height_opt {
        eprintln!("Height {}: SAT final k={}", height, k);
    }

    (best_coloring, k)
}

fn build_cadical_coloring_solver(
    adj: &Vec<Vec<usize>>,
    max_colors: usize,
) -> (CadicalSolver, Vec<Vec<i32>>) {
    let n = adj.len();
    let mut solver = CadicalSolver::new();
    let mut vars: Vec<Vec<i32>> = Vec::with_capacity(n);
    let mut next_var: i32 = 1;
    for _ in 0..n {
        let mut row = Vec::with_capacity(max_colors);
        for _ in 0..max_colors {
            row.push(next_var);
            next_var += 1;
        }
        vars.push(row);
    }

    for v in 0..n {
        for c in (v + 1)..max_colors {
            solver.add_clause([-vars[v][c]]);
        }
    }

    for v in 0..n {
        solver.add_clause(vars[v].iter().copied());
    }

    for v in 0..n {
        for c1 in 0..max_colors {
            for c2 in (c1 + 1)..max_colors {
                solver.add_clause([-vars[v][c1], -vars[v][c2]]);
            }
        }
    }

    for u in 0..n {
        for &v in &adj[u] {
            if v > u {
                for c in 0..max_colors {
                    solver.add_clause([-vars[u][c], -vars[v][c]]);
                }
            }
        }
    }

    if n > 0 && max_colors > 0 {
        solver.add_clause([vars[0][0]]);
    }

    (solver, vars)
}

fn extract_cadical_coloring(
    solver: &CadicalSolver,
    vars: &Vec<Vec<i32>>,
    k: usize,
) -> Vec<usize> {
    let n = vars.len();
    let mut colors = vec![0usize; n];
    for v in 0..n {
        let mut assigned = None;
        for c in 0..k {
            if solver.value(vars[v][c]).unwrap_or(false) {
                assigned = Some(c);
                break;
            }
        }
        let color = assigned.expect("CaDiCaL model missing vertex color");
        colors[v] = color;
    }
    colors
}

fn sat_color_cadical(
    solver: &mut CadicalSolver,
    vars: &Vec<Vec<i32>>,
    k: usize,
) -> Option<Vec<usize>> {
    let n = vars.len();
    if n == 0 {
        return Some(vec![]);
    }
    let max_colors = vars[0].len();
    let mut assumptions = Vec::with_capacity(n * (max_colors - k));
    for v in 0..n {
        for c in k..max_colors {
            assumptions.push(-vars[v][c]);
        }
    }
    let sat = solver
        .solve_with(assumptions.iter().copied())
        .expect("CaDiCaL solver failed");
    if !sat {
        return None;
    }
    Some(extract_cadical_coloring(solver, vars, k))
}

fn solve_sat_exact_cadical(
    adj: &Vec<Vec<usize>>,
    max_colors: usize,
    height_opt: Option<usize>,
) -> (Vec<usize>, usize) {
    let (mut solver, vars) = build_cadical_coloring_solver(adj, max_colors);

    let mut k = max_colors;
    let start_k = std::time::Instant::now();
    if let Some(height) = height_opt {
        eprintln!("Height {}: SAT attempting k={}", height, k);
    } else {
        eprintln!("SAT attempting k={}", k);
    }
    let mut best_coloring = sat_color_cadical(&mut solver, &vars, k)
        .expect("CaDiCaL solver returned UNSAT at upper bound");
    let start_k_ms = start_k.elapsed().as_millis();
    if let Some(height) = height_opt {
        eprintln!("Height {}: SAT k={}: SAT in {} ms", height, k, start_k_ms);
    } else {
        eprintln!("SAT k={}: SAT in {} ms", k, start_k_ms);
    }

    while k > 1 {
        let candidate = k - 1;
        let step_start = std::time::Instant::now();
        if let Some(height) = height_opt {
            eprintln!("Height {}: SAT attempting k={}", height, candidate);
        } else {
            eprintln!("SAT attempting k={}", candidate);
        }
        let sat_colors = sat_color_cadical(&mut solver, &vars, candidate);
        let step_time_ms = step_start.elapsed().as_millis();
        let step_status = if sat_colors.is_some() { "SAT" } else { "UNSAT" };
        if let Some(height) = height_opt {
            eprintln!(
                "Height {}: SAT k={}: {} in {} ms",
                height,
                candidate,
                step_status,
                step_time_ms,
            );
        } else {
            eprintln!("SAT k={}: {} in {} ms", candidate, step_status, step_time_ms);
        }
        if let Some(colors) = sat_colors {
            k = candidate;
            best_coloring = colors;
        } else {
            break;
        }
    }

    if let Some(height) = height_opt {
        eprintln!("Height {}: SAT final k={}", height, k);
    }

    (best_coloring, k)
}

/// Exact graph coloring solver using DSATUR branch-and-bound.
pub fn solve_exact_graph_coloring_dsatur(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    let n = adj.len();
    if n == 0 {
        return vec![];
    }

    eprintln!("DSATUR exact: graph nodes={}", n);

    let degrees: Vec<usize> = adj.iter().map(|v| v.len()).collect();

    fn select_vertex(colors: &[usize], saturation: &[usize], degrees: &[usize]) -> Option<usize> {
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

    let greedy_colors = solve_greedy_coloring(adj);
    let dsatur_colors = solve_dsatur_greedy(adj, &degrees);
    let smallest_last_colors = solve_smallest_last_greedy(adj);
    let greedy_num = greedy_colors.iter().max().map(|&c| c + 1).unwrap_or(0);
    let dsatur_num = dsatur_colors.iter().max().map(|&c| c + 1).unwrap_or(0);
    let smallest_last_num = smallest_last_colors.iter().max().map(|&c| c + 1).unwrap_or(0);
    let mut best_colors = if dsatur_num <= greedy_num && dsatur_num <= smallest_last_num {
        dsatur_colors
    } else if smallest_last_num <= greedy_num {
        smallest_last_colors
    } else {
        greedy_colors
    };
    let mut best_num = greedy_num.min(dsatur_num).min(smallest_last_num);

    if best_num <= 1 {
        return best_colors;
    }

    let mut colors = vec![usize::MAX; n];
    let mut saturation = vec![0usize; n];
    let mut neighbor_color_counts = vec![vec![0u16; best_num]; n];

    fn dfs(
        adj: &Vec<Vec<usize>>,
        degrees: &[usize],
        colors: &mut Vec<usize>,
        saturation: &mut Vec<usize>,
        neighbor_color_counts: &mut Vec<Vec<u16>>,
        num_used_colors: usize,
        colored_count: usize,
        best_num: &mut usize,
        best_colors: &mut Vec<usize>,
        visit_count: &mut u64,
    ) {
        *visit_count += 1;
        if *visit_count % 10_000 == 0 {
            let lb = num_used_colors;
            let ub = *best_num;
            eprintln!(
                "DSATUR exact: visits={} colored={}/{} lb={} ub={}",
                *visit_count,
                colored_count,
                colors.len(),
                lb,
                ub,
            );
        }

        if colored_count == colors.len() {
            if num_used_colors < *best_num {
                *best_num = num_used_colors;
                best_colors.clone_from(colors);
            }
            return;
        }

        if num_used_colors >= *best_num {
            return;
        }

        let Some(u) = select_vertex(colors, saturation, degrees) else { return; };

        for c in 0..num_used_colors {
            if neighbor_color_counts[u][c] > 0 {
                continue;
            }

            colors[u] = c;
            let mut changed_neighbors: Vec<usize> = Vec::new();
            for &v in &adj[u] {
                if colors[v] != usize::MAX {
                    continue;
                }
                if neighbor_color_counts[v][c] == 0 {
                    saturation[v] += 1;
                }
                neighbor_color_counts[v][c] = neighbor_color_counts[v][c].saturating_add(1);
                changed_neighbors.push(v);
            }

            dfs(
                adj,
                degrees,
                colors,
                saturation,
                neighbor_color_counts,
                num_used_colors,
                colored_count + 1,
                best_num,
                best_colors,
                visit_count,
            );

            for v in changed_neighbors {
                neighbor_color_counts[v][c] = neighbor_color_counts[v][c].saturating_sub(1);
                if neighbor_color_counts[v][c] == 0 {
                    saturation[v] -= 1;
                }
            }
            colors[u] = usize::MAX;
        }

        if num_used_colors + 1 < *best_num {
            let c = num_used_colors;
            colors[u] = c;
            let mut changed_neighbors: Vec<usize> = Vec::new();
            for &v in &adj[u] {
                if colors[v] != usize::MAX {
                    continue;
                }
                if neighbor_color_counts[v][c] == 0 {
                    saturation[v] += 1;
                }
                neighbor_color_counts[v][c] = neighbor_color_counts[v][c].saturating_add(1);
                changed_neighbors.push(v);
            }

            dfs(
                adj,
                degrees,
                colors,
                saturation,
                neighbor_color_counts,
                num_used_colors + 1,
                colored_count + 1,
                best_num,
                best_colors,
                visit_count,
            );

            for v in changed_neighbors {
                neighbor_color_counts[v][c] = neighbor_color_counts[v][c].saturating_sub(1);
                if neighbor_color_counts[v][c] == 0 {
                    saturation[v] -= 1;
                }
            }
            colors[u] = usize::MAX;
        }
    }

    let mut visit_count: u64 = 0;
    dfs(
        adj,
        &degrees,
        &mut colors,
        &mut saturation,
        &mut neighbor_color_counts,
        0,
        0,
        &mut best_num,
        &mut best_colors,
        &mut visit_count,
    );

    best_colors
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
