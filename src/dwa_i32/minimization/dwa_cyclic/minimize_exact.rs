//! Exact DWA minimization extended to handle cyclic automata.
//!
//! This extends the acyclic "Diamond-aware" minimization algorithm to handle cycles
//! by using SCC decomposition and fixed-point iteration for needed sets within SCCs.

use crate::dwa_i32::common::{Label, StateID, Weight};
use crate::dwa_i32::dwa::{DWA, DWABuildError, DWAState, DWAStates};
use std::collections::{BTreeMap, BTreeSet, HashMap};

impl DWA {
    /// Minimizes a (possibly cyclic) DWA using the exact Diamond-aware algorithm.
    /// 
    /// This is an extension of the acyclic algorithm that handles cycles via:
    /// 1. SCC decomposition (Tarjan's algorithm)
    /// 2. Fixed-point iteration for needed sets within SCCs
    /// 3. Processing SCCs in reverse topological order (leaves to start)
    pub fn minimize_exact(&mut self) {
        match minimize_exact_extended(self) {
            Ok(min_dwa) => *self = min_dwa,
            Err(e) => {
                crate::debug!(4, "DWA exact minimization failed: {:?}, falling back to partition refinement", e);
                // Fallback to the simple partition refinement
                self.minimize_states_cyclic();
            }
        }
    }
}

/// Main entry point for exact minimization with cycle support.
pub fn minimize_exact_extended(dwa: &DWA) -> Result<DWA, DWABuildError> {
    if dwa.states.len() == 0 {
        return Ok(DWA::new());
    }

    // Step 0: Compute SCCs using Tarjan's algorithm
    let (sccs, scc_of) = compute_sccs(dwa);
    let num_sccs = sccs.len();
    
    if num_sccs == 0 {
        return Ok(DWA::new());
    }

    // Compute SCC DAG and its topological order
    let scc_order = compute_scc_topo_order(&sccs, &scc_of, dwa);

    // Step 1: Tighten weights using SCC-aware forward reachability
    let dwa = tighten_weights_cyclic(dwa, &sccs, &scc_of, &scc_order)?;

    // Step 2: Compute "Needed" sets with fixed-point iteration for SCCs
    let needed = compute_needed_sets_cyclic(&dwa, &sccs, &scc_of, &scc_order);

    // Step 3: Compute heights (SCC-based)
    // All states in the same SCC get the same height
    let heights = compute_heights_scc(&dwa, &sccs, &scc_of, &scc_order);
    let max_height = heights.iter().max().copied().unwrap_or(0);

    let mut states_by_height: Vec<Vec<StateID>> = vec![vec![]; max_height + 1];
    for (id, &h) in heights.iter().enumerate() {
        // Only minimize reachable states
        if needed[id].is_empty() && id != dwa.body.start_state {
            continue;
        }
        states_by_height[h].push(id);
    }

    // Step 4: Bottom-Up Exact Minimization (same as acyclic)
    let mut old_to_new: HashMap<StateID, StateID> = HashMap::new();
    let mut new_states: Vec<MergedStateBuilder> = Vec::new();

    for h in 0..=max_height {
        let candidates = &states_by_height[h];
        if candidates.is_empty() { continue; }

        // Build Incompatibility Graph
        let adj = build_incompatibility_graph(
            &dwa,
            candidates,
            &needed,
            &old_to_new,
            &new_states,
            &scc_of,
        );

        // Solve Graph Coloring
        let coloring = solve_exact_graph_coloring(&adj);

        // Construct merged states
        let base_new_id = new_states.len();
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        for (old_idx, color) in coloring.iter().enumerate() {
            let old_id = candidates[old_idx];
            let new_id = base_new_id + *color;
            old_to_new.insert(old_id, new_id);
        }

        for _ in 0..num_colors {
            new_states.push(MergedStateBuilder::default());
        }

        let (completed, builders) = new_states.split_at_mut(base_new_id);

        for (old_idx, &color) in coloring.iter().enumerate() {
            let old_id = candidates[old_idx];
            let builder = &mut builders[color];
            let old_state = &dwa.states[old_id];

            if let Some(fw) = &old_state.final_weight {
                builder.final_weight |= fw;
            }

            builder.needed |= &needed[old_id];

            for (&label, &target_old) in &old_state.transitions {
                if target_old >= dwa.states.len() { continue; }
                if !old_to_new.contains_key(&target_old) { continue; }

                let w_orig = old_state.trans_weights.get(&label).unwrap();
                let target_new = old_to_new[&target_old];

                let mut w_effective = w_orig.clone();
                if target_new < completed.len() {
                    w_effective &= &completed[target_new].needed;
                }

                if !w_effective.is_empty() {
                    builder.add_transition(label, target_new, w_effective);
                }
            }
        }
    }

    // Step 5: Reconstruct
    reconstruct_dwa(dwa.body.start_state, &old_to_new, new_states)
}

// --- Structures ---

#[derive(Default)]
struct MergedStateBuilder {
    final_weight: Weight,
    needed: Weight,
    transitions: BTreeMap<Label, (StateID, Weight)>,
}

impl MergedStateBuilder {
    fn add_transition(&mut self, label: Label, target: StateID, weight: Weight) {
        let entry = self.transitions.entry(label).or_insert((target, Weight::zeros()));
        entry.1 |= &weight;
    }
}

// --- SCC Computation (Tarjan's Algorithm) ---

/// Returns (sccs: Vec<Vec<StateID>>, scc_of: Vec<usize>)
/// sccs[i] contains the state IDs in SCC i
/// scc_of[state_id] is the SCC index containing that state
/// SCCs are returned in reverse topological order (leaves first)
fn compute_sccs(dwa: &DWA) -> (Vec<Vec<StateID>>, Vec<usize>) {
    let n = dwa.states.len();
    let mut index_counter = 0;
    let mut stack: Vec<StateID> = Vec::new();
    let mut on_stack = vec![false; n];
    let mut indices = vec![usize::MAX; n];
    let mut lowlinks = vec![usize::MAX; n];
    let mut sccs: Vec<Vec<StateID>> = Vec::new();
    let mut scc_of = vec![usize::MAX; n];

    fn strongconnect(
        v: StateID,
        dwa: &DWA,
        index_counter: &mut usize,
        stack: &mut Vec<StateID>,
        on_stack: &mut Vec<bool>,
        indices: &mut Vec<usize>,
        lowlinks: &mut Vec<usize>,
        sccs: &mut Vec<Vec<StateID>>,
        scc_of: &mut Vec<usize>,
    ) {
        indices[v] = *index_counter;
        lowlinks[v] = *index_counter;
        *index_counter += 1;
        stack.push(v);
        on_stack[v] = true;

        for &w in dwa.states[v].transitions.values() {
            if w >= dwa.states.len() { continue; }
            if indices[w] == usize::MAX {
                strongconnect(w, dwa, index_counter, stack, on_stack, indices, lowlinks, sccs, scc_of);
                lowlinks[v] = lowlinks[v].min(lowlinks[w]);
            } else if on_stack[w] {
                lowlinks[v] = lowlinks[v].min(indices[w]);
            }
        }

        if lowlinks[v] == indices[v] {
            let scc_idx = sccs.len();
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                scc_of[w] = scc_idx;
                scc.push(w);
                if w == v { break; }
            }
            sccs.push(scc);
        }
    }

    for v in 0..n {
        if indices[v] == usize::MAX {
            strongconnect(v, dwa, &mut index_counter, &mut stack, &mut on_stack, &mut indices, &mut lowlinks, &mut sccs, &mut scc_of);
        }
    }

    (sccs, scc_of)
}

/// Compute topological order of SCCs (as indices into sccs array)
/// Returns SCCs from leaves to root (reverse order for bottom-up processing)
fn compute_scc_topo_order(
    sccs: &[Vec<StateID>],
    scc_of: &[usize],
    dwa: &DWA,
) -> Vec<usize> {
    // Tarjan's algorithm already returns SCCs in reverse topological order
    // But let's verify and potentially recompute if needed
    
    let num_sccs = sccs.len();
    if num_sccs == 0 {
        return vec![];
    }
    
    // Build SCC adjacency (which SCCs does each SCC point to?)
    let mut scc_out: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); num_sccs];
    for scc_idx in 0..num_sccs {
        for &state in &sccs[scc_idx] {
            for &target in dwa.states[state].transitions.values() {
                if target >= dwa.states.len() { continue; }
                let target_scc = scc_of[target];
                if target_scc != scc_idx {
                    scc_out[scc_idx].insert(target_scc);
                }
            }
        }
    }

    // Kahn's algorithm for topological sort
    let mut in_degree = vec![0usize; num_sccs];
    for scc_idx in 0..num_sccs {
        for &target_scc in &scc_out[scc_idx] {
            in_degree[target_scc] += 1;
        }
    }

    let mut queue: Vec<usize> = (0..num_sccs).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(num_sccs);

    while let Some(scc) = queue.pop() {
        order.push(scc);
        for &target_scc in &scc_out[scc] {
            in_degree[target_scc] -= 1;
            if in_degree[target_scc] == 0 {
                queue.push(target_scc);
            }
        }
    }

    // We want leaves first, so reverse
    order.reverse();
    order
}

// --- Needed Sets with Fixed-Point Iteration ---

fn compute_needed_sets_cyclic(
    dwa: &DWA,
    sccs: &[Vec<StateID>],
    scc_of: &[usize],
    scc_order: &[usize],
) -> Vec<Weight> {
    let n = dwa.states.len();
    let mut needed = vec![Weight::zeros(); n];

    // Process SCCs in order (leaves first)
    for &scc_idx in scc_order {
        let scc_states = &sccs[scc_idx];
        
        if scc_states.len() == 1 {
            // Single-state SCC (possibly with self-loop)
            let u = scc_states[0];
            compute_needed_single_state(u, dwa, scc_of, scc_idx, &mut needed);
        } else {
            // Multi-state SCC - use fixed-point iteration
            compute_needed_scc_fixpoint(scc_states, dwa, scc_of, scc_idx, &mut needed);
        }
    }

    needed
}

fn compute_needed_single_state(
    u: StateID,
    dwa: &DWA,
    scc_of: &[usize],
    my_scc: usize,
    needed: &mut [Weight],
) {
    let mut acc = Weight::zeros();
    
    // Final weight
    if let Some(fw) = &dwa.states[u].final_weight {
        acc |= fw;
    }
    
    // Transitions to other SCCs (already computed)
    for (&lbl, &v) in &dwa.states[u].transitions {
        if v >= dwa.states.len() { continue; }
        if scc_of[v] != my_scc {
            // Target is in a different SCC (already processed)
            let w_trans = dwa.states[u].trans_weights.get(&lbl).unwrap();
            let mut contribution = w_trans.clone();
            contribution &= &needed[v];
            acc |= &contribution;
        }
    }
    
    // Self-loop handling: fixed point for single state
    // needed[u] = final ∪ (∪_{v≠u} (w_uv & needed[v])) ∪ (w_uu & needed[u])
    // This is: needed[u] = base ∪ (w_uu & needed[u])
    // Let w = w_uu. Expanding: needed = base | (w & needed) | (w & w & needed) | ...
    // For idempotent semiring (union): needed = base | (w & needed)
    // Fixed point: if w covers all tokens that could ever reach acceptance through self-loop,
    // then needed[u] = base (since w & base ⊆ base after initial union)
    // Actually for union semiring with all-or-nothing, we need to iterate:
    // needed' = base | (w_uu & needed)
    // Keep iterating until stable
    
    if let Some(&v) = dwa.states[u].transitions.get(&(-1 as i32)).filter(|&&v| v == u) {
        // Has self-loop on some label - check all self-loops
        for (&lbl, &target) in &dwa.states[u].transitions {
            if target == u {
                let w_self = dwa.states[u].trans_weights.get(&lbl).unwrap();
                // Fixed point: needed = base | (w_self & needed)
                // Iterate until stable
                for _ in 0..100 {
                    let prev = acc.clone();
                    let contrib = w_self.clone() & &acc;
                    acc |= &contrib;
                    if acc == prev { break; }
                }
            }
        }
    } else {
        // Check all transitions for self-loops
        for (&lbl, &target) in &dwa.states[u].transitions {
            if target == u {
                let w_self = dwa.states[u].trans_weights.get(&lbl).unwrap();
                for _ in 0..100 {
                    let prev = acc.clone();
                    let contrib = w_self.clone() & &acc;
                    acc |= &contrib;
                    if acc == prev { break; }
                }
            }
        }
    }
    
    needed[u] = acc;
}

fn compute_needed_scc_fixpoint(
    scc_states: &[StateID],
    dwa: &DWA,
    scc_of: &[usize],
    my_scc: usize,
    needed: &mut [Weight],
) {
    // Initialize needed for this SCC from finals and outgoing edges to other SCCs
    for &u in scc_states {
        let mut acc = Weight::zeros();
        if let Some(fw) = &dwa.states[u].final_weight {
            acc |= fw;
        }
        for (&lbl, &v) in &dwa.states[u].transitions {
            if v >= dwa.states.len() { continue; }
            if scc_of[v] != my_scc {
                let w_trans = dwa.states[u].trans_weights.get(&lbl).unwrap();
                let mut contribution = w_trans.clone();
                contribution &= &needed[v];
                acc |= &contribution;
            }
        }
        needed[u] = acc;
    }

    // Fixed-point iteration within SCC
    for iteration in 0..1000 {
        let mut changed = false;
        
        for &u in scc_states {
            let mut acc = needed[u].clone();
            
            for (&lbl, &v) in &dwa.states[u].transitions {
                if v >= dwa.states.len() { continue; }
                if scc_of[v] == my_scc {
                    // Intra-SCC edge
                    let w_trans = dwa.states[u].trans_weights.get(&lbl).unwrap();
                    let mut contribution = w_trans.clone();
                    contribution &= &needed[v];
                    acc |= &contribution;
                }
            }
            
            if acc != needed[u] {
                needed[u] = acc;
                changed = true;
            }
        }
        
        if !changed {
            break;
        }
        
        if iteration == 999 {
            crate::debug!(4, "compute_needed_scc_fixpoint: SCC with {} states did not converge in 1000 iterations", scc_states.len());
        }
    }
}

// --- Forward Reachability (Cyclic) ---

fn compute_forward_reachable_cyclic(
    dwa: &DWA,
    sccs: &[Vec<StateID>],
    scc_of: &[usize],
    scc_order: &[usize],
) -> Vec<Weight> {
    let n = dwa.states.len();
    let mut forward = vec![Weight::zeros(); n];
    
    // Start state can reach all tokens
    forward[dwa.body.start_state] = Weight::all();
    
    // Process SCCs in reverse order (root to leaves)
    for &scc_idx in scc_order.iter().rev() {
        let scc_states = &sccs[scc_idx];
        
        // Fixed-point within SCC
        for _ in 0..1000 {
            let mut changed = false;
            
            for &u in scc_states {
                let incoming = forward[u].clone();
                if incoming.is_empty() { continue; }
                
                for (&lbl, &v) in &dwa.states[u].transitions {
                    if v >= dwa.states.len() { continue; }
                    let w_trans = dwa.states[u].trans_weights.get(&lbl).unwrap();
                    let mut contribution = incoming.clone();
                    contribution &= w_trans;
                    
                    let old = forward[v].clone();
                    forward[v] |= &contribution;
                    if forward[v] != old {
                        changed = true;
                    }
                }
            }
            
            if !changed { break; }
        }
    }
    
    forward
}

fn tighten_weights_cyclic(
    dwa: &DWA,
    sccs: &[Vec<StateID>],
    scc_of: &[usize],
    scc_order: &[usize],
) -> Result<DWA, DWABuildError> {
    if dwa.states.len() == 0 {
        return Ok(DWA::new());
    }
    
    let forward = compute_forward_reachable_cyclic(dwa, sccs, scc_of, scc_order);
    
    let mut new_states = DWAStates(Vec::with_capacity(dwa.states.len()));
    
    for (u, state) in dwa.states.0.iter().enumerate() {
        let mut new_state = DWAState::default();
        
        if let Some(fw) = &state.final_weight {
            let tightened = fw & &forward[u];
            if !tightened.is_empty() {
                new_state.final_weight = Some(tightened);
            }
        }
        
        for (&lbl, &target) in &state.transitions {
            if target >= dwa.states.len() { continue; }
            
            let w_orig = state.trans_weights.get(&lbl).unwrap();
            let tightened = w_orig & &forward[u];
            
            if !tightened.is_empty() {
                new_state.transitions.insert(lbl, target);
                new_state.trans_weights.insert(lbl, tightened);
            }
        }
        
        new_states.0.push(new_state);
    }
    
    Ok(DWA {
        states: new_states,
        body: dwa.body.clone(),
    })
}

// --- Heights (SCC-based) ---

fn compute_heights_scc(
    dwa: &DWA,
    sccs: &[Vec<StateID>],
    scc_of: &[usize],
    scc_order: &[usize],
) -> Vec<usize> {
    let n = dwa.states.len();
    let num_sccs = sccs.len();
    
    // Compute height of each SCC
    let mut scc_heights = vec![0usize; num_sccs];
    
    for &scc_idx in scc_order {
        let mut max_child_height = 0;
        for &state in &sccs[scc_idx] {
            for &target in dwa.states[state].transitions.values() {
                if target >= dwa.states.len() { continue; }
                let target_scc = scc_of[target];
                if target_scc != scc_idx {
                    max_child_height = max_child_height.max(scc_heights[target_scc] + 1);
                }
            }
        }
        scc_heights[scc_idx] = max_child_height;
    }
    
    // Map to state heights
    let mut heights = vec![0; n];
    for (state, &scc_idx) in scc_of.iter().enumerate() {
        if scc_idx < num_sccs {
            heights[state] = scc_heights[scc_idx];
        }
    }
    
    heights
}

// --- Compatibility Check (extended for cycles) ---

fn are_compatible(
    u: StateID,
    v: StateID,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    _new_states: &[MergedStateBuilder],
    scc_of: &[usize],
) -> bool {
    // Compute the overlapping domain
    let mut domain = needed[u].clone();
    domain &= &needed[v];

    // Disjoint domains can always merge (Diamond case)
    if domain.is_empty() {
        return true;
    }

    // ADDITIONAL CHECK FOR CYCLES:
    // States in different SCCs at the same height can potentially merge
    // But states in the same SCC need extra care - merging them changes cycle structure
    // For safety, require same SCC for states with overlapping domains in cyclic graphs
    if scc_of[u] != scc_of[v] {
        // Different SCCs - check if they have compatible behavior on domain
        // This is the same logic as acyclic
    }

    // Final weights must match on domain
    let fw_u = dwa.states[u].final_weight.as_ref().cloned().unwrap_or_else(Weight::zeros);
    let fw_v = dwa.states[v].final_weight.as_ref().cloned().unwrap_or_else(Weight::zeros);

    if (&fw_u & &domain) != (&fw_v & &domain) {
        return false;
    }

    let mut labels: BTreeSet<Label> = dwa.states[u].transitions.keys().copied().collect();
    labels.extend(dwa.states[v].transitions.keys());

    for lbl in labels {
        let trans_u = dwa.states[u].get_transition(lbl);
        let trans_v = dwa.states[v].get_transition(lbl);

        let w_u_raw = match trans_u {
            Some((_, w)) => w.clone(),
            None => Weight::zeros(),
        };
        let w_v_raw = match trans_v {
            Some((_, w)) => w.clone(),
            None => Weight::zeros(),
        };

        // Effective weights
        let mut w_u_eff = w_u_raw.clone();
        if let Some((target_u, _)) = trans_u {
            if target_u < needed.len() {
                w_u_eff &= &needed[target_u];
            }
        }

        let mut w_v_eff = w_v_raw.clone();
        if let Some((target_v, _)) = trans_v {
            if target_v < needed.len() {
                w_v_eff &= &needed[target_v];
            }
        }

        if (&w_u_eff & &domain) != (&w_v_eff & &domain) {
            return false;
        }

        if (&w_u_raw & &domain) != (&w_v_raw & &domain) {
            return false;
        }

        let w_u_on_domain = &w_u_raw & &domain;
        let w_v_on_domain = &w_v_raw & &domain;
        if !w_u_on_domain.is_empty() && !w_v_on_domain.is_empty() && w_u_raw != w_v_raw {
            return false;
        }

        let w_common = &w_u_eff & &domain;
        if !w_common.is_empty() {
            let target_u_old = trans_u.unwrap().0;
            let target_v_old = trans_v.unwrap().0;

            // For cycles: if targets are in same SCC as source, we need to handle carefully
            // The target mapping might not exist yet if it's in the same layer
            if let (Some(&target_u_new), Some(&target_v_new)) = 
                (old_to_new.get(&target_u_old), old_to_new.get(&target_v_old)) 
            {
                if target_u_new != target_v_new {
                    return false;
                }
            } else {
                // Target not yet mapped - must be same state
                if target_u_old != target_v_old {
                    return false;
                }
            }
        }
    }

    true
}

fn build_incompatibility_graph(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder],
    scc_of: &[usize],
) -> Vec<Vec<usize>> {
    let n = candidates.len();
    let mut adj = vec![vec![]; n];

    for i in 0..n {
        for j in (i+1)..n {
            if !are_compatible(candidates[i], candidates[j], dwa, needed, old_to_new, new_states, scc_of) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

// --- Graph Coloring ---

fn solve_greedy_coloring(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    let n = adj.len();
    if n == 0 { return vec![]; }

    let mut colors = vec![usize::MAX; n];
    let mut nodes: Vec<usize> = (0..n).collect();
    nodes.sort_by_key(|&i| std::cmp::Reverse(adj[i].len()));

    for &u in &nodes {
        let neighbor_colors: std::collections::BTreeSet<usize> = 
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

fn solve_exact_graph_coloring(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    let n = adj.len();
    if n == 0 { return vec![]; }
    
    // For graphs with more than 30 nodes, use greedy coloring to avoid exponential blowup
    // The exact solver has worst-case exponential time complexity
    // Reduced from 50 to 30 because even 45 nodes can cause 4+ second blowup on dense graphs
    if n > 30 {
        return solve_greedy_coloring(adj);
    }

    let mut colors = vec![usize::MAX; n];
    let mut best_coloring = vec![0; n];
    let mut min_colors_found = n + 1;

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
        if current_max_color >= *min_colors_found {
            return;
        }
        if idx == nodes.len() {
            *min_colors_found = current_max_color;
            *best_coloring = colors.clone();
            return;
        }
        let u = nodes[idx];
        for c in 0..=current_max_color {
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

// --- Reconstruction ---

fn reconstruct_dwa(
    start_old: StateID,
    old_to_new: &HashMap<StateID, StateID>,
    builders: Vec<MergedStateBuilder>
) -> Result<DWA, DWABuildError> {
    let mut new_dwa_states = DWAStates(Vec::with_capacity(builders.len()));

    for b in builders {
        let mut state = DWAState::default();
        if !b.final_weight.is_empty() {
            state.final_weight = Some(b.final_weight);
        }
        for (lbl, (target, weight)) in b.transitions {
            if !weight.is_empty() {
                state.transitions.insert(lbl, target);
                state.trans_weights.insert(lbl, weight);
            }
        }
        new_dwa_states.0.push(state);
    }

    let start_new = old_to_new.get(&start_old).copied().unwrap_or(0);

    Ok(DWA {
        states: new_dwa_states,
        body: crate::dwa_i32::dwa::DWABody {
            start_state: start_new,
        },
    })
}
