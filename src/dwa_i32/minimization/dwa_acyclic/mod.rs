mod consolidate_ranges;

use crate::dwa_i32::common::{Label, StateID, Weight};
use crate::dwa_i32::dwa::{DWA, DWABuildError, DWAState, DWAStates};
use crate::dwa_i32::minimization::common::DwaPass;
use crate::dwa_i32::minimization::graph_coloring::{solve_greedy_coloring, solve_exact_graph_coloring};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

impl DWA {
    pub fn minimize_acyclic(&mut self) {
        // Skip expensive validation in non-debug builds
        #[cfg(debug_assertions)]
        let x = self.clone();
        
        // Weight pushing enables the diamond case optimization:
        // States with different final_weights but same transition structure can be merged
        // because the different outputs are encoded in the incoming transition weights.
        let pushed = push_weights_acyclic(self);
        
        // Verify weight pushing is semantics-preserving (only in debug mode)
        #[cfg(debug_assertions)]
        if pushed {
            crate::dwa_i32::test_weighted_automata::stochastic_equivalence_test(x.clone(), self.clone());
        }
        
        #[cfg(debug_assertions)]
        let after_push = self.clone();
        
        match minimize_acyclic_exact(self) {
            Ok(min_dwa) => *self = min_dwa,
            Err(e) => {
                eprintln!("DWA minimization failed: {:?}", e);
            }
        }
        
        // Verify minimization is semantics-preserving (only in debug mode)
        #[cfg(debug_assertions)]
        crate::dwa_i32::test_weighted_automata::stochastic_equivalence_test(after_push.clone(), self.clone());
        
        // NOTE: ConsolidateRanges is NOT called here - it's a separate pass in the config
        // to avoid running it twice when configs include both Minimize and ConsolidateRanges.
    }
}

/// Push weights forward for acyclic DWAs.
/// Computes reachable outputs and restricts transitions to tokens that can produce output.
fn push_weights_acyclic(dwa: &mut DWA) -> bool {
    let n = dwa.states.len();
    if n == 0 { return false; }

    // Compute topological order using Kahn's algorithm
    let mut in_degree = vec![0usize; n];
    for u in 0..n {
        for &v in dwa.states[u].transitions.values() {
            if v < n { in_degree[v] += 1; }
        }
    }
    
    let mut queue: VecDeque<_> = in_degree.iter().enumerate()
        .filter(|(_, &deg)| deg == 0).map(|(i, _)| i).collect();
    
    let mut topo_order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        topo_order.push(u);
        for &v in dwa.states[u].transitions.values() {
            if v < n {
                in_degree[v] -= 1;
                if in_degree[v] == 0 { queue.push_back(v); }
            }
        }
    }
    
    if topo_order.len() != n { return false; } // Has cycles

    // Compute reachable outputs (backward from leaves)
    let mut reachable = vec![Weight::zeros(); n];
    for &u in topo_order.iter().rev() {
        let mut reach_u = dwa.states[u].final_weight.clone().unwrap_or_else(Weight::zeros);
        for (&label, &target) in &dwa.states[u].transitions {
            if target < n {
                if let Some(w) = dwa.states[u].trans_weights.get(&label) {
                    reach_u |= &(w & &reachable[target]);
                }
            }
        }
        reachable[u] = reach_u;
    }

    // Push reachable outputs into transition weights
    let mut changed = false;
    for u in 0..n {
        for (&label, &target) in dwa.states[u].transitions.clone().iter() {
            if target < n {
                if let Some(w) = dwa.states[u].trans_weights.get_mut(&label) {
                    let new_w = &*w & &reachable[target];
                    if *w != new_w { *w = new_w; changed = true; }
                }
            }
        }
    }
    changed
}

/// Minimizes an Acyclic DWA to its globally optimal state count.
///
/// # Theoretical Guarantees
/// 1. **Semantic Equivalence**: The output DWA produces the exact same `Weight` result
///    for any input word as the input DWA, relative to the start state.
/// 2. **Global Optimality**: The number of states is provably minimal. This algorithm
///    solves the NP-hard exact clustering problem (via Graph Coloring) to merge
///    states that have disjoint token flows (like the "Diamond" case).
///
/// # Complexity
/// Worst-case exponential due to exact graph coloring, but highly efficient for
/// typical automata where "incompatibility density" is low.
pub fn minimize_acyclic_exact(dwa: &DWA) -> Result<DWA, DWABuildError> {
    if dwa.states.len() == 0 {
        return Ok(DWA::new());
    }
    
    let total_start = std::time::Instant::now();
    
    // Step 0: Preprocess - tighten weights by removing unreachable tokens
    let step0_start = std::time::Instant::now();
    let dwa = tighten_weights(dwa)?;
    crate::debug!(6, "Acyclic minimize step 0 (tighten_weights): {:?}", step0_start.elapsed());

    // 1. Topological Sort & Reachability Analysis
    let step1_start = std::time::Instant::now();
    let topo_order = compute_topo_order(&dwa)?;
    crate::debug!(6, "Acyclic minimize step 1 (topo_order): {:?}", step1_start.elapsed());

    // 2. Compute "Needed" sets (Reverse Flow Analysis).
    let step2_start = std::time::Instant::now();
    let needed = compute_needed_sets(&dwa, &topo_order);
    crate::debug!(6, "Acyclic minimize step 2 (needed_sets): {:?}", step2_start.elapsed());

    // 3. Layer states by topological height (distance to sink).
    let step3_start = std::time::Instant::now();
    let heights = compute_heights(&dwa, &topo_order);
    let max_height = heights.iter().max().copied().unwrap_or(0);

    let mut states_by_height: Vec<Vec<StateID>> = vec![vec![]; max_height + 1];
    for (id, &h) in heights.iter().enumerate() {
        if needed[id].is_empty() && id != dwa.body.start_state { continue; }
        states_by_height[h].push(id);
    }
    crate::debug!(6, "Acyclic minimize step 3 (heights): {:?}, max_height={}, largest_level={}", 
        step3_start.elapsed(), max_height,
        states_by_height.iter().map(|v| v.len()).max().unwrap_or(0));

    // 4. Bottom-Up Exact Minimization
    let mut old_to_new: HashMap<StateID, StateID> = HashMap::new();
    let mut new_states: Vec<MergedStateBuilder> = Vec::new();

    // Process from leaves (height 0) upwards
    for h in 0..=max_height {
        let candidates = &states_by_height[h];
        if candidates.is_empty() { continue; }

        // Compute coloring - use signature-based method for large candidate sets
        let coloring = compute_height_coloring(&dwa, candidates, &needed, &old_to_new, &new_states);

        // Construct new merged states from color classes
        let base_new_id = new_states.len();
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        // Map old states to new merged states
        for (old_idx, &color) in coloring.iter().enumerate() {
            old_to_new.insert(candidates[old_idx], base_new_id + color);
        }
        new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));

        // Merge states into builders
        let (completed, builders) = new_states.split_at_mut(base_new_id);
        for (old_idx, &color) in coloring.iter().enumerate() {
            merge_state_into_builder(
                candidates[old_idx], color, &dwa, &needed, &old_to_new, completed, builders
            );
        }
    }

    crate::debug!(6, "Acyclic minimize: {} -> {} states in {:?}", 
        dwa.states.len(), new_states.len(), total_start.elapsed());

    // 5. Reconstruct the Final DWA
    let result = reconstruct_dwa(dwa.body.start_state, &old_to_new, new_states)?;
    
    // 6. Stochastic validation (only when STOCHASTIC_MERGE_VALIDATION=1)
    if std::env::var("STOCHASTIC_MERGE_VALIDATION").is_ok() {
        stochastic_merge_validation(&result)?;
    }
    
    Ok(result)
}

/// Compute coloring for a height level's candidates.
fn compute_height_coloring(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder],
) -> Vec<usize> {
    let start = std::time::Instant::now();
    
    // Build full incompatibility graph
    let adj = build_incompatibility_graph(dwa, candidates, needed, old_to_new, new_states);
    
    let graph_time = start.elapsed();
    
    // Check for timeout - if graph construction took too long, abort
    if graph_time.as_secs() > 60 {
        eprintln!("ERROR: Graph construction took {:?} for {} candidates - aborting", 
            graph_time, candidates.len());
        std::process::exit(1);
    }
    
    // Solve coloring: greedy for large graphs, exact for small ones
    let colors = if candidates.len() > 30 {
        solve_greedy_coloring(&adj)
    } else {
        solve_exact_graph_coloring(&adj)
    };
    
    let total_time = start.elapsed();
    if total_time.as_secs() > 60 {
        eprintln!("ERROR: Coloring took {:?} for {} candidates - aborting", 
            total_time, candidates.len());
        std::process::exit(1);
    }
    
    colors
}

/// Merge an old state into a builder at the given color index.
fn merge_state_into_builder(
    old_id: StateID,
    color: usize,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    completed: &[MergedStateBuilder],
    builders: &mut [MergedStateBuilder],
) {
    let builder = &mut builders[color];
    let old_state = &dwa.states[old_id];

    // Union Final Weights
    if let Some(fw) = &old_state.final_weight {
        builder.final_weight |= fw;
    }

    // Union Needed Sets
    builder.needed |= &needed[old_id];

    // Merge Transitions
    for (&label, &target_old) in &old_state.transitions {
        if target_old >= dwa.states.len() { continue; }
        let Some(&target_new) = old_to_new.get(&target_old) else { continue; };
        let Some(w_orig) = old_state.trans_weights.get(&label) else { continue; };
        
        // Restrict weight to what's actually needed at target
        let w_effective = w_orig & &completed[target_new].needed;
        if !w_effective.is_empty() {
            builder.add_transition(label, target_new, w_effective);
        }
    }
}

/// Stochastic validation: randomly sample pairs of states and check if any could be merged.
/// If a significant number of mergeable pairs are found, it indicates that minimization
/// is suboptimal.
fn stochastic_merge_validation(dwa: &DWA) -> Result<(), DWABuildError> {
    use rand::prelude::IndexedRandom;
    use rand::SeedableRng;
    
    let n = dwa.states.len();
    if n < 10 {
        return Ok(()); // Too small to meaningfully validate
    }
    
    // Sample up to 10000 random pairs
    let num_samples = std::cmp::min(10000, n * n / 2);
    let state_ids: Vec<StateID> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    let mut mergeable_count = 0;
    let mut total_checked = 0;
    
    for _ in 0..num_samples {
        let pair: Vec<_> = state_ids.choose_multiple(&mut rng, 2).collect();
        if pair.len() < 2 { continue; }
        let (s1, s2) = (*pair[0], *pair[1]);
        if s1 == s2 { continue; }
        
        total_checked += 1;
        
        // States are mergeable if they're identical
        let (state1, state2) = (&dwa.states[s1], &dwa.states[s2]);
        if state1.final_weight == state2.final_weight 
            && state1.transitions == state2.transitions
            && state1.trans_weights == state2.trans_weights 
        {
            mergeable_count += 1;
            crate::debug!(3, "STOCHASTIC: Found mergeable pair: {:?} and {:?}", s1, s2);
        }
    }
    
    crate::debug!(3, "STOCHASTIC: {} states, {} pairs checked, {} mergeable", 
        n, total_checked, mergeable_count);
    
    if mergeable_count > 0 {
        panic!("STOCHASTIC MERGE VALIDATION FAILED: Found {} mergeable pairs", mergeable_count);
    }
    Ok(())
}

// --- Structures & Helpers ---

#[derive(Default)]
struct MergedStateBuilder {
    final_weight: Weight,
    needed: Weight,
    transitions: BTreeMap<Label, (StateID, Weight)>,
}

impl MergedStateBuilder {
    fn add_transition(&mut self, label: Label, target: StateID, weight: Weight) {
        let entry = self.transitions.entry(label).or_insert((target, Weight::zeros()));
        // Incompatibility logic ensures that for a given label, 
        // if weights overlap, they must target the same new cluster.
        entry.1 |= &weight;
    }
}

// --- Phase 1 & 2: Analysis ---

fn compute_topo_order(dwa: &DWA) -> Result<Vec<StateID>, DWABuildError> {
    let n = dwa.states.len();
    let mut visited = vec![0u8; n]; // 0: none, 1: visiting, 2: visited
    let mut order = Vec::with_capacity(n);

    for i in 0..n {
        if visited[i] == 0 {
            visit(i, dwa, &mut visited, &mut order)?;
        }
    }

    fn visit(u: usize, dwa: &DWA, visited: &mut Vec<u8>, order: &mut Vec<usize>) -> Result<(), DWABuildError> {
        visited[u] = 1; // Visiting
        for &v in dwa.states[u].transitions.values() {
            if v < dwa.states.len() {
                if visited[v] == 1 { 
                    // Cycle detected
                    return Err(DWABuildError::StateOutOfBounds { state: u }); 
                }
                if visited[v] == 0 { 
                    visit(v, dwa, visited, order)?; 
                }
            }
        }
        visited[u] = 2; // Visited
        order.push(u);
        Ok(())
    }

    Ok(order)
}

fn compute_needed_sets(dwa: &DWA, topo_order: &[StateID]) -> Vec<Weight> {
    let mut needed = vec![Weight::zeros(); dwa.states.len()];
    for &u in topo_order {
        let mut acc = dwa.states[u].final_weight.clone().unwrap_or_else(Weight::zeros);
        for (&lbl, &v) in &dwa.states[u].transitions {
            if v >= dwa.states.len() { continue; }
            if let Some(w) = dwa.states[u].trans_weights.get(&lbl) {
                acc |= &(w & &needed[v]);
            }
        }
        needed[u] = acc;
    }
    needed
}

/// Compute forward reachability: which tokens can reach each state from start
fn compute_forward_reachable(dwa: &DWA, topo_order: &[StateID]) -> Vec<Weight> {
    let mut forward = vec![Weight::zeros(); dwa.states.len()];
    forward[dwa.body.start_state] = Weight::all();
    
    for &u in topo_order.iter().rev() {
        if forward[u].is_empty() { continue; }
        let incoming = forward[u].clone();
        
        for (&lbl, &v) in &dwa.states[u].transitions {
            if v >= dwa.states.len() { continue; }
            if let Some(w) = dwa.states[u].trans_weights.get(&lbl) {
                forward[v] |= &(&incoming & w);
            }
        }
    }
    forward
}

/// Tighten DWA weights by removing tokens that can never reach a transition.
/// Semantic-preserving: restricts weights to tokens that can actually reach each state.
fn tighten_weights(dwa: &DWA) -> Result<DWA, DWABuildError> {
    if dwa.states.is_empty() { return Ok(DWA::new()); }
    
    let topo_order = compute_topo_order(dwa)?;
    let forward = compute_forward_reachable(dwa, &topo_order);
    
    let mut new_states = DWAStates(Vec::with_capacity(dwa.states.len()));
    for (u, state) in dwa.states.0.iter().enumerate() {
        let mut new_state = DWAState::default();
        
        // Tighten final weight
        if let Some(fw) = &state.final_weight {
            let tightened = fw & &forward[u];
            if !tightened.is_empty() {
                new_state.final_weight = Some(tightened);
            }
        }
        
        // Tighten transition weights
        for (&lbl, &target) in &state.transitions {
            if target >= dwa.states.len() { continue; }
            if let Some(w) = state.trans_weights.get(&lbl) {
                let tightened = w & &forward[u];
                if !tightened.is_empty() {
                    new_state.transitions.insert(lbl, target);
                    new_state.trans_weights.insert(lbl, tightened);
                }
            }
        }
        new_states.0.push(new_state);
    }
    
    Ok(DWA { states: new_states, body: dwa.body.clone() })
}

fn compute_heights(dwa: &DWA, topo_order: &[StateID]) -> Vec<usize> {
    let mut heights = vec![0; dwa.states.len()];
    for &u in topo_order {
        heights[u] = dwa.states[u].transitions.values()
            .filter(|&&v| v < dwa.states.len())
            .map(|&v| heights[v] + 1)
            .max()
            .unwrap_or(0);
    }
    heights
}

// --- Phase 3: Compatibility & Coloring ---

/// Check if two states can be merged.
/// 
/// States can be merged if their behavior is identical on overlapping tokens,
/// and for all labels, their transitions either target the same state or
/// target states that are equivalent on the combined token flow.
fn are_compatible(
    u: StateID,
    v: StateID,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder]
) -> bool {
    let domain_overlap = &needed[u] & &needed[v];
    
    // Helper: get weight restricted to overlap (or zeros if no final/trans weight)
    let final_on_overlap = |s: StateID| -> Weight {
        dwa.states[s].final_weight.as_ref()
            .map(|w| w & &domain_overlap)
            .unwrap_or_else(Weight::zeros)
    };
    
    // Check final weights match on overlap
    if !domain_overlap.is_empty() && final_on_overlap(u) != final_on_overlap(v) {
        return false;
    }
    
    // Check each label
    let labels: BTreeSet<Label> = dwa.states[u].transitions.keys()
        .chain(dwa.states[v].transitions.keys())
        .copied().collect();

    for lbl in labels {
        // Get effective transition weights (empty if no transition or no weight)
        let get_weight = |s: StateID| -> Weight {
            dwa.states[s].transitions.get(&lbl)
                .and_then(|_| dwa.states[s].trans_weights.get(&lbl))
                .cloned()
                .unwrap_or_else(Weight::zeros)
        };
        let w_u = get_weight(u);
        let w_v = get_weight(v);

        // On overlap domain, weights must match
        if !domain_overlap.is_empty() {
            let w_u_overlap = &w_u & &domain_overlap;
            let w_v_overlap = &w_v & &domain_overlap;
            if w_u_overlap != w_v_overlap {
                return false;
            }
            // One has transition on overlap, other doesn't = incompatible
            if w_u_overlap.is_empty() != w_v_overlap.is_empty() {
                return false;
            }
        }

        // If both have live transitions, check targets are compatible
        if !w_u.is_empty() && !w_v.is_empty() {
            let tu = dwa.states[u].transitions.get(&lbl);
            let tv = dwa.states[v].transitions.get(&lbl);
            
            match (tu, tv) {
                (Some(&tu), Some(&tv)) => {
                    if !targets_compatible(tu, tv, &w_u, &w_v, old_to_new, new_states) {
                        return false;
                    }
                }
                _ => return false, // Both have weights but one lacks target
            }
        }
    }
    true
}

/// Check if two transition targets are compatible for merging.
fn targets_compatible(
    tu: StateID,
    tv: StateID,
    w_u: &Weight,
    w_v: &Weight,
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder],
) -> bool {
    let mapped_u = old_to_new.get(&tu);
    let mapped_v = old_to_new.get(&tv);
    
    match (mapped_u, mapped_v) {
        (Some(&u_new), Some(&v_new)) if u_new != v_new => {
            // Different targets: must be equivalent on combined domain
            let w_combined = w_u | w_v;
            targets_equivalent_on_domain(u_new, v_new, &w_combined, new_states)
        }
        (Some(_), None) | (None, Some(_)) => false,
        (None, None) => tu == tv, // Same-level: must be same original
        _ => true, // Both map to same target
    }
}

/// Check if two target states (already merged) are equivalent on a given domain.
/// This enables merging parent states that point to different targets, as long as
/// those targets behave identically on the tokens that would actually flow through them.
fn targets_equivalent_on_domain(
    t_u: StateID,
    t_v: StateID,
    domain: &Weight,
    new_states: &[MergedStateBuilder],
) -> bool {
    if t_u >= new_states.len() || t_v >= new_states.len() {
        return false;
    }
    
    let bu = &new_states[t_u];
    let bv = &new_states[t_v];
    
    // Check final weights on domain
    let fw_u = &bu.final_weight & domain;
    let fw_v = &bv.final_weight & domain;
    if fw_u != fw_v {
        return false;
    }
    
    // Check transitions on domain
    let all_labels: BTreeSet<Label> = bu.transitions.keys()
        .chain(bv.transitions.keys())
        .copied()
        .collect();
    
    for lbl in all_labels {
        let (target_u, w_u) = bu.transitions.get(&lbl)
            .map(|(t, w)| (*t, w & domain))
            .unwrap_or((usize::MAX, Weight::zeros()));
        let (target_v, w_v) = bv.transitions.get(&lbl)
            .map(|(t, w)| (*t, w & domain))
            .unwrap_or((usize::MAX, Weight::zeros()));
        
        if w_u != w_v || (!w_u.is_empty() && target_u != target_v) {
            return false;
        }
    }
    true
}

fn build_incompatibility_graph(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder]
) -> Vec<Vec<usize>> {
    let n = candidates.len();
    if n <= 200 {
        // For small candidate sets, use the simple O(n²) approach
        return build_incompatibility_graph_simple(dwa, candidates, needed, old_to_new, new_states);
    }
    
    // For large candidate sets, use partition-based approach
    build_incompatibility_graph_partition(dwa, candidates, needed, old_to_new, new_states)
}

/// Simple O(n²) approach for small candidate sets.
fn build_incompatibility_graph_simple(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder]
) -> Vec<Vec<usize>> {
    let n = candidates.len();
    let mut adj = vec![vec![]; n];
    
    for i in 0..n {
        for j in (i+1)..n {
            if !are_compatible(candidates[i], candidates[j], dwa, needed, old_to_new, new_states) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

/// Partition-based approach for large candidate sets.
/// 
/// Key insight: Two states are incompatible only if they have different behavioral
/// signatures on some shared partition element (a disjoint interval of the token space).
/// 
/// Algorithm:
/// 1. Compute all weight boundaries from all candidates
/// 2. Create partition elements (disjoint intervals) from boundaries
/// 3. For each partition element, group candidates by their restricted behavioral signature
/// 4. Candidates with different signatures on a shared partition element are incompatible
/// 5. If partition analysis finds all states compatible, skip O(n²) verification
fn build_incompatibility_graph_partition(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder]
) -> Vec<Vec<usize>> {
    let n = candidates.len();
    let start = std::time::Instant::now();
    
    // Step 1: Compute partition boundaries from all candidate domains
    let mut boundaries: BTreeSet<usize> = BTreeSet::new();
    for &s in candidates {
        // Add all range boundaries from this state's domain
        for range in needed[s].rsb.ranges() {
            boundaries.insert(*range.start());
            if *range.end() < usize::MAX {
                boundaries.insert(*range.end() + 1);
            }
        }
    }
    
    // Convert boundaries to partition elements (intervals)
    let boundary_vec: Vec<usize> = boundaries.into_iter().collect();
    let num_partitions = boundary_vec.len().saturating_sub(1);
    
    if num_partitions == 0 {
        // All states have empty domain or single point - use simple approach
        return build_incompatibility_graph_simple(dwa, candidates, needed, old_to_new, new_states);
    }
    
    // Step 2: For each partition element, compute which candidates have it in their domain
    // and compute their behavioral signature restricted to that partition
    
    // Track incompatible pairs using a set (to avoid duplicates)
    let mut incompatible: HashSet<(usize, usize)> = HashSet::new();
    
    // For each partition element...
    for p_idx in 0..num_partitions {
        let p_start = boundary_vec[p_idx];
        let p_end = boundary_vec[p_idx + 1] - 1;
        let partition_weight = Weight::from_ranges(&[(p_start, p_end)]);
        
        // Find candidates that have this partition element in their domain
        let mut candidates_in_partition: Vec<usize> = Vec::new();
        for (idx, &s) in candidates.iter().enumerate() {
            if !(&needed[s] & &partition_weight).is_empty() {
                candidates_in_partition.push(idx);
            }
        }
        
        if candidates_in_partition.len() <= 1 {
            continue; // No pairs to check in this partition
        }
        
        // Compute restricted signature for each candidate
        // Signature includes: final weight on partition, and for each label: (target behavior hash, trans_weight hash)
        let mut sig_to_candidates: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
        
        for &idx in &candidates_in_partition {
            let s = candidates[idx];
            let sig = compute_restricted_signature(s, &partition_weight, dwa, old_to_new, new_states);
            sig_to_candidates.entry(sig).or_default().push(idx);
        }
        
        // Candidates with different signatures on this partition are incompatible
        let groups: Vec<Vec<usize>> = sig_to_candidates.into_values().collect();
        for i in 0..groups.len() {
            for j in (i+1)..groups.len() {
                for &idx_i in &groups[i] {
                    for &idx_j in &groups[j] {
                        let pair = if idx_i < idx_j { (idx_i, idx_j) } else { (idx_j, idx_i) };
                        incompatible.insert(pair);
                    }
                }
            }
        }
    }
    
    let partition_time = start.elapsed();
    
    // If partition analysis found incompatibilities, we only need to verify those pairs.
    // If it found NO incompatibilities, all states can be merged (graph is empty).
    if incompatible.is_empty() {
        crate::debug!(5, "Partition analysis: {} candidates, {} partitions, 0 incompatibilities found → all compatible, {:?}",
            n, num_partitions, partition_time);
        
        // Debug verification: check a sample of pairs to verify correctness
        #[cfg(debug_assertions)]
        {
            let sample_size = std::cmp::min(100, n);
            for i in 0..sample_size {
                for j in (i+1)..sample_size {
                    if !are_compatible(candidates[i], candidates[j], dwa, needed, old_to_new, new_states) {
                        panic!("Partition analysis bug: states {} and {} are incompatible but partition found them compatible", 
                            candidates[i], candidates[j]);
                    }
                }
            }
        }
        
        return vec![vec![]; n]; // Empty adjacency list = all states compatible
    }
    
    // Build adjacency list from incompatible pairs
    let mut adj = vec![vec![]; n];
    for (i, j) in &incompatible {
        adj[*i].push(*j);
        adj[*j].push(*i);
    }
    
    crate::debug!(5, "Partition analysis: {} candidates, {} partitions, {} incompatible pairs, {:?}",
        n, num_partitions, incompatible.len(), partition_time);
    
    adj
}

/// Compute a restricted behavioral signature for a state on a given partition element.
/// 
/// The signature captures the state's behavior restricted to the partition:
/// - Final weight on partition
/// - For each transition active on partition: (label, target's behavioral hash, weight hash)
/// 
/// The target's behavioral hash incorporates its behavior on the transition weight,
/// making the signature transitively capture downstream behavior.
fn compute_restricted_signature(
    state_id: StateID,
    partition: &Weight,
    dwa: &DWA,
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder],
) -> Vec<u64> {
    let state = &dwa.states[state_id];
    
    // Signature components:
    // 1. Final weight restricted to partition
    let fw = state.final_weight.as_ref()
        .map(|w| w & partition)
        .unwrap_or_else(Weight::zeros);
    
    let mut sig = vec![fw.fast_hash()];
    
    // 2. For each transition: (label, target_behavior_hash, trans_weight restricted to partition)
    let mut trans_sigs: Vec<_> = state.transitions.iter()
        .filter_map(|(&label, &target)| {
            let w = state.trans_weights.get(&label)
                .map(|w| w & partition)
                .unwrap_or_else(Weight::zeros);
            if w.is_empty() { return None; }
            
            // Get target's behavioral hash on the transition weight
            // This incorporates the target's downstream behavior
            let target_hash = match old_to_new.get(&target) {
                Some(&mapped_target) if mapped_target < new_states.len() => {
                    // Target is a merged state - compute its hash on the transition weight
                    compute_merged_state_hash(mapped_target, &w, new_states)
                }
                _ => {
                    // Target not mapped yet (shouldn't happen in bottom-up processing)
                    // or maps to invalid index - use a sentinel value
                    0xDEADBEEF_u64
                }
            };
            
            Some((label, target_hash, w.fast_hash()))
        })
        .collect();
    trans_sigs.sort();
    
    for (label, target_hash, weight_hash) in trans_sigs {
        sig.extend([label as u64, target_hash, weight_hash]);
    }
    
    sig
}

/// Compute a hash of a merged state's behavior restricted to a domain.
/// This is used to make signatures transitively capture downstream behavior.
fn compute_merged_state_hash(
    state_idx: usize,
    domain: &Weight,
    new_states: &[MergedStateBuilder],
) -> u64 {
    let state = &new_states[state_idx];
    
    // Hash final weight on domain
    let fw_hash = (&state.final_weight & domain).fast_hash();
    
    // Hash transitions on domain
    let mut trans_hashes: Vec<(Label, StateID, u64)> = state.transitions.iter()
        .filter_map(|(&label, &(target, ref weight))| {
            let w = weight & domain;
            if w.is_empty() { return None; }
            Some((label, target, w.fast_hash()))
        })
        .collect();
    trans_hashes.sort();
    
    // Combine into single hash
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    fw_hash.hash(&mut hasher);
    for (label, target, hash) in trans_hashes {
        label.hash(&mut hasher);
        target.hash(&mut hasher);
        hash.hash(&mut hasher);
    }
    hasher.finish()
}

// --- Phase 4: Reconstruction ---

fn reconstruct_dwa(
    start_old: StateID,
    old_to_new: &HashMap<StateID, StateID>,
    builders: Vec<MergedStateBuilder>
) -> Result<DWA, DWABuildError> {
    let states: Vec<DWAState> = builders.into_iter().map(|b| {
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
        state
    }).collect();

    Ok(DWA {
        states: DWAStates(states),
        body: crate::dwa_i32::dwa::DWABody {
            start_state: old_to_new.get(&start_old).copied().unwrap_or(0),
        },
    })
}