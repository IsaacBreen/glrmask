mod consolidate_ranges;

use crate::dwa_i32::common::{Label, StateID, Weight};
use crate::dwa_i32::dwa::{DWA, DWABuildError, DWAState, DWAStates};
use crate::dwa_i32::minimization::common::DwaPass;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

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
/// 
/// This computes the "reachable outputs" for each state (union of all outputs
/// reachable from that state) and pushes this information into transition weights.
/// 
/// After this transformation:
/// - Transition weights represent all possible outputs reachable via that transition
/// - States with different final_weights but same "transition signature" can be merged
///   (the diamond case optimization)
fn push_weights_acyclic(dwa: &mut DWA) -> bool {
    let n = dwa.states.len();
    if n == 0 { return false; }

    // 1. Compute topological order using Kahn's algorithm
    let mut in_degree = vec![0usize; n];
    for u in 0..n {
        for &v in dwa.states[u].transitions.values() {
            if v < n {
                in_degree[v] += 1;
            }
        }
    }
    
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }
    
    let mut topo_order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        topo_order.push(u);
        for &v in dwa.states[u].transitions.values() {
            if v < n {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
    }
    
    if topo_order.len() != n {
        // Has cycles, cannot process as acyclic
        return false;
    }

    // 2. Compute "reachable outputs" for each state (backward from leaves)
    // reachable[s] = union of all outputs that can be produced starting from state s
    let mut reachable = vec![Weight::zeros(); n];
    
    // Process in reverse topological order (leaves first)
    for &u in topo_order.iter().rev() {
        let mut reach_u = Weight::zeros();
        
        // Include final weight
        if let Some(fw) = &dwa.states[u].final_weight {
            reach_u |= fw;
        }
        
        // Include outputs reachable via transitions
        // Only consider transitions that have explicit weights (dead transitions don't contribute)
        for (&label, &target) in &dwa.states[u].transitions {
            if target < n {
                if let Some(trans_w) = dwa.states[u].trans_weights.get(&label) {
                    // Tokens that can use this transition AND reach outputs from target
                    reach_u |= &(trans_w & &reachable[target]);
                }
            }
        }
        
        reachable[u] = reach_u;
    }

    // 3. Push reachable outputs into transition weights
    let mut changed = false;
    for u in 0..n {
        for (&label, &target) in dwa.states[u].transitions.clone().iter() {
            if target < n {
                if let Some(w) = dwa.states[u].trans_weights.get_mut(&label) {
                    // New weight = old_weight AND reachable[target]
                    // This restricts the transition to only tokens that can actually produce output
                    let new_w = &*w & &reachable[target];
                    if *w != new_w {
                        *w = new_w;
                        changed = true;
                    }
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
        // Only minimize reachable states
        if needed[id].is_empty() && id != dwa.body.start_state {
            continue;
        }
        states_by_height[h].push(id);
    }
    crate::debug!(6, "Acyclic minimize step 3 (heights): {:?}, max_height={}, largest_level={}", 
        step3_start.elapsed(), max_height,
        states_by_height.iter().map(|v| v.len()).max().unwrap_or(0));

    // 4. Bottom-Up Exact Minimization
    let step4_start = std::time::Instant::now();
    // We map old_id -> new_id (in the minimized machine).
    let mut old_to_new: HashMap<StateID, StateID> = HashMap::new();
    let mut new_states: Vec<MergedStateBuilder> = Vec::new();

    // Track time spent in major operations
    let mut total_incomp_time = std::time::Duration::ZERO;
    let mut total_coloring_time = std::time::Duration::ZERO;
    let mut total_merge_time = std::time::Duration::ZERO;

    // Process from leaves (height 0) upwards
    for h in 0..=max_height {
        let candidates = &states_by_height[h];
        if candidates.is_empty() { continue; }

        // For large height levels, use fast signature-based coloring
        // This avoids O(n²) incompatibility graph construction
        // Threshold lowered because are_compatible is expensive (Weight operations)
        // Note: 200 threshold chosen because ~200 candidates causes ~20k are_compatible calls
        // which at ~0.2ms each becomes ~4 seconds
        let coloring = if candidates.len() > 200 {
            let sig_start = std::time::Instant::now();
            let result = solve_signature_based_coloring(
                &dwa,
                candidates,
                &needed,
                &old_to_new,
                &new_states,
            );
            let sig_time = sig_start.elapsed();
            total_incomp_time += sig_time; // Count as incomp time
            // Only log for significant runs (>500 candidates means this is notable)
            crate::debug!(5, "Height {}: {} candidates, signature coloring took {:?}", 
                h, candidates.len(), sig_time);
            result
        } else {
            // A. Build Incompatibility Graph for this layer
            let incomp_start = std::time::Instant::now();
            let adj = build_incompatibility_graph(
                &dwa,
                candidates,
                &needed,
                &old_to_new,
                &new_states,
            );
            let incomp_time = incomp_start.elapsed();
            total_incomp_time += incomp_time;
            
            // Only log for levels with many candidates
            if candidates.len() > 200 {
                crate::debug!(5, "Height {}: {} candidates, incomp graph took {:?}", 
                    h, candidates.len(), incomp_time);
            }

            // B. Solve Exact Graph Coloring to find minimum cliques
            let coloring_start = std::time::Instant::now();
            let result = solve_exact_graph_coloring(&adj);
            let coloring_time = coloring_start.elapsed();
            total_coloring_time += coloring_time;
            
            // Log slow coloring operations to identify bottlenecks
            if coloring_time.as_millis() > 100 {
                crate::debug!(5, "Height {}: {} candidates, graph coloring took {:?}", 
                    h, candidates.len(), coloring_time);
            }
            result
        };

        // C. Construct new merged states from color classes
        let merge_start = std::time::Instant::now();
        let base_new_id = new_states.len();
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        for (old_idx, color) in coloring.iter().enumerate() {
            let old_id = candidates[old_idx];
            let new_id = base_new_id + *color;
            old_to_new.insert(old_id, new_id);
        }

        // Create the actual builder structs for the new states
        for _ in 0..num_colors {
            new_states.push(MergedStateBuilder::default());
        }

        // Merge logic: Combine transitions and finals
        // We split new_states to allow immutable access to previously completed layers
        // while holding a mutable reference to the current layer's builders.
        let (completed, builders) = new_states.split_at_mut(base_new_id);

        for (old_idx, &color) in coloring.iter().enumerate() {
            let old_id = candidates[old_idx];
            let builder = &mut builders[color];
            let old_state = &dwa.states[old_id];

            // Union Final Weights
            if let Some(fw) = &old_state.final_weight {
                builder.final_weight |= fw;
            }

            // Union Needed Sets (for upstream calculation)
            builder.needed |= &needed[old_id];

            // Merge Transitions
            for (&label, &target_old) in &old_state.transitions {
                if target_old >= dwa.states.len() { continue; }
                
                // If the target state was skipped (e.g. not needed), ignore this transition branch
                if !old_to_new.contains_key(&target_old) { continue; }

                // Only consider transitions that have explicit weights
                // Transitions without weights are "dead" (get_transition returns None)
                if let Some(w_orig) = old_state.trans_weights.get(&label) {
                    let target_new = old_to_new[&target_old];

                    // CRITICAL OPTIMIZATION:
                    // Effectively w_trans = w_orig & Needed[target_old].
                    // Since target is already merged, we use completed[target_new].needed.
                    let mut w_effective = w_orig.clone();
                    w_effective &= &completed[target_new].needed;

                    if !w_effective.is_empty() {
                        builder.add_transition(label, target_new, w_effective);
                    }
                }
            }
        }
        total_merge_time += merge_start.elapsed();
    }

    crate::debug!(6, "Acyclic minimize step 4 totals: incomp={:?}, coloring={:?}, merge={:?}", 
        total_incomp_time, total_coloring_time, total_merge_time);
    crate::debug!(6, "Acyclic minimize: {} -> {} states in {:?}", 
        dwa.states.len(), new_states.len(), total_start.elapsed());

    // 5. Reconstruct the Final DWA
    let result = reconstruct_dwa(dwa.body.start_state, &old_to_new, new_states)?;
    
    // 6. Stochastic validation: sample random pairs and check for missed merges
    // Only run when STOCHASTIC_MERGE_VALIDATION=1 is set
    if std::env::var("STOCHASTIC_MERGE_VALIDATION").is_ok() {
        stochastic_merge_validation(&result)?;
    }
    
    Ok(result)
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
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42); // Deterministic for reproducibility
    let mut mergeable_count = 0;
    let mut total_checked = 0;
    
    for _ in 0..num_samples {
        // Pick two random distinct states
        let pair: Vec<_> = state_ids.choose_multiple(&mut rng, 2).collect();
        if pair.len() < 2 { continue; }
        let (s1, s2) = (*pair[0], *pair[1]);
        if s1 == s2 { continue; }
        
        total_checked += 1;
        
        // Check if they're compatible (could be merged)
        // Two states are compatible if:
        // 1. Same final weight (or both non-final)
        // 2. Same transition targets for all labels
        // 3. Same transition weights
        let state1 = &dwa.states[s1];
        let state2 = &dwa.states[s2];
        
        // Check final weights
        if state1.final_weight != state2.final_weight {
            continue;
        }
        
        // Check transitions - need same domain, same targets
        if state1.transitions != state2.transitions {
            continue;
        }
        
        // Check transition weights
        if state1.trans_weights != state2.trans_weights {
            continue;
        }
        
        // Found a mergeable pair!
        mergeable_count += 1;
        crate::debug!(3, "STOCHASTIC: Found mergeable pair: {:?} and {:?}", s1, s2);
    }
    
    // If we find a significant number of mergeable pairs, something is wrong
    let mergeable_rate = mergeable_count as f64 / total_checked as f64;
    crate::debug!(3, "STOCHASTIC: {} states, checked {} pairs, {} mergeable ({:.2}%)", 
        n, total_checked, mergeable_count, mergeable_rate * 100.0);
    
    if mergeable_count > 0 {
        // Any mergeable pairs after minimization suggests the minimization is not optimal
        panic!(
            "STOCHASTIC MERGE VALIDATION FAILED: Found {} mergeable pairs out of {} checked ({:.2}%). \
             This suggests minimization is suboptimal.",
            mergeable_count, total_checked, mergeable_rate * 100.0
        );
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
        let mut acc = Weight::zeros();
        if let Some(fw) = &dwa.states[u].final_weight {
            acc |= fw;
        }
        for (&lbl, &v) in &dwa.states[u].transitions {
            if v >= dwa.states.len() { continue; }
            // Only consider transitions that have explicit weights
            // Transitions without weights are "dead" (get_transition returns None)
            if let Some(w_trans) = dwa.states[u].trans_weights.get(&lbl) {
                let mut contribution = w_trans.clone();
                contribution &= &needed[v];
                acc |= &contribution;
            }
        }
        needed[u] = acc;
    }
    needed
}

/// Compute forward reachability: which tokens can reach each state from the start
fn compute_forward_reachable(dwa: &DWA, topo_order: &[StateID]) -> Vec<Weight> {
    let mut forward = vec![Weight::zeros(); dwa.states.len()];
    
    // Start state can reach all tokens
    forward[dwa.body.start_state] = Weight::all();
    
    // Process in reverse topo order (from start toward leaves)
    for &u in topo_order.iter().rev() {
        let incoming = forward[u].clone();
        if incoming.is_empty() { continue; }
        
        for (&lbl, &v) in &dwa.states[u].transitions {
            if v >= dwa.states.len() { continue; }
            // Only consider transitions that have explicit weights
            // Transitions without weights are "dead" (get_transition returns None)
            if let Some(w_trans) = dwa.states[u].trans_weights.get(&lbl) {
                // Tokens that can reach v through this transition
                let mut contribution = incoming.clone();
                contribution &= w_trans;
                forward[v] |= &contribution;
            }
        }
    }
    
    forward
}

/// Tighten DWA weights by removing tokens that can never reach a transition.
/// 
/// This is a semantic-preserving transformation that restricts each transition's
/// weight to only include tokens that can actually reach that transition from the start.
/// 
/// The key insight: if a token T can never reach state S, then the weight of S's
/// outgoing transitions doesn't matter for T. By removing T from those weights,
/// we might create more opportunities for state merging (disjoint domains).
fn tighten_weights(dwa: &DWA) -> Result<DWA, DWABuildError> {
    if dwa.states.len() == 0 {
        return Ok(DWA::new());
    }
    
    // Compute topo order
    let topo_order = compute_topo_order(dwa)?;
    
    // Compute forward reachability
    let forward = compute_forward_reachable(dwa, &topo_order);
    
    // Create new DWA with tightened weights
    let mut new_states = DWAStates(Vec::with_capacity(dwa.states.len()));
    
    for (u, state) in dwa.states.0.iter().enumerate() {
        let mut new_state = DWAState::default();
        
        // Tighten final weight: only keep tokens that can reach this state
        if let Some(fw) = &state.final_weight {
            let tightened = fw & &forward[u];
            if !tightened.is_empty() {
                new_state.final_weight = Some(tightened);
            }
        }
        
        // Tighten transition weights
        for (&lbl, &target) in &state.transitions {
            if target >= dwa.states.len() { continue; }
            
            // Only consider transitions that have explicit weights
            // Transitions without weights are "dead" (get_transition returns None)
            if let Some(w_orig) = state.trans_weights.get(&lbl) {
                // Tighten: only keep tokens that can reach this state
                let tightened = w_orig & &forward[u];
                
                if !tightened.is_empty() {
                    new_state.transitions.insert(lbl, target);
                    new_state.trans_weights.insert(lbl, tightened);
                }
            }
        }
        
        new_states.0.push(new_state);
    }
    
    Ok(DWA {
        states: new_states,
        body: dwa.body.clone(),
    })
}

fn compute_heights(dwa: &DWA, topo_order: &[StateID]) -> Vec<usize> {
    let mut heights = vec![0; dwa.states.len()];
    for &u in topo_order {
        let mut h = 0;
        for &v in dwa.states[u].transitions.values() {
            if v < dwa.states.len() {
                h = std::cmp::max(h, heights[v] + 1);
            }
        }
        heights[u] = h;
    }
    heights
}

// --- Phase 3: Compatibility & Coloring ---

/// Check if two states can be merged.
/// 
/// After weight pushing and tightening, states can be merged if:
/// 1. For each label, the transitions are "compatible":
///    - Both have the same target (after mapping)
///    - Their weights are identical on the domain overlap (diamond case)
/// 2. Final weights are compatible on the domain overlap
/// 
/// The diamond case is correct because:
/// - After tighten_weights, trans weights only include tokens that can reach the state
/// - If the behavior is identical on overlapping tokens, merging is safe
/// - Tokens outside the overlap only reach one state anyway (disjoint domains)
fn are_compatible(
    u: StateID,
    v: StateID,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder]
) -> bool {
    // Check if domains are disjoint (diamond case)
    let domain_u = &needed[u];
    let domain_v = &needed[v];
    let domain_overlap = domain_u & domain_v;
    let domains_disjoint = domain_overlap.is_empty();
    
    // Check final weight compatibility on the overlap domain
    // For tokens in the overlap, final weights must produce the same output
    if !domains_disjoint {
        let fw_u = dwa.states[u].final_weight.as_ref()
            .map(|w| w & &domain_overlap)
            .unwrap_or_else(Weight::zeros);
        let fw_v = dwa.states[v].final_weight.as_ref()
            .map(|w| w & &domain_overlap)
            .unwrap_or_else(Weight::zeros);
        if fw_u != fw_v {
            return false;
        }
    }
    
    // Collect all labels present in either state
    let mut labels: BTreeSet<Label> = dwa.states[u].transitions.keys().copied().collect();
    labels.extend(dwa.states[v].transitions.keys());

    for lbl in labels {
        // Check if transitions exist and get their targets
        let target_u = dwa.states[u].transitions.get(&lbl);
        let target_v = dwa.states[v].transitions.get(&lbl);

        // Get weights
        // A transition is only "live" if it has an explicit weight in trans_weights
        // Missing trans_weights entries mean the transition is "dead" (get_transition returns None)
        let w_u_raw = if target_u.is_some() {
            dwa.states[u].trans_weights.get(&lbl).cloned()
        } else {
            None
        };
        let w_v_raw = if target_v.is_some() {
            dwa.states[v].trans_weights.get(&lbl).cloned()
        } else {
            None
        };
        
        // Treat missing weights as empty (dead transitions)
        let w_u_effective = w_u_raw.as_ref().cloned().unwrap_or_else(Weight::zeros);
        let w_v_effective = w_v_raw.as_ref().cloned().unwrap_or_else(Weight::zeros);

        // Check weight compatibility on the overlap domain
        if !domains_disjoint {
            // For tokens in the overlap, transition weights must produce the same output
            let w_u_overlap = &w_u_effective & &domain_overlap;
            let w_v_overlap = &w_v_effective & &domain_overlap;
            if w_u_overlap != w_v_overlap {
                return false;
            }
        }
        // If domains are disjoint, weights can differ (they'll be unioned when merged)
        // But both must go to the same target (after mapping) OR be equivalent on their combined domain

        // CRITICAL: If BOTH states have transitions on this label, they must either:
        // 1. Go to the same target (after mapping), OR
        // 2. Go to different targets that are EQUIVALENT on the combined token flow
        let u_has_live_trans = !w_u_effective.is_empty();
        let v_has_live_trans = !w_v_effective.is_empty();
        
        if u_has_live_trans && v_has_live_trans {
            // Both states have live transitions on this label
            match (target_u, target_v) {
                (Some(&tu), Some(&tv)) => {
                    let target_u_new = old_to_new.get(&tu);
                    let target_v_new = old_to_new.get(&tv);
                    
                    match (target_u_new, target_v_new) {
                        (Some(&u_new), Some(&v_new)) if u_new != v_new => {
                            // Targets differ - check if they're equivalent on the combined token flow
                            let w_combined = &w_u_effective | &w_v_effective;
                            if !targets_equivalent_on_domain(u_new, v_new, &w_combined, new_states) {
                                return false;
                            }
                            // Targets are equivalent on the combined domain - can merge
                        }
                        (Some(_), None) | (None, Some(_)) => return false,
                        (None, None) => {
                            // Both targets not yet processed - must be same original state
                            if tu != tv {
                                return false;
                            }
                        }
                        _ => {} // Both mapped to same state
                    }
                }
                (Some(_), None) | (None, Some(_)) => {
                    // One has target, other doesn't (but both have weights) - shouldn't happen
                    // If there's a live weight, there should be a transition
                    return false;
                }
                (None, None) => {} // Neither has target (shouldn't happen with live weights)
            }
        } else if !domains_disjoint {
            // Domains overlap, so check if one has live transition on overlap when other doesn't
            let w_u_overlap = &w_u_effective & &domain_overlap;
            let w_v_overlap = &w_v_effective & &domain_overlap;
            if (!w_u_overlap.is_empty() && w_v_overlap.is_empty()) ||
               (w_u_overlap.is_empty() && !w_v_overlap.is_empty()) {
                // One has transition on overlap, other doesn't - incompatible
                return false;
            }
        }
        // If domains are disjoint AND only one has transition on this label, that's fine
        // The merged state will just have that transition restricted to its domain
    }

    true
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
        
        // Weights on domain must match
        if w_u != w_v {
            return false;
        }
        // If there's weight, targets must match
        if !w_u.is_empty() && target_u != target_v {
            return false;
        }
    }
    
    true
}

/// Compute a structural signature for a state.
/// States with identical signatures can potentially be merged.
/// The signature captures: final_weight + (label, target_new, weight) for each transition
fn compute_state_signature(
    state_id: StateID,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
) -> Vec<u64> {
    let state = &dwa.states[state_id];
    let mut sig: Vec<u64> = Vec::new();
    
    // Include needed set hash (domain)
    let domain = &needed[state_id];
    sig.push(domain.fast_hash());
    
    // Include final weight restricted to domain
    let fw_on_domain = state.final_weight.as_ref()
        .map(|w| w & domain)
        .unwrap_or_else(Weight::zeros);
    sig.push(fw_on_domain.fast_hash());
    
    // Include transitions sorted by label
    let mut trans_sigs: Vec<(i32, usize, u64)> = Vec::new();
    for (&label, &target) in &state.transitions {
        if target >= dwa.states.len() { continue; }
        
        // Get mapped target (if available)
        let target_new = old_to_new.get(&target).copied().unwrap_or(usize::MAX);
        
        // Get weight restricted to domain
        let w = state.trans_weights.get(&label)
            .map(|w| w & domain)
            .unwrap_or_else(Weight::zeros);
        
        if !w.is_empty() {
            trans_sigs.push((label, target_new, w.fast_hash()));
        }
    }
    trans_sigs.sort();
    
    for (label, target, hash) in trans_sigs {
        sig.push(label as u64);
        sig.push(target as u64);
        sig.push(hash);
    }
    
    sig
}

fn build_incompatibility_graph(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder]
) -> Vec<Vec<usize>> {
    let n = candidates.len();
    
    // For large candidate sets, use signature-based pre-grouping
    // This avoids O(n²) are_compatible calls which are expensive
    if n > 200 {
        return build_incompatibility_graph_with_signatures(dwa, candidates, needed, old_to_new, new_states);
    }
    
    // For small sets, use the direct O(n²) approach
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

fn build_incompatibility_graph_with_signatures(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder]
) -> Vec<Vec<usize>> {
    let n = candidates.len();
    
    // Step 1: Compute signatures for all candidates
    let signatures: Vec<Vec<u64>> = candidates.iter()
        .map(|&s| compute_state_signature(s, dwa, needed, old_to_new))
        .collect();
    
    // Step 2: Group candidates by signature
    let mut sig_to_indices: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    for (idx, sig) in signatures.iter().enumerate() {
        sig_to_indices.entry(sig.clone()).or_default().push(idx);
    }
    
    let num_groups = sig_to_indices.len();
    
    // FAST PATH: For very large problems with many signature groups,
    // the cross-group comparison is O(groups² * avg_group_size²) which can be huge.
    // Instead, assume cross-group states are incompatible (conservative but fast).
    // This loses some diamond-case merging opportunities but avoids O(n²) blowup.
    if num_groups > 50 && n > 5000 {
        // Only check pairs within the SAME signature group for incompatibility
        let mut adj = vec![vec![]; n];
        
        for indices in sig_to_indices.values() {
            if indices.len() <= 1 { continue; }
            for i in 0..indices.len() {
                for j in (i+1)..indices.len() {
                    let idx_i = indices[i];
                    let idx_j = indices[j];
                    if !are_compatible(candidates[idx_i], candidates[idx_j], dwa, needed, old_to_new, new_states) {
                        adj[idx_i].push(idx_j);
                        adj[idx_j].push(idx_i);
                    }
                }
            }
        }
        
        // We don't add cross-group edges because we'll handle this differently:
        // We'll use signature-aware coloring that assigns different base colors to different signatures
        return adj;
    }
    
    // Standard path for smaller problems
    let mut adj = vec![vec![]; n];
    
    // Check pairs within the same signature group
    for indices in sig_to_indices.values() {
        if indices.len() <= 1 { continue; }
        for i in 0..indices.len() {
            for j in (i+1)..indices.len() {
                let idx_i = indices[i];
                let idx_j = indices[j];
                if !are_compatible(candidates[idx_i], candidates[idx_j], dwa, needed, old_to_new, new_states) {
                    adj[idx_i].push(idx_j);
                    adj[idx_j].push(idx_i);
                }
            }
        }
    }
    
    // Check cross-group domain overlaps
    let sig_keys: Vec<&Vec<u64>> = sig_to_indices.keys().collect();
    for i in 0..sig_keys.len() {
        for j in (i+1)..sig_keys.len() {
            let indices_i = &sig_to_indices[sig_keys[i]];
            let indices_j = &sig_to_indices[sig_keys[j]];
            
            for &idx_i in indices_i {
                for &idx_j in indices_j {
                    let domain_overlap = &needed[candidates[idx_i]] & &needed[candidates[idx_j]];
                    if !domain_overlap.is_empty() {
                        adj[idx_i].push(idx_j);
                        adj[idx_j].push(idx_i);
                    }
                }
            }
        }
    }
    
    adj
}

/// Fast O(n log n) coloring for very large height levels.
/// Uses signatures to group states, assigns each signature group a unique color range.
/// This is conservative (may produce more colors than optimal) but avoids O(n²) blowup.
fn solve_signature_based_coloring(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder],
) -> Vec<usize> {
    let n = candidates.len();
    if n == 0 { return vec![]; }
    
    // Step 1: Compute signatures for all candidates
    let signatures: Vec<Vec<u64>> = candidates.iter()
        .map(|&s| compute_state_signature(s, dwa, needed, old_to_new))
        .collect();
    
    // Step 2: Group candidates by signature
    let mut sig_to_indices: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    for (idx, sig) in signatures.iter().enumerate() {
        sig_to_indices.entry(sig.clone()).or_default().push(idx);
    }
    
    // Debug: report signature distribution (only at high verbosity)
    let num_groups = sig_to_indices.len();
    let max_group_size = sig_to_indices.values().map(|v| v.len()).max().unwrap_or(0);
    let singleton_groups = sig_to_indices.values().filter(|v| v.len() == 1).count();
    crate::debug!(6, "  Signature groups: {} total, {} singletons, max_group_size={}", 
        num_groups, singleton_groups, max_group_size);
    
    // Step 3: Assign colors
    // Each signature group gets its own range of colors
    // States with the same signature hash are LIKELY compatible, but we need to verify
    let mut colors = vec![0usize; n];
    let mut next_color = 0;
    
    // Process signature groups
    for (_sig, indices) in sig_to_indices.iter() {
        if indices.len() == 1 {
            // Single state in group - assign one color
            colors[indices[0]] = next_color;
            next_color += 1;
        } else {
            // Multiple states with same signature hash
            // Build incompatibility graph within this group using full are_compatible checks
            // This is O(m²) where m = group size, which is manageable for reasonable group sizes
            let group_size = indices.len();
            let mut local_adj = vec![vec![]; group_size];
            
            for i in 0..group_size {
                for j in (i+1)..group_size {
                    let idx_i = indices[i];
                    let idx_j = indices[j];
                    
                    // Use full are_compatible check to verify actual compatibility
                    if !are_compatible(candidates[idx_i], candidates[idx_j], dwa, needed, old_to_new, new_states) {
                        local_adj[i].push(j);
                        local_adj[j].push(i);
                    }
                }
            }
            
            // Use greedy coloring within this group
            let local_colors = solve_greedy_coloring(&local_adj);
            let local_max = local_colors.iter().max().copied().unwrap_or(0);
            
            // Map local colors to global colors
            for (local_idx, &local_color) in local_colors.iter().enumerate() {
                colors[indices[local_idx]] = next_color + local_color;
            }
            
            next_color += local_max + 1;
        }
    }
    
    colors
}

/// Greedy graph coloring - fast but not optimal
fn solve_greedy_coloring(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    let n = adj.len();
    if n == 0 { return vec![]; }

    let mut colors = vec![usize::MAX; n];
    
    // Sort by degree (high degree nodes first)
    let mut nodes: Vec<usize> = (0..n).collect();
    nodes.sort_by_key(|&i| std::cmp::Reverse(adj[i].len()));

    for &u in &nodes {
        // Find smallest color not used by neighbors
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

// --- Phase 4: Reconstruction ---

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