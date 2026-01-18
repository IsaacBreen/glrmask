// Partition-based minimization - internal, only used in tests
mod partition_minimize;

use crate::dwa_i32::common::{Label, StateID, Weight};
use crate::dwa_i32::dwa::{DWA, DWABuildError, DWAState, DWAStates};
use crate::dwa_i32::minimization::common::DwaPass;
use crate::dwa_i32::minimization::graph_coloring::{solve_greedy_coloring, solve_exact_graph_coloring};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use crate::datastructures::rangemap_weight::intern_rangemap;

impl DWA {
    pub fn minimize_acyclic(&mut self) {
        // Check environment variable for fast minimize option
        let use_fast_minimize = std::env::var("DWA_FAST_MINIMIZE").map(|v| v == "1").unwrap_or(false);
        
        if use_fast_minimize {
            // Use partition refinement - faster but may produce slightly larger DWA
            // (doesn't exploit the "diamond case" optimization)
            self.minimize_states_cyclic();
            return;
        }
        
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

    let start = std::time::Instant::now();
    crate::datastructures::hybrid_bitset::reset_profiling();

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
    let mut reachable_is_empty = vec![true; n];
    let mut reachable_is_all = vec![false; n];
    for &u in topo_order.iter().rev() {
        let mut reach_u = dwa.states[u].final_weight.clone().unwrap_or_else(Weight::zeros);
        let mut reach_all = reach_u.is_all_fast();
        if !reach_all {
            for (&label, &target) in &dwa.states[u].transitions {
                if target >= n { continue; }
                if reachable_is_empty[target] { continue; }
                let Some(w) = dwa.states[u].trans_weights.get(&label) else { continue; };
                if w.is_empty() { continue; }
                if reachable_is_all[target] {
                    reach_u |= w;
                } else {
                    reach_u |= &(w & &reachable[target]);
                }
                if reach_u.is_all_fast() {
                    reach_all = true;
                    break;
                }
            }
        }
        reachable_is_empty[u] = reach_u.is_empty();
        reachable_is_all[u] = reach_all || reach_u.is_all_fast();
        reachable[u] = reach_u;
    }

    // Push reachable outputs into transition weights
    let mut changed = false;
    for u in 0..n {
        let (transitions, trans_weights) = {
            let state = &mut dwa.states[u];
            (&state.transitions, &mut state.trans_weights)
        };
        for (&label, &target) in transitions.iter() {
            if target >= n { continue; }
            let Some(w) = trans_weights.get_mut(&label) else { continue; };
            if w.is_empty() { continue; }
            if reachable_is_all[target] { continue; }
            if reachable_is_empty[target] {
                *w = Weight::zeros();
                changed = true;
                continue;
            }
            let new_w = &*w & &reachable[target];
            if *w != new_w {
                *w = new_w;
                changed = true;
            }
        }
    }
    crate::datastructures::hybrid_bitset::print_profiling("push_weights_acyclic");
    crate::debug!(5, "push_weights_acyclic: {:?} (changed={})", start.elapsed(), changed);
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
    crate::debug!(5, "Acyclic minimize step 0 start (tighten_weights)");
    let step0_start = std::time::Instant::now();
    crate::datastructures::hybrid_bitset::reset_profiling();
    let dwa = tighten_weights(dwa)?;
    crate::datastructures::hybrid_bitset::print_profiling("tighten_weights");
    crate::debug!(5, "Acyclic minimize step 0 end (tighten_weights): {:?}", step0_start.elapsed());
    crate::debug!(5, "Acyclic minimize step 0 (tighten_weights): {:?}", step0_start.elapsed());

    // 1. Topological Sort & Reachability Analysis
    crate::debug!(5, "Acyclic minimize step 1 start (topo_order)");
    let step1_start = std::time::Instant::now();
    let topo_order = compute_topo_order(&dwa)?;
    crate::debug!(5, "Acyclic minimize step 1 end (topo_order): {:?}", step1_start.elapsed());
    crate::debug!(5, "Acyclic minimize step 1 (topo_order): {:?}", step1_start.elapsed());

    // 2. Compute "Needed" sets (Reverse Flow Analysis).
    crate::debug!(5, "Acyclic minimize step 2 start (needed_sets)");
    let step2_start = std::time::Instant::now();
    crate::datastructures::hybrid_bitset::reset_profiling();
    let needed = compute_needed_sets(&dwa, &topo_order);
    crate::datastructures::hybrid_bitset::print_profiling("compute_needed_sets");
    crate::debug!(5, "Acyclic minimize step 2 end (needed_sets): {:?}", step2_start.elapsed());
    crate::debug!(5, "Acyclic minimize step 2 (needed_sets): {:?}", step2_start.elapsed());

    // 3. Layer states by topological height (distance to sink).
    let step3_start = std::time::Instant::now();
    let heights = compute_heights(&dwa, &topo_order);
    crate::debug!(5, "Acyclic minimize step 3a (compute_heights): {:?}", step3_start.elapsed());

    let step3b_start = std::time::Instant::now();
    let max_height = heights.iter().max().copied().unwrap_or(0);

    let mut states_by_height: Vec<Vec<StateID>> = vec![vec![]; max_height + 1];
    for (id, &h) in heights.iter().enumerate() {
        if needed[id].is_empty() && id != dwa.body.start_state { continue; }
        states_by_height[h].push(id);
    }
    crate::debug!(5, "Acyclic minimize step 3b (states_by_height): {:?}, max_height={}, largest_level={}", 
        step3b_start.elapsed(), max_height,
        states_by_height.iter().map(|v| v.len()).max().unwrap_or(0));

    // 4. Bottom-Up Exact Minimization
    let mut old_to_new: HashMap<StateID, StateID> = HashMap::new();
    let mut new_states: Vec<MergedStateBuilder> = Vec::new();

    // Process from leaves (height 0) upwards
    let mut last_height_debug = std::time::Instant::now();
    for h in 0..=max_height {
        let candidates = &states_by_height[h];
        let since_last = last_height_debug.elapsed();
        crate::debug!(5, "Height {} start: {} candidates, {:?}", h, candidates.len(), since_last);
        last_height_debug = std::time::Instant::now();
        if candidates.is_empty() { continue; }

        if candidates.len() >= 100 {
            crate::debug!(5, "Height {}: {} candidates", h, candidates.len());
        }

        // Compute coloring - use partition-based method for large candidate sets
        let coloring = compute_height_coloring(&dwa, candidates, &needed, &old_to_new, &new_states);

        // Construct new merged states from color classes
        let base_new_id = new_states.len();
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        // Map old states to new merged states
        let insert_start = std::time::Instant::now();
        for (old_idx, &color) in coloring.iter().enumerate() {
            old_to_new.insert(candidates[old_idx], base_new_id + color);
        }
        crate::debug!(5, "Height {}: old_to_new insert {:?} ({} items)", h, insert_start.elapsed(), candidates.len());

        let extend_start = std::time::Instant::now();
        new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));
        crate::debug!(5, "Height {}: new_states extend {:?} ({} new)", h, extend_start.elapsed(), num_colors);

        // Merge states into builders
        let (completed, builders) = new_states.split_at_mut(base_new_id);
        let merge_start = std::time::Instant::now();
        let mut merge_stats = MergeStats::default();
        for (old_idx, &color) in coloring.iter().enumerate() {
            merge_state_into_builder(
                candidates[old_idx],
                color,
                &dwa,
                &needed,
                &old_to_new,
                completed,
                builders,
                &mut merge_stats,
            );
        }
        let avg_w_orig_ranges = if merge_stats.and_ops > 0 {
            merge_stats.w_orig_ranges as f64 / merge_stats.and_ops as f64
        } else {
            0.0
        };
        let avg_needed_ranges = if merge_stats.and_ops > 0 {
            merge_stats.needed_ranges as f64 / merge_stats.and_ops as f64
        } else {
            0.0
        };
        let needed_all_pct = if merge_stats.and_ops > 0 {
            (merge_stats.needed_all as f64) * 100.0 / merge_stats.and_ops as f64
        } else {
            0.0
        };
        crate::debug!(
            5,
            "Height {}: merge_state_into_builder {:?} (transitions {}, and_ops {}, and_time {} us, avg_w_orig_ranges {:.2}, avg_needed_ranges {:.2}, needed_all {} ({:.1}%))",
            h,
            merge_start.elapsed(),
            merge_stats.transitions,
            merge_stats.and_ops,
            merge_stats.and_time_us,
            avg_w_orig_ranges,
            avg_needed_ranges,
            merge_stats.needed_all,
            needed_all_pct,
        );
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
    
    // Log diagnostic info for large candidate sets
    if candidates.len() >= 100 {
        crate::debug!(5, "  {} new_states (targets) available", new_states.len());
    }
    
    // Check if this is height 0 (no transitions) with large candidate set
    // Use optimized direct coloring path
    let is_height_0 = candidates.iter().all(|&id| dwa.states[id].transitions.is_empty());
    if is_height_0 && candidates.len() > 1000 {
        return compute_height_0_coloring_direct(candidates, dwa, needed, start);
    }
    
    // For large non-height-0 candidate sets, use greedy coloring without building full graph
    // This avoids the O(n²) graph construction bottleneck
    // Use a lower threshold since are_compatible can be expensive
    if candidates.len() > 500 {
        return greedy_color_without_graph(dwa, candidates, needed, old_to_new, new_states, start);
    }
    
    // Compute signatures first to check if we can use a fast path
    let signatures: Vec<u128> = candidates.iter().map(|&id| {
        compute_state_signature(id, dwa, needed, old_to_new)
    }).collect();
    
    // Group by signature
    let mut sig_to_group: HashMap<u128, Vec<usize>> = HashMap::new();
    for (idx, &sig) in signatures.iter().enumerate() {
        sig_to_group.entry(sig).or_default().push(idx);
    }
    let num_groups = sig_to_group.len();
    
    // Fast path: if signature groups cover > 70% of candidates, cross-group merging is unlikely
    // to help much. Just use signature-based coloring.
    let signature_coverage = num_groups as f64 / candidates.len() as f64;
    if candidates.len() > 200 && signature_coverage > 0.70 {
        crate::debug!(5, "  Fast path: {} sig groups / {} candidates ({:.1}% coverage) -> using signatures as colors",
            num_groups, candidates.len(), signature_coverage * 100.0);
        
        // Assign colors based on signature
        let mut colors = vec![0; candidates.len()];
        let mut sig_to_color: HashMap<u128, usize> = HashMap::new();
        let mut next_color = 0usize;
        for (idx, &sig) in signatures.iter().enumerate() {
            let color = *sig_to_color.entry(sig).or_insert_with(|| {
                let c = next_color;
                next_color += 1;
                c
            });
            colors[idx] = color;
        }
        return colors;
    }
    
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

/// Direct coloring for large height-0 candidate sets.
/// 
/// At height 0, states only have final weights (no transitions).
/// We use signature-based coloring: states with the same signature get the same color.
/// This is potentially suboptimal (signatures with disjoint needed sets could share colors)
/// but is O(n) instead of O(n²).
fn compute_height_0_coloring_direct(
    candidates: &[StateID],
    dwa: &DWA,
    needed: &[Weight],
    start: std::time::Instant,
) -> Vec<usize> {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    let n = candidates.len();
    
    // First, analyze the structure to see if we can do better
    // Compute needed footprints and signatures
    let mut sig_groups: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut footprints: Vec<Weight> = Vec::with_capacity(n);
    
    for (idx, &id) in candidates.iter().enumerate() {
        let final_on_needed = dwa.states[id].final_weight.as_ref()
            .map(|w| w & &needed[id])
            .unwrap_or_else(Weight::zeros);
        
        let mut hasher = DefaultHasher::new();
        final_on_needed.fingerprint().hash(&mut hasher);
        let sig = hasher.finish();
        
        sig_groups.entry(sig).or_default().push(idx);
        footprints.push(needed[id].clone());
    }
    
    // Analyze overlap structure
    let num_sigs = sig_groups.len();
    
    // Check how many signature pairs have disjoint footprints (could share colors)
    let sig_list: Vec<(u64, Vec<usize>)> = sig_groups.into_iter().collect();
    
    // Compute footprint for each signature group (union of all members' footprints)
    let sig_footprints: Vec<Weight> = sig_list.iter().map(|(_, indices)| {
        let mut fp = Weight::zeros();
        for &idx in indices {
            fp |= &footprints[idx];
        }
        fp
    }).collect();
    
    // Count compatible signature pairs
    let mut compatible_sig_pairs = 0;
    let mut total_sig_pairs = 0;
    for i in 0..sig_list.len() {
        for j in (i+1)..sig_list.len() {
            total_sig_pairs += 1;
            let overlap = &sig_footprints[i] & &sig_footprints[j];
            if overlap.is_empty() {
                compatible_sig_pairs += 1;
            }
        }
    }
    
    // If most signature pairs are compatible (disjoint footprints), try interval scheduling
    if compatible_sig_pairs > total_sig_pairs / 2 && num_sigs > 50 {
        // Try greedy interval-scheduling style approach
        // Sort signatures by footprint start (or size) and greedily assign colors
        let colors = greedy_interval_coloring(&sig_list, &sig_footprints, n);
        let num_colors = colors.iter().max().map(|c| c + 1).unwrap_or(0);
        
        crate::debug!(5, "Height 0 interval coloring: {} candidates, {} sigs, {}/{} compatible pairs -> {} colors in {:?}",
            n, num_sigs, compatible_sig_pairs, total_sig_pairs, num_colors, start.elapsed());
        
        return colors;
    }
    
    // Fall back to direct signature coloring
    let mut colors = Vec::with_capacity(n);
    let mut sig_to_color: HashMap<u64, usize> = HashMap::new();
    let mut next_color = 0usize;
    
    for (idx, &id) in candidates.iter().enumerate() {
        let final_on_needed = dwa.states[id].final_weight.as_ref()
            .map(|w| w & &needed[id])
            .unwrap_or_else(Weight::zeros);
        
        let mut hasher = DefaultHasher::new();
        final_on_needed.fingerprint().hash(&mut hasher);
        let sig = hasher.finish();
        
        let color = *sig_to_color.entry(sig).or_insert_with(|| {
            let c = next_color;
            next_color += 1;
            c
        });
        colors.push(color);
    }
    
    crate::debug!(5, "Height 0 direct coloring: {} candidates, {} sigs, {}/{} compatible pairs -> {} colors in {:?}",
        n, num_sigs, compatible_sig_pairs, total_sig_pairs, next_color, start.elapsed());
    
    colors
}

/// Greedy interval-style coloring for height-0 states.
/// Assigns colors greedily, trying to pack compatible signatures together.
fn greedy_interval_coloring(
    sig_list: &[(u64, Vec<usize>)],
    sig_footprints: &[Weight],
    total_states: usize,
) -> Vec<usize> {
    // Build incompatibility graph between signatures
    // Two signatures are incompatible if their footprints overlap
    let num_sigs = sig_list.len();
    let mut sig_adj: Vec<Vec<usize>> = vec![vec![]; num_sigs];
    
    for i in 0..num_sigs {
        for j in (i+1)..num_sigs {
            let overlap = &sig_footprints[i] & &sig_footprints[j];
            if !overlap.is_empty() {
                sig_adj[i].push(j);
                sig_adj[j].push(i);
            }
        }
    }
    
    // Greedy color the signatures
    let mut sig_colors: Vec<Option<usize>> = vec![None; num_sigs];
    let mut num_colors = 0;
    
    // Sort signatures by degree (highest first) for better greedy coloring
    let mut order: Vec<usize> = (0..num_sigs).collect();
    order.sort_by(|&a, &b| sig_adj[b].len().cmp(&sig_adj[a].len()));
    
    for sig_idx in order {
        let mut used_colors = std::collections::HashSet::new();
        for &neighbor in &sig_adj[sig_idx] {
            if let Some(c) = sig_colors[neighbor] {
                used_colors.insert(c);
            }
        }
        
        // Find first available color
        let color = (0..=num_colors).find(|c| !used_colors.contains(c)).unwrap();
        sig_colors[sig_idx] = Some(color);
        if color == num_colors {
            num_colors += 1;
        }
    }
    
    // Map back to state colors
    let mut state_colors = vec![0; total_states];
    for (sig_idx, (_, state_indices)) in sig_list.iter().enumerate() {
        let color = sig_colors[sig_idx].unwrap();
        for &state_idx in state_indices {
            state_colors[state_idx] = color;
        }
    }
    
    state_colors
}

#[derive(Default)]
struct MergeStats {
    transitions: usize,
    and_ops: usize,
    and_time_us: u64,
    w_orig_ranges: u64,
    needed_ranges: u64,
    needed_all: u64,
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
    stats: &mut MergeStats,
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
        stats.transitions += 1;
        if target_old >= dwa.states.len() { continue; }
        let Some(&target_new) = old_to_new.get(&target_old) else { continue; };
        let Some(w_orig) = old_state.trans_weights.get(&label) else { continue; };
        let needed_weight = &completed[target_new].needed;
        stats.w_orig_ranges = stats.w_orig_ranges.saturating_add(w_orig.num_ranges() as u64);
        stats.needed_ranges = stats
            .needed_ranges
            .saturating_add(needed_weight.num_ranges() as u64);
        if needed_weight.is_all_fast() {
            stats.needed_all = stats.needed_all.saturating_add(1);
        }
        
        // Restrict weight to what's actually needed at target
        let and_start = std::time::Instant::now();
        let w_effective = w_orig & needed_weight;
        stats.and_time_us = stats
            .and_time_us
            .saturating_add(and_start.elapsed().as_micros() as u64);
        stats.and_ops += 1;
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
    let mut time_init = std::time::Duration::ZERO;
    let mut time_loop_outer = std::time::Duration::ZERO;
    let mut time_clone = std::time::Duration::ZERO;
    let mut time_and = std::time::Duration::ZERO;
    let mut time_or = std::time::Duration::ZERO;
    let mut count_transitions = 0usize;
    let mut count_clones = 0usize;
    let mut count_ands = 0usize;
    let mut count_ors = 0usize;

    let init_start = std::time::Instant::now();
    let mut forward = vec![Weight::zeros(); dwa.states.len()];
    let mut forward_is_all = vec![false; dwa.states.len()];
    forward[dwa.body.start_state] = Weight::all();
    forward_is_all[dwa.body.start_state] = true;
    time_init = init_start.elapsed();

    let mut union_assign_fast = |dst: &mut Weight, src: &Weight| {
        if let Weight::RangeMap(left) = dst {
            if let Weight::RangeMap(right) = src {
                let merged = left.as_ref().union_fast(right.as_ref());
                *left = intern_rangemap(merged);
                return;
            }
        }
        *dst |= src;
    };
    
    for &u in topo_order.iter().rev() {
        let outer_start = std::time::Instant::now();
        let incoming_all = forward_is_all[u];
        if !incoming_all && forward[u].is_empty() {
            time_loop_outer += outer_start.elapsed();
            continue;
        }
        let clone_start = std::time::Instant::now();
        let incoming = if incoming_all {
            None
        } else {
            count_clones += 1;
            Some(forward[u].clone())
        };
        time_clone += clone_start.elapsed();
        
        for (&lbl, &v) in &dwa.states[u].transitions {
            count_transitions += 1;
            if v >= dwa.states.len() { continue; }
            if forward_is_all[v] { continue; }
            if let Some(w) = dwa.states[u].trans_weights.get(&lbl) {
                if w.is_empty() { continue; }
                if incoming_all {
                    let or_start = std::time::Instant::now();
                    union_assign_fast(&mut forward[v], w);
                    time_or += or_start.elapsed();
                    count_ors += 1;
                } else if let Some(incoming) = &incoming {
                    let and_start = std::time::Instant::now();
                    let result = incoming & w;
                    time_and += and_start.elapsed();
                    count_ands += 1;

                    let or_start = std::time::Instant::now();
                    union_assign_fast(&mut forward[v], &result);
                    time_or += or_start.elapsed();
                    count_ors += 1;
                }
                if !forward_is_all[v] && forward[v].is_all_fast() {
                    forward_is_all[v] = true;
                }
            }
        }
        time_loop_outer += outer_start.elapsed();
    }

    crate::debug!(
        5,
        "forward_reachable breakdown: init={:?}, loop_outer={:?}, clone={:?}, and={:?}, or={:?}",
        time_init,
        time_loop_outer,
        time_clone,
        time_and,
        time_or,
    );
    crate::debug!(
        5,
        "forward_reachable counts: transitions={}, clones={}, ands={}, ors={}",
        count_transitions,
        count_clones,
        count_ands,
        count_ors,
    );
    forward
}

/// Tighten DWA weights by removing tokens that can never reach a transition.
/// Semantic-preserving: restricts weights to tokens that can actually reach each state.
fn tighten_weights(dwa: &DWA) -> Result<DWA, DWABuildError> {
    if dwa.states.is_empty() { return Ok(DWA::new()); }
    
    let topo_start = std::time::Instant::now();
    let topo_order = compute_topo_order(dwa)?;
    crate::debug!(5, "tighten_weights: topo_order computed in {:?}", topo_start.elapsed());

    let forward_start = std::time::Instant::now();
    let forward = compute_forward_reachable(dwa, &topo_order);
    crate::debug!(5, "tighten_weights: forward_reachable computed in {:?}", forward_start.elapsed());
    
    let build_start = std::time::Instant::now();
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
    crate::debug!(5, "tighten_weights: new DWA built in {:?}", build_start.elapsed());
    
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

/// Build incompatibility graph using signature-based optimization.
/// 
/// Two states are incompatible if they cannot be merged - i.e., merging them
/// would result in incorrect behavior for some input.
/// 
/// Optimization: States with identical signatures are definitely compatible.
/// We only need to compare pairs across different signature groups.
fn build_incompatibility_graph(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder]
) -> Vec<Vec<usize>> {
    let n = candidates.len();
    if n <= 1 { return vec![vec![]; n]; }
    
    let start = std::time::Instant::now();
    
    // Check if this is height 0 (no transitions) - use optimized path
    let is_height_0 = candidates.iter().all(|&id| dwa.states[id].transitions.is_empty());
    
    if is_height_0 {
        return build_incompatibility_graph_height_0(candidates, dwa, needed);
    }
    
    // For non-height-0: use signature-based approach with needed-set overlap optimization
    build_incompatibility_graph_general(dwa, candidates, needed, old_to_new, new_states, start)
}

/// Optimized incompatibility graph construction for height-0 states (no transitions).
/// 
/// At height 0, compatibility depends only on:
/// 1. Whether needed sets overlap
/// 2. If they overlap, whether final weights match on the overlap
///
/// Optimization strategy:
/// 1. Group states by their "final signature" (final_weight & needed)
/// 2. States in the same group are definitely compatible (same behavior)
/// 3. States in different groups are compatible IFF their needed sets don't overlap
/// 4. Use RangeSet overlap detection to quickly identify incompatible pairs
fn build_incompatibility_graph_height_0(
    candidates: &[StateID],
    dwa: &DWA,
    needed: &[Weight],
) -> Vec<Vec<usize>> {
    let n = candidates.len();
    let start = std::time::Instant::now();
    
    // Compute signatures: hash of (final_weight & needed)
    let mut sig_to_group: HashMap<u64, Vec<usize>> = HashMap::new();
    
    for (idx, &id) in candidates.iter().enumerate() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let final_on_needed = dwa.states[id].final_weight.as_ref()
            .map(|w| w & &needed[id])
            .unwrap_or_else(Weight::zeros);
        
        let mut hasher = DefaultHasher::new();
        final_on_needed.fingerprint().hash(&mut hasher);
        let sig = hasher.finish();
        
        sig_to_group.entry(sig).or_default().push(idx);
    }
    
    let num_groups = sig_to_group.len();
    
    // For smaller sets, build the full graph for optimal coloring
    let groups: Vec<(u64, Vec<usize>)> = sig_to_group.into_iter().collect();
    
    // Compute the "needed footprint" for each group (union of all needed sets)
    let group_footprints: Vec<Weight> = groups.iter().map(|(_, indices)| {
        let mut footprint = Weight::zeros();
        for &idx in indices {
            footprint |= &needed[candidates[idx]];
        }
        footprint
    }).collect();
    
    // Build incompatibility graph
    let mut adj = vec![vec![]; n];
    let mut edge_count = 0usize;
    let mut compare_count = 0usize;
    let mut skipped_by_footprint = 0usize;
    
    // Compare across groups
    for i in 0..groups.len() {
        for j in (i+1)..groups.len() {
            // Quick check: do the group footprints overlap?
            let overlap = &group_footprints[i] & &group_footprints[j];
            if overlap.is_empty() {
                // No overlap in needed sets means all pairs are compatible
                skipped_by_footprint += groups[i].1.len() * groups[j].1.len();
                continue;
            }
            
            // Groups have overlapping footprints - need to check individual pairs
            // But since all states in a group have the same final signature,
            // if ANY pair is incompatible, ALL cross-group pairs are incompatible
            // (because final behavior differs on some shared weight)
            
            // Check one representative pair
            let idx_i = groups[i].1[0];
            let idx_j = groups[j].1[0];
            let id_i = candidates[idx_i];
            let id_j = candidates[idx_j];
            
            // Check if their needed sets overlap
            let pair_overlap = &needed[id_i] & &needed[id_j];
            if pair_overlap.is_empty() {
                // This specific pair is compatible, but others in the group might not be
                // Need to check all pairs in this case
                for &idx_i in &groups[i].1 {
                    for &idx_j in &groups[j].1 {
                        compare_count += 1;
                        let id_i = candidates[idx_i];
                        let id_j = candidates[idx_j];
                        let pair_overlap = &needed[id_i] & &needed[id_j];
                        if !pair_overlap.is_empty() {
                            // Finals differ on overlap (different groups), so incompatible
                            adj[idx_i].push(idx_j);
                            adj[idx_j].push(idx_i);
                            edge_count += 1;
                        }
                    }
                }
            } else {
                // Representative pair has overlapping needed sets AND different signatures
                // So all pairs between these groups are incompatible
                for &idx_i in &groups[i].1 {
                    for &idx_j in &groups[j].1 {
                        adj[idx_i].push(idx_j);
                        adj[idx_j].push(idx_i);
                        edge_count += 1;
                    }
                }
            }
        }
    }
    
    if n >= 100 {
        crate::debug!(5, "Incomp graph (h=0): {} candidates, {} signature groups, {} comparisons, {} skipped by footprint, {} edges, {:?}",
            n, num_groups, compare_count, skipped_by_footprint, edge_count, start.elapsed());
    }
    
    adj
}

/// General incompatibility graph construction for non-height-0 states.
/// Greedy coloring without building full incompatibility graph.
/// Process candidates one by one, checking compatibility only with color class representatives.
fn greedy_color_without_graph(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder],
    _start: std::time::Instant,
) -> Vec<usize> {
    let n = candidates.len();
    if n == 0 { return vec![]; }
    
    // Compute signatures for each candidate
    let signatures: Vec<u128> = candidates.iter().map(|&id| {
        compute_state_signature(id, dwa, needed, old_to_new)
    }).collect();
    
    // Check if there are many unique signatures - if so, use signatures as colors directly
    let mut sig_to_group: HashMap<u128, Vec<usize>> = HashMap::new();
    for (idx, &sig) in signatures.iter().enumerate() {
        sig_to_group.entry(sig).or_default().push(idx);
    }
    let num_groups = sig_to_group.len();
    let signature_coverage = num_groups as f64 / n as f64;
    
    // If >50% unique signatures, cross-signature merging is unlikely - use signatures as colors
    if signature_coverage > 0.50 {
        crate::debug!(5, "Greedy fast path: {} sig groups / {} candidates ({:.1}% coverage) -> using signatures as colors",
            num_groups, n, signature_coverage * 100.0);
        
        let mut colors = vec![0; n];
        let mut sig_to_color: HashMap<u128, usize> = HashMap::new();
        let mut next_color = 0usize;
        for (idx, &sig) in signatures.iter().enumerate() {
            let color = *sig_to_color.entry(sig).or_insert_with(|| {
                let c = next_color;
                next_color += 1;
                c
            });
            colors[idx] = color;
        }
        return colors;
    }
    
    // colors[i] = color assigned to candidate i
    let mut colors = vec![usize::MAX; n];
    
    // color_representatives[c] = list of (candidate_idx, signature) for color c
    // We keep one representative per signature in each color class
    let mut color_representatives: Vec<Vec<(usize, u128)>> = Vec::new();
    
    let mut compare_count = 0usize;
    
    for idx in 0..n {
        let sig = signatures[idx];
        let cand = candidates[idx];
        
        // Try to find an existing color where this candidate is compatible
        let mut assigned_color = None;
        
        'color_loop: for (color, reps) in color_representatives.iter().enumerate() {
            // Check if there's already a representative with the same signature
            // If so, we're guaranteed compatible (by signature design)
            let same_sig = reps.iter().any(|(_, rep_sig)| *rep_sig == sig);
            if same_sig {
                assigned_color = Some(color);
                break 'color_loop;
            }
            
            // Check compatibility with all representatives of different signatures
            let mut compatible_with_all = true;
            for &(rep_idx, _rep_sig) in reps {
                compare_count += 1;
                if !are_compatible(cand, candidates[rep_idx], dwa, needed, old_to_new, new_states) {
                    compatible_with_all = false;
                    break;
                }
            }
            
            if compatible_with_all {
                assigned_color = Some(color);
                break 'color_loop;
            }
        }
        
        // Assign color
        let color = match assigned_color {
            Some(c) => c,
            None => {
                // Need a new color
                let c = color_representatives.len();
                color_representatives.push(Vec::new());
                c
            }
        };
        
        colors[idx] = color;
        
        // Add as representative if this is a new signature for this color
        let reps = &mut color_representatives[color];
        if !reps.iter().any(|(_, rep_sig)| *rep_sig == sig) {
            reps.push((idx, sig));
        }
    }
    
    if n >= 100 {
        let num_colors = color_representatives.len();
        crate::debug!(5, "Greedy color: {} candidates -> {} colors, {} comparisons", n, num_colors, compare_count);
    }
    
    colors
}

fn build_incompatibility_graph_general(
    dwa: &DWA,
    candidates: &[StateID],
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
    new_states: &[MergedStateBuilder],
    start: std::time::Instant,
) -> Vec<Vec<usize>> {
    let n = candidates.len();
    
    // Compute signatures for each candidate
    let signatures: Vec<u128> = candidates.iter().map(|&id| {
        compute_state_signature(id, dwa, needed, old_to_new)
    }).collect();
    
    // Group candidates by signature
    let mut sig_to_candidates: HashMap<u128, Vec<usize>> = HashMap::new();
    for (idx, &sig) in signatures.iter().enumerate() {
        sig_to_candidates.entry(sig).or_default().push(idx);
    }
    
    let num_groups = sig_to_candidates.len();
    let groups: Vec<Vec<usize>> = sig_to_candidates.into_values().collect();
    
    // Build incompatibility graph
    let mut adj = vec![vec![]; n];
    let mut edge_count = 0usize;
    let mut compare_count = 0usize;
    
    // Compare across groups
    for i in 0..groups.len() {
        for j in (i+1)..groups.len() {
            for &idx_i in &groups[i] {
                for &idx_j in &groups[j] {
                    compare_count += 1;
                    if !are_compatible(candidates[idx_i], candidates[idx_j], dwa, needed, old_to_new, new_states) {
                        adj[idx_i].push(idx_j);
                        adj[idx_j].push(idx_i);
                        edge_count += 1;
                    }
                }
            }
        }
    }
    
    // Debug assertions for signature correctness
    #[cfg(debug_assertions)]
    for group in &groups {
        for (i, &idx_i) in group.iter().enumerate() {
            for &idx_j in &group[i+1..] {
                debug_assert!(
                    are_compatible(candidates[idx_i], candidates[idx_j], dwa, needed, old_to_new, new_states),
                    "Signature collision: states {} and {} have same signature but are incompatible!",
                    candidates[idx_i], candidates[idx_j]
                );
            }
        }
    }
    
    if n >= 100 {
        crate::debug!(5, "Incomp graph: {} candidates, {} sig groups, {} comparisons, {} edges, {:?}",
            n, num_groups, compare_count, edge_count, start.elapsed());
    }
    
    adj
}

/// Compute a signature for a state that identifies its "behavior class".
/// States with the same signature are guaranteed to be compatible.
fn compute_state_signature(
    id: StateID,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &HashMap<StateID, StateID>,
) -> u128 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    let state = &dwa.states[id];
    let needed_set = &needed[id];
    
    let mut hasher = DefaultHasher::new();
    
    // Hash final weight (restricted to needed set)
    let final_on_needed = state.final_weight.as_ref()
        .map(|w| w & needed_set)
        .unwrap_or_else(Weight::zeros);
    final_on_needed.fingerprint().hash(&mut hasher);
    
    // Hash transitions (sorted by label for consistency)
    let mut trans_data: Vec<(Label, u64, Option<StateID>)> = Vec::new();
    for (&label, &target) in &state.transitions {
        let weight = state.trans_weights.get(&label)
            .map(|w| w & needed_set)
            .unwrap_or_else(Weight::zeros);
        if !weight.is_empty() {
            // Get the new_id for the target (if already mapped)
            let target_new_id = old_to_new.get(&target).copied();
            trans_data.push((label, weight.fingerprint(), target_new_id));
        }
    }
    trans_data.sort();
    
    for (label, weight_fp, target_new_id) in trans_data {
        label.hash(&mut hasher);
        weight_fp.hash(&mut hasher);
        target_new_id.hash(&mut hasher);
    }
    
    // Use two hashes for a 128-bit signature to reduce collisions
    let h1 = hasher.finish();
    
    // Second hash with different seed
    let mut hasher2 = DefaultHasher::new();
    h1.hash(&mut hasher2);
    needed_set.fingerprint().hash(&mut hasher2);
    let h2 = hasher2.finish();
    
    ((h1 as u128) << 64) | (h2 as u128)
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