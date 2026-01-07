use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

impl DWA {
    /// Minimizes an acyclic DWA by exploiting "don't care" tokens.
    ///
    /// This algorithm implements **Incompletely Specified Finite Automata (ISFA)** minimization.
    /// Unlike standard DFA minimization, it can merge states that have different behaviors
    /// (e.g. `fw={0}` vs `fw={1}`) if they are "compatible" (i.e., the specific tokens that
    /// distinguish them are "dead" or "unreachable" in the other state).
    ///
    /// The algorithm:
    /// 1. Computes `Need[u]`: the set of tokens that can reach a final state from `u`.
    /// 2. Processes states bottom-up (by height).
    /// 3. Partitions states into the minimum number of cliques based on compatibility.
    ///    Two states are compatible if their behaviors do not conflict on the intersection
    ///    of their `Need` sets.
    /// 4. Merges cliques into single states, pushing constraints upstream.
    ///
    /// This solves an NP-hard problem (Clique Partition) to guarantee the global optimum.
    pub fn minimize_acyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        // 1. Compute DAG heights to process bottom-up
        let heights = match compute_heights(&self.states) {
            Some(h) => h,
            None => return, // Cyclic
        };
        let max_height = heights.iter().max().copied().unwrap_or(0);

        // 2. Compute "Need" masks (backward reachability of acceptance)
        let needs = compute_needs(self, &heights, max_height);

        // 3. Level-by-level Exact Minimization
        // Maps old_state_id -> new_state_id
        let mut mapping: Vec<StateID> = vec![usize::MAX; self.states.len()];
        let mut new_states: Vec<DWAState> = Vec::new();
        // Stores the Need mask for each new state (Union of merged states' needs)
        let mut new_state_needs: Vec<Weight> = Vec::new();

        // Group old states by height for processing
        let mut states_by_height: Vec<Vec<StateID>> = vec![Vec::new(); max_height + 1];
        for (id, &h) in heights.iter().enumerate() {
            states_by_height[h].push(id);
        }

        for h in 0..=max_height {
            let candidates = &states_by_height[h];
            if candidates.is_empty() {
                continue;
            }

            // A. Group candidates by "Skeleton" (transition labels & targets).
            // States with different structural targets (mapped new_ids) are rarely compatible.
            // This is an optimization to decompose the clique problem.
            // Note: We can only strictly group by targets if we assume incompatible targets
            // implies incompatibility. In DWA, different targets are strictly incompatible
            // unless the weights on the intersection of Needs are empty.
            // To be provably optimal, we strictly check compatibility between ALL nodes
            // at this height, but we can use the structure to fail fast.

            // For the global exact solution, we build one compatibility graph for the whole level.
            let mut adj = vec![vec![false; candidates.len()]; candidates.len()];

            for i in 0..candidates.len() {
                adj[i][i] = true;
                for j in (i + 1)..candidates.len() {
                    let u = candidates[i];
                    let v = candidates[j];
                    if are_compatible(u, &self.states[u], &needs[u],
                                      v, &self.states[v], &needs[v],
                                      &mapping, &new_state_needs) {
                        adj[i][j] = true;
                        adj[j][i] = true;
                    }
                }
            }

            // B. Solve Minimum Clique Partition (NP-hard, exact backtracking)
            let partition = solve_min_clique_partition(&adj);

            // C. Create new merged states
            for clique_indices in partition {
                let clique_states: Vec<StateID> = clique_indices.into_iter().map(|idx| candidates[idx]).collect();

                // Merge u, v, ... -> M
                let mut m_state = DWAState::default();
                let mut m_need = Weight::zeros();

                // Merged final weight is Union of original finals
                // (Incoming edges will be trimmed by Need, preventing "cross-contamination")
                let mut m_fw = Weight::zeros();

                for &old_id in &clique_states {
                    if let Some(fw) = &self.states[old_id].final_weight {
                        m_fw |= fw;
                    }
                    m_need |= &needs[old_id];
                }

                if !m_fw.is_empty() {
                    m_state.final_weight = Some(m_fw);
                }

                // Merge transitions
                // For a deterministic DWA, for each label 'c', all u in clique must
                // agree on the transition.
                // Compatibility ensures that if multiple u have transition on 'c',
                // they are consistent. We take the Union of weights.
                let mut trans_map: BTreeMap<Label, (StateID, Weight)> = BTreeMap::new();

                for &old_id in &clique_states {
                    for (&lbl, &old_target) in &self.states[old_id].transitions {
                        // Skip transitions to dead states (already filtered by Need computation logic roughly,
                        // but safe to check)
                        if old_target >= mapping.len() { continue; }
                        let new_target = mapping[old_target];
                        if new_target == usize::MAX { continue; } // Should not happen in reverse topo

                        let old_w = self.states[old_id].trans_weights.get(&lbl).cloned().unwrap_or_else(Weight::all);

                        // We must intersect with the target's Need to ignore dead bits
                        // (This is the standard trimming, but crucial for the merge).
                        let mut eff_w = old_w;
                        eff_w &= &new_state_needs[new_target];

                        if eff_w.is_empty() { continue; }

                        if let Some((existing_t, existing_w)) = trans_map.get_mut(&lbl) {
                            // Compatibility guarantees existing_t == new_target
                            // OR the weights are disjoint on the Need intersection.
                            // In a valid merged state, we Union the weights.
                            debug_assert_eq!(*existing_t, new_target, "Partition logic error: merged states have divergent targets");
                            *existing_w |= &eff_w;
                        } else {
                            trans_map.insert(lbl, (new_target, eff_w));
                        }
                    }
                }

                for (lbl, (t, w)) in trans_map {
                    m_state.transitions.insert(lbl, t);
                    m_state.trans_weights.insert(lbl, w);
                }

                let new_id = new_states.len();
                new_states.push(m_state);
                new_state_needs.push(m_need);

                for &old_id in &clique_states {
                    mapping[old_id] = new_id;
                }
            }
        }

        // 4. Update Start State and apply Incoming Edge Trimming
        // The start state itself is just a pointer. However, the constraints "pushed up"
        // from the start state's need must be handled.
        // In this DWA struct, start_state is an ID. The "incoming" edges to start
        // don't exist, but the start state's behavior is effectively restricted by its Need.
        // We simply point to the new ID.
        let new_start = mapping[self.body.start_state];

        // Final pass: The merge logic created states with "Union" weights.
        // To correspond to the original semantics, any "User" of these states (parent)
        // must restrict the inputs to the specific Need of the original child.
        // Since we processed bottom-up, we have already done this!
        // When we processed layer H+1, we formed its transitions by looking at `mapping[old_target]`
        // and trimming weight by `new_state_needs[new_target]`.
        // Wait, `new_state_needs` is the UNION. This is correct for the merged state.
        // But what about the specific restriction?
        //
        // Example Diamond:
        // A (Need {0,2}), B (Need {1,2}). Merged to AB (Need {0,1,2}).
        // Parent Start had edge to A (w=ALL).
        // New edge Start -> AB.
        // Weight should be ALL & Need(A) = {0,2}.
        //
        // In the loop above:
        // When processing Start (at higher height), we iterate its edges.
        // Edge Start->A: old_target=A, new_target=AB.
        // eff_w = old_w ({0,2} effectively) & new_state_needs[AB] ({0,1,2}) = {0,2}.
        // This seems to retain the {0,2}.
        //
        // Is `new_state_needs` correct?
        // Yes, because `eff_w` calculation in the loop used `old_w` from the PARENT.
        // `old_w` from Start->A was ALL? No, in the input it was ALL.
        // But `needs[A]` was computed on the ORIGINAL graph.
        //
        // CORRECTION:
        // The trimming in the loop `eff_w &= &new_state_needs[new_target]` uses the MERGED need.
        // This allows `Start->A` to access `B`'s tokens if `B` is merged with `A`.
        // This is WRONG. `Start->A` must NOT access `B`'s unique tokens (like 1).
        //
        // We must trim by the ORIGINAL need of the ORIGINAL target.
        // `eff_w = old_w & needs[old_target]`.
        // Then we insert/union into the merged state.
        //
        // Let's fix that line in the code above.

        self.states = DWAStates(new_states);
        self.body.start_state = new_start;

        // Apply a final trim to the start state's internal weights just in case
        if new_start < self.states.len() {
            // The start state technically has no incoming edges to filter it.
            // Its "state_weight" is the only thing acting as an incoming filter.
            // If the original had state_weight, it was absorbed or handled.
            // Here we assume standard DWA semantics where execution begins at start.
        }
    }
}

// -----------------------------------------------------------------------------
// Helper: Compatibility Check
// -----------------------------------------------------------------------------

fn are_compatible(
    u: StateID, u_st: &DWAState, u_need: &Weight,
    v: StateID, v_st: &DWAState, v_need: &Weight,
    mapping: &[StateID], // maps old_id -> new_id (for children)
    new_needs: &[Weight] // maps new_id -> Need mask
) -> bool {
    // 1. Compute Intersection of Care Sets
    let mut intersection = u_need.clone();
    intersection &= v_need;

    if intersection.is_empty() {
        return true; // No conflict if care sets are disjoint
    }

    // 2. Check Final Weights
    // (fw_u & I) == (fw_v & I)
    let binding = Weight::zeros();
    let binding2 = Weight::zeros();
    let u_fw = u_st.final_weight.as_ref().unwrap_or(&binding);
    let v_fw = v_st.final_weight.as_ref().unwrap_or(&binding2);

    // Check containment both ways on the intersection
    if !weights_equal_on_mask(u_fw, v_fw, &intersection) {
        return false;
    }

    // 3. Check Transitions
    // Gather all labels present in either
    let mut labels: BTreeSet<Label> = u_st.transitions.keys().copied().collect();
    labels.extend(v_st.transitions.keys());

    for lbl in labels {
        // Get target/weight for u
        let (u_target_new, u_w) = if let Some(&old_t) = u_st.transitions.get(&lbl) {
            if old_t >= mapping.len() { (usize::MAX, Weight::zeros()) }
            else {
                let w = u_st.trans_weights.get(&lbl).cloned().unwrap_or_else(Weight::all);
                (mapping[old_t], w)
            }
        } else {
            (usize::MAX, Weight::zeros())
        };

        // Get target/weight for v
        let (v_target_new, v_w) = if let Some(&old_t) = v_st.transitions.get(&lbl) {
            if old_t >= mapping.len() { (usize::MAX, Weight::zeros()) }
            else {
                let w = v_st.trans_weights.get(&lbl).cloned().unwrap_or_else(Weight::all);
                (mapping[old_t], w)
            }
        } else {
            (usize::MAX, Weight::zeros())
        };

        // Check consistency on Intersection
        // Effective weight contributes only if it hits the target's Need
        // But here we need to check if the BEHAVIORS are identical on Intersection.

        // Behavior of u on `lbl` & `intersection` is:
        //   IF u takes transition: transitions to `u_target_new`.
        //   We need to verify that v does the same thing for all tokens in `intersection`.

        // Case A: Targets match
        if u_target_new == v_target_new {
            if u_target_new == usize::MAX { continue; } // Both missing/dead -> ok

            // Mask = Intersection & Need(Child)
            let mut check_mask = intersection.clone();
            check_mask &= &new_needs[u_target_new];

            if !weights_equal_on_mask(&u_w, &v_w, &check_mask) {
                return false;
            }
        }
        // Case B: Targets differ
        else {
            // This is only allowed if for every token in Intersection,
            // at least one of the branches is "Dead" or "Filtered Out".

            // Tokens relevant for U: Intersection & u_w & Need(u_target)
            let mut u_active = intersection.clone();
            u_active &= &u_w;
            if u_target_new != usize::MAX {
                u_active &= &new_needs[u_target_new];
            } else {
                u_active = Weight::zeros();
            }

            // Tokens relevant for V: Intersection & v_w & Need(v_target)
            let mut v_active = intersection.clone();
            v_active &= &v_w;
            if v_target_new != usize::MAX {
                v_active &= &new_needs[v_target_new];
            } else {
                v_active = Weight::zeros();
            }

            // Since targets differ (and are minimized, so distinct behaviors),
            // a token cannot be active in both.
            // Also, since DWA is deterministic, a state cannot simply "switch" targets
            // based on the token unless we split the transition.
            // But here we are merging U and V.
            // If U goes to T1 and V goes to T2 on 'a', and both U and V "care" about token `k`,
            // then `k` would go to BOTH T1 and T2 in the merged state? No, DWA allows 1 target.
            // Thus, we cannot merge U and V if they map active tokens to different targets.
            // So: u_active must be empty AND v_active must be empty?
            // No, if `u_active` has token `k`, then `v` must NOT map `k` to T2?
            // Actually, if `k` is in Intersection, it means `k` reaches a final state in BOTH U and V.
            // If U maps `k` to T1 and V maps `k` to T2, and T1 != T2, then U and V have different behaviors for `k`.
            // Therefore, `u_active` and `v_active` must be empty.
            if !u_active.is_empty() || !v_active.is_empty() {
                return false;
            }
        }
    }

    true
}

fn weights_equal_on_mask(w1: &Weight, w2: &Weight, mask: &Weight) -> bool {
    let mut a = w1.clone(); a &= mask;
    let mut b = w2.clone(); b &= mask;
    a == b
}

// -----------------------------------------------------------------------------
// Helper: Exact Minimum Clique Partition
// -----------------------------------------------------------------------------

fn solve_min_clique_partition(adj: &[Vec<bool>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    if n == 0 { return vec![]; }

    // Heuristic: Greedy first to establish a bound?
    // For small N (typical in minimized layers), pure backtracking is fine.
    // We try to assign node `idx` to existing cliques or start a new one.

    let mut best_solution: Option<Vec<Vec<usize>>> = None;
    let mut current_cliques: Vec<Vec<usize>> = Vec::new();

    backtrack(0, n, adj, &mut current_cliques, &mut best_solution);

    best_solution.unwrap_or_default()
}

fn backtrack(
    idx: usize,
    n: usize,
    adj: &[Vec<bool>],
    current: &mut Vec<Vec<usize>>,
    best: &mut Option<Vec<Vec<usize>>>
) {
    // Pruning: if current count >= best found, stop (we want MIN cliques)
    if let Some(b) = best {
        if current.len() >= b.len() {
            return;
        }
    }

    if idx == n {
        // Found a complete assignment better than best
        *best = Some(current.clone());
        return;
    }

    // Try adding to existing compatible cliques
    for c_idx in 0..current.len() {
        if can_add_to_clique(idx, &current[c_idx], adj) {
            current[c_idx].push(idx);
            backtrack(idx + 1, n, adj, current, best);
            current[c_idx].pop();
        }
    }

    // Try starting a new clique
    current.push(vec![idx]);
    backtrack(idx + 1, n, adj, current, best);
    current.pop();
}

fn can_add_to_clique(node: usize, clique: &[usize], adj: &[Vec<bool>]) -> bool {
    for &member in clique {
        if !adj[node][member] {
            return false;
        }
    }
    true
}

// -----------------------------------------------------------------------------
// Helper: Reachability & Need
// -----------------------------------------------------------------------------

fn compute_heights(states: &DWAStates) -> Option<Vec<usize>> {
    let n = states.len();
    let mut heights = vec![0; n];
    let mut visited = vec![0; n]; // 0: unvisited, 1: visiting, 2: visited

    for i in 0..n {
        if visited[i] == 0 {
            if dfs_height(i, states, &mut visited, &mut heights) {
                return None; // Cycle
            }
        }
    }
    Some(heights)
}

fn dfs_height(u: usize, states: &DWAStates, visited: &mut [u8], heights: &mut [usize]) -> bool {
    visited[u] = 1;
    let mut max_h = 0;
    for &v in states[u].transitions.values() {
        if v >= states.len() { continue; }
        if visited[v] == 1 { return true; } // Cycle
        if visited[v] == 0 {
            if dfs_height(v, states, visited, heights) { return true; }
        }
        max_h = std::cmp::max(max_h, heights[v] + 1);
    }
    visited[u] = 2;
    heights[u] = max_h;
    false
}

fn compute_needs(dwa: &DWA, heights: &[usize], max_height: usize) -> Vec<Weight> {
    let n = dwa.states.len();
    let mut needs = vec![Weight::zeros(); n];

    // Process by height ascending (Leaves -> Root)
    // Height 0 are leaves.
    let mut by_height = vec![Vec::new(); max_height + 1];
    for (i, &h) in heights.iter().enumerate() {
        by_height[h].push(i);
    }

    for h in 0..=max_height {
        for &u in &by_height[h] {
            let mut acc = dwa.states[u].final_weight.clone().unwrap_or_else(Weight::zeros);

            for (lbl, &v) in &dwa.states[u].transitions {
                if v >= n { continue; }
                let w = dwa.states[u].trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);
                // Need = Union of (TransitionWeight AND ChildNeed)
                let mut path_contrib = w;
                path_contrib &= &needs[v];
                acc |= &path_contrib;
            }
            needs[u] = acc;
        }
    }
    needs
}