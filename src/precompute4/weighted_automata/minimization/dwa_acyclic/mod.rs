use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{BTreeMap, HashMap, VecDeque};

impl DWA {
    /// Minimizes an acyclic DWA.
    ///
    /// This algorithm is stronger than standard DFA minimization. It performs "Need" analysis
    /// (liveness analysis) to determine which tokens are relevant at each state.
    ///
    /// It then merges states `u` and `v` if they are **compatible**:
    /// i.e., for every token `t` that is "live" in both `u` and `v`, `u` and `v` behave identically.
    ///
    /// This allows merging the "Diamond" structure:
    /// - A accepts {0}, B accepts {1}.
    /// - They are strictly different.
    /// - But if A is only entered with {0,2} and B with {1,2}, their "overlap" is {2}.
    /// - If they behave the same on {2}, they can be merged into AB (accepting {0,1}),
    ///   relying on incoming edges to filter {0} vs {1}.
    pub fn minimize_acyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        // 1. Compute Topological Sort (and check for cycles)
        let topo_order = match self.get_topological_sort() {
            Some(order) => order,
            None => {
                eprintln!("Warning: DWA::minimize_acyclic called on cyclic graph. Skipping.");
                return;
            }
        };

        // 2. Compute "Need" (Liveness)
        // Need[s] = Union of all path weights from s to any acceptance.
        // This tells us which tokens *matter* at state s.
        let mut need = vec![Weight::zeros(); self.states.len()];

        // Iterate reverse topological order
        for &u in topo_order.iter().rev() {
            let state = &self.states[u];
            let mut u_need = Weight::zeros();

            // 2a. Contribution from Final Weight
            if let Some(fw) = &state.final_weight {
                u_need |= fw;
            }

            // 2b. Contribution from Transitions
            for (label, &target) in &state.transitions {
                if target < self.states.len() {
                    if let Some(w) = state.trans_weights.get(label) {
                        // The tokens needed by target, masked by the transition weight
                        let mut flow = need[target].clone();
                        flow &= w;
                        u_need |= &flow;
                    }
                }
            }

            // 2c. Mask by State Weight
            if let Some(sw) = &state.state_weight {
                u_need &= sw;
            }

            need[u] = u_need;
        }

        // If start state needs nothing, the whole machine is empty
        if need[self.body.start_state].is_empty() {
            *self = DWA::new();
            return;
        }

        // 3. Build Minimized Automaton Bottom-Up
        // We will construct new states. We map old_id -> new_id.
        let mut old_to_new = vec![None; self.states.len()];
        let mut new_states = DWAStates::default();

        // Store indices of new states to check for compatibility
        // We scan linearly or use a helper. Since we need "Compatibility" (not equality),
        // we cannot use a simple HashMap signature. We use a list of unique states.
        // Optimization: In a large graph, we would bucket by (finalness, transition_keys).
        let mut unique_indices: Vec<StateID> = Vec::new();

        // Needs for the *new* states (accumulated during merges)
        let mut new_state_needs: Vec<Weight> = Vec::new();

        for &u in topo_order.iter().rev() {
            // If the state is dead (Need is empty), map it to a dummy or skip
            // We just skip; incoming transitions to it will be filtered by Need logic anyway.
            if need[u].is_empty() {
                continue;
            }

            // Construct the "Proposed" Candidate State
            // Note: Targets must already be mapped because we are in reverse topo order.
            let mut candidate = DWAState::default();

            // Normalize State Weight:
            // In the merged logic, we usually push state_weight into incoming edges or final.
            // But to preserve semantics, we keep it, but we can union it during merges.
            candidate.state_weight = self.states[u].state_weight.clone();

            // Normalize Final Weight:
            // We only care about the intersection with Need.
            if let Some(fw) = &self.states[u].final_weight {
                let mut eff = fw.clone();
                eff &= &need[u];
                if !eff.is_empty() {
                    candidate.final_weight = Some(eff);
                }
            }

            // Normalize Transitions
            for (lbl, &old_tgt) in &self.states[u].transitions {
                // If target is dead, ignore transition
                if need[old_tgt].is_empty() { continue; }

                // If target is alive, it must have been processed
                if let Some(new_tgt) = old_to_new[old_tgt] {
                    let w_original = self.states[u].trans_weights.get(lbl).unwrap();

                    // CRITICAL: The weight on the transition is trimmed by the
                    // Need of the ORIGINAL target. This preserves the path constraints
                    // (e.g. "Start->A" allows {0}, "Start->B" allows {1}).
                    let mut w_eff = w_original.clone();
                    w_eff &= &need[old_tgt];

                    if !w_eff.is_empty() {
                        candidate.transitions.insert(*lbl, new_tgt);
                        candidate.trans_weights.insert(*lbl, w_eff);
                    }
                }
            }

            // Try to merge with an existing compatible state
            let mut merged_id = None;

            for &existing_id in &unique_indices {
                // Check compatibility between `candidate` and `new_states[existing_id]`
                // Context: They must agree on the intersection of `need[u]` and `new_state_needs[existing_id]`.

                let existing = &new_states[existing_id];
                let existing_need = &new_state_needs[existing_id];

                // Calculate intersection of domains
                let mut common = need[u].clone();
                common &= existing_need;

                if is_compatible(&candidate, existing, &common) {
                    merged_id = Some(existing_id);
                    break;
                }
            }

            if let Some(id) = merged_id {
                // MERGE
                old_to_new[u] = Some(id);

                // Update the existing state by UNIONING the behaviors
                // This creates the "super-state" (e.g., AB accepts {0} U {1})
                merge_states(&mut new_states[id], &candidate);

                // Update need
                new_state_needs[id] |= &need[u];
            } else {
                // CREATE NEW
                let new_id = new_states.add_existing_state(candidate);
                old_to_new[u] = Some(new_id);
                unique_indices.push(new_id);
                new_state_needs.push(need[u].clone());
            }
        }

        // 4. Update Self
        self.states = new_states;
        // Start state might have been mapped. If it was dead, we defaulted to empty earlier.
        if let Some(new_start) = old_to_new[self.body.start_state] {
            self.body.start_state = new_start;
        } else {
            // Start state was unreachable or dead
            *self = DWA::new();
        }
    }

    fn get_topological_sort(&self) -> Option<Vec<usize>> {
        let n = self.states.len();
        let mut in_degree = vec![0; n];
        for state in self.states.iter() {
            for &to in state.transitions.values() {
                if to < n { in_degree[to] += 1; }
            }
        }

        let mut queue = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 { queue.push_back(i); }
        }

        let mut order = Vec::new();
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &to in self.states[u].transitions.values() {
                if to < n {
                    in_degree[to] -= 1;
                    if in_degree[to] == 0 {
                        queue.push_back(to);
                    }
                }
            }
        }

        if order.len() == n { Some(order) } else { None }
    }
}

/// Checks if two states behave identically for all tokens in `mask`.
fn is_compatible(a: &DWAState, b: &DWAState, mask: &Weight) -> bool {
    if mask.is_empty() {
        return true;
    }

    // 1. Check State Weight
    // (Optional strictness: strictly, state_weight is applied on entry.
    // If we merge, the new state_weight is Union.
    // We must ensure that for tokens in Common, the restriction is identical.)
    let empty = Weight::zeros();
    let sw_a = a.state_weight.as_ref().unwrap_or(&Weight::all()); // Treat None as All for comparison?
    // Actually, in DWAState logic, None is "All".
    // BUT we normalized candidate above. However, safe to check logic:
    // a.sw & mask == b.sw & mask
    // Since we are merging, specific implementation of Weight equality is needed.
    // Let's assume strict check on effective weights.

    // Helper for effective weight comparison
    let check_weight_eq = |w1: Option<&Weight>, w2: Option<&Weight>, m: &Weight| -> bool {
        let mut v1 = w1.cloned().unwrap_or_else(Weight::all);
        v1 &= m;
        let mut v2 = w2.cloned().unwrap_or_else(Weight::all);
        v2 &= m;
        v1 == v2
    };

    if !check_weight_eq(a.state_weight.as_ref(), b.state_weight.as_ref(), mask) {
        return false;
    }

    // 2. Check Final Weight
    // a.fw & mask == b.fw & mask
    // Note: DWAState default for final_weight is None (which means 0/Empty).
    // Wait, struct says `final_weight: Option<Weight>`. Usually None implies not final (empty).
    // Let's verify `DWAState` def: "if fw.is_empty() { self.final_weight = None; }"
    // So None == Empty.
    let mut fw_a = a.final_weight.as_ref().map(|w| w.clone()).unwrap_or_else(|| empty.clone()); fw_a &= mask;
    let mut fw_b = b.final_weight.as_ref().map(|w| w.clone()).unwrap_or_else(|| empty.clone()); fw_b &= mask;
    if fw_a != fw_b {
        return false;
    }

    // 3. Check Transitions
    // For every label, targets must be same, and weights (under mask) must be same.
    // Collect all labels
    let mut labels: Vec<&Label> = a.transitions.keys().collect();
    labels.extend(b.transitions.keys());
    labels.sort();
    labels.dedup();

    for &lbl in labels {
        let t_a = a.transitions.get(&lbl);
        let t_b = b.transitions.get(&lbl);

        // Effective weights
        let mut w_a = if t_a.is_some() { a.trans_weights.get(&lbl).cloned().unwrap_or_else(Weight::all) } else { Weight::zeros() };
        w_a &= mask;

        let mut w_b = if t_b.is_some() { b.trans_weights.get(&lbl).cloned().unwrap_or_else(Weight::all) } else { Weight::zeros() };
        w_b &= mask;

        // If both are effectively zero under the mask, they are compatible (ignoring target mismatch)
        if w_a.is_empty() && w_b.is_empty() {
            continue;
        }

        // If one exists and other doesn't (and weight is non-zero), incompatible
        // If both exist:
        // 1. Weights must match
        if w_a != w_b {
            return false;
        }
        // 2. Targets must match
        // Since we are building bottom-up, targets are NewStateIDs.
        if t_a != t_b {
            return false;
        }
    }

    true
}

/// Merges `src` into `dest` (in-place union).
fn merge_states(dest: &mut DWAState, src: &DWAState) {
    // Union State Weight
    // (None is All. If one is All, result is All (None). Else Union).
    // Actually, careful:
    // DWAState def: None in state_weight means ??? Usually ALL?
    // Let's look at `apply_weight`: `sw &= weight`.
    // If sw is None, it acts like ALL.
    // Union(All, X) = All.
    if dest.state_weight.is_none() || src.state_weight.is_none() {
        dest.state_weight = None;
    } else {
        // Both are Some. Union them.
        let mut combined = dest.state_weight.clone().unwrap();
        combined |= src.state_weight.as_ref().unwrap();
        if combined.is_all_fast() { // Optimization check
            dest.state_weight = None;
        } else {
            dest.state_weight = Some(combined);
        }
    }

    // Union Final Weight
    // (None is Empty).
    if let Some(fw_src) = &src.final_weight {
        if let Some(fw_dest) = &mut dest.final_weight {
            *fw_dest |= fw_src;
        } else {
            dest.final_weight = Some(fw_src.clone());
        }
    }

    // Union Transitions
    for (lbl, &tgt) in &src.transitions {
        let w_src = src.trans_weights.get(lbl).unwrap();

        if dest.transitions.contains_key(lbl) {
            // Target matches (checked by is_compatible), just union weights
            let w_dest = dest.trans_weights.get_mut(lbl).unwrap();
            *w_dest |= w_src;
        } else {
            dest.transitions.insert(*lbl, tgt);
            dest.trans_weights.insert(*lbl, w_src.clone());
        }
    }
}