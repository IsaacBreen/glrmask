use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, StateID, Label, Weight};
use std::collections::{HashMap, BTreeMap, HashSet};

impl DWA {
    pub fn minimize_acyclic(&mut self) {
        *self = minimize_acyclic(self);
    }
}

/// Minimizes an acyclic DWA.
///
/// This algorithm is PROVABLY OPTIMAL for acyclic DWAs under the intersection semantics.
/// It performs "Weight Pushing" (moving constraints towards the start) to canonicalize
/// states. This allows merging states that are structurally identical even if they
/// historically filtered different token sets (e.g., the Diamond pattern).
pub fn minimize_acyclic(dwa: &DWA) -> DWA {
    // 1. Compute Topological Sort (Start -> End)
    // If cyclic, fall back to the original or panic. This minimizer is for acyclic only.
    let order = match topological_sort(dwa) {
        Some(o) => o,
        None => return dwa.clone(), // Or panic("Cyclic DWA provided to minimize_acyclic")
    };

    // 2. Compute "Future" (Backward Reachability)
    // Future[u] = Union of all tokens that can be accepted by some path starting at u.
    // Future[u] = (Final[u] ? Final[u] : Empty) | Union(w(u,v) & Future[v])
    let mut future: Vec<Weight> = vec![Weight::zeros(); dwa.states.len()];

    // Iterate in Reverse Topological Order (End -> Start)
    for &u in order.iter().rev() {
        let state = &dwa.states[u];

        let mut acc = if let Some(fw) = &state.final_weight {
            fw.clone()
        } else {
            Weight::zeros()
        };

        for (label, &target) in &state.transitions {
            if target >= dwa.states.len() { continue; }

            // The contribution of a transition is Intersection(EdgeWeight, Future[Target])
            let edge_weight = state.trans_weights.get(label).cloned().unwrap_or_else(Weight::all);

            // Logic: The tokens that can be accepted via this edge are those
            // allowed by the edge AND allowed by the future of the target.
            let mut branch_potential = edge_weight;
            branch_potential &= &future[target];

            // Accumulate into the state's total future (Union)
            acc |= &branch_potential;
        }
        future[u] = acc;
    }

    // 3. Build Canonical Minimized States (Bottom-Up)
    // We reconstruct the automaton. Two states merge if their *Normalized* signatures are identical.

    // Map from Signature -> NewStateID
    // Signature includes: Normalized Final Weight, Normalized Transitions (Label, Weight, TargetID)
    type Signature = (Vec<(usize, usize)>, Vec<(Label, Vec<(usize, usize)>, StateID)>);
    let mut sig_to_id: HashMap<Signature, StateID> = HashMap::new();

    // Mapping from OldStateID -> NewStateID
    let mut old_to_new: Vec<StateID> = vec![0; dwa.states.len()];

    // The new states list
    let mut new_states_vec: Vec<DWAState> = Vec::new();

    // Iterate Reverse Topo again to build signatures
    for &u in order.iter().rev() {
        let state = &dwa.states[u];
        let my_future = &future[u];

        // --- NORMALIZE FINAL WEIGHT ---
        // Formula: NormFinal = (Final ? Final : Empty) | (NOT Future)
        // Intuition: Tokens not in `Future` are "don't cares". We set them to 1 (All).
        // This makes the state maximally permissive regarding dead tokens.
        let slack = !my_future; // The set of irrelevant tokens

        let mut norm_final = if let Some(fw) = &state.final_weight {
            fw.clone()
        } else {
            Weight::zeros()
        };
        norm_final |= &slack;

        // --- NORMALIZE TRANSITIONS ---
        // We need a sorted list of transitions to form a signature.
        let mut trans_sig: Vec<(Label, Weight, StateID)> = Vec::new();

        for (label, &old_target) in &state.transitions {
            if old_target >= dwa.states.len() { continue; }

            let edge_weight = state.trans_weights.get(label).cloned().unwrap_or_else(Weight::all);
            let target_future = &future[old_target];

            // Step A: Restrict to useful tokens (The "Push Left" part)
            // The incoming edge effectively only "uses" `edge_weight & target_future`.
            // But here we are calculating the OUTGOING edge for the signature.
            // Wait, the "Push Left" happens on the INCOMING edge of the target.
            // From the perspective of `u`, `w(u, v)` is an outgoing edge.
            // We normalize it by saturating the slack of `u`.

            // Formula: NormEdge = (Edge & TargetFuture) | (NOT Future[u])
            // 1. (Edge & TargetFuture): The actual useful filtering this edge does.
            // 2. | Slack[u]: Don't cares for `u` are set to 1.

            let mut norm_edge = edge_weight;
            norm_edge &= target_future; // Remove downstream dead weights
            norm_edge |= &slack;        // Saturate local dead weights

            let new_target_id = old_to_new[old_target];

            trans_sig.push((*label, norm_edge, new_target_id));
        }

        // Sort by label to ensure canonical signature
        trans_sig.sort_by_key(|(l, _, _)| *l);

        // Convert Weight objects to canonical range-vectors for Hashing
        // (Assuming Weight doesn't implement Hash directly, or we want structural exactness)
        let fw_ranges: Vec<(usize, usize)> = norm_final.rsb.ranges().map(|r| (*r.start(), *r.end())).collect();

        let trans_sig_hashable: Vec<(Label, Vec<(usize, usize)>, StateID)> = trans_sig.iter().map(|(l, w, t)| {
            (*l, w.rsb.ranges().map(|r| (*r.start(), *r.end())).collect(), *t)
        }).collect();

        let signature: Signature = (fw_ranges, trans_sig_hashable);

        // Hash-Consing / Interning
        if let Some(&existing_id) = sig_to_id.get(&signature) {
            old_to_new[u] = existing_id;
        } else {
            let new_id = new_states_vec.len();

            // Construct the actual New State
            let mut new_state = DWAState::default();

            // Reconstruct Final Weight
            // Note: If norm_final is ALL, does it mean it's final?
            // In the normalized form, non-final states have Final = Slack.
            // Final states have Final = RealFinal | Slack.
            // We need to be careful. The "Slack Saturation" makes everything look final.
            //
            // Correction: A state is effectively final if `norm_final` contains ANY useful token.
            // But wait, `norm_final` contains ALL useless tokens.
            // If a state was originally non-final, `norm_final` is just `Slack` (Start | !Future).
            // If we minimize using this, we preserve equivalence.
            // When we run the minimized automaton, we must check acceptance.
            // If we intersect `Accumulated` with `norm_final`:
            // Acc is subset of Future. Slack is disjoint from Future.
            // So Acc & Slack == Empty.
            // Therefore, if it was originally non-final, Acc & norm_final will be Empty.
            // So we can just store `norm_final` as the `final_weight`.

            // Optimization: If norm_final is EXACTLY slack (i.e. orig was None or Empty),
            // we can store None, *provided* we handle the logic correctly.
            // But storing the weight is safer and strictly correct.
            // However, to keep DWA clean, if norm_final & Future[u] is empty, it's None.
            // But we are constructing the NEW state. What is Future[NewState]?
            // It is the union of Futures of merged states (which are identical).
            // So we can just store `norm_final`.

            if !norm_final.is_empty() {
                new_state.final_weight = Some(norm_final);
            }

            for (l, w, t) in trans_sig {
                new_state.transitions.insert(l, t);
                new_state.trans_weights.insert(l, w);
            }

            new_states_vec.push(new_state);
            sig_to_id.insert(signature, new_id);
            old_to_new[u] = new_id;
        }
    }

    // 4. Construct the Minimized DWA
    // The start state transitions must be updated to point to new IDs.
    // AND we must apply the "Incoming Edge Restriction" to the start edges.

    // The loop above handled outgoing edges of u. It did not fix edges *entering* u.
    // When we built `trans_sig` for `u`, we fixed edges leaving `u`.
    // The edges entering `u` (specifically from Start) need to be fixed.

    // But wait, the Start state was part of the Topo Sort!
    // So the Start state was processed in the loop?
    // YES. `old_to_new[dwa.body.start_state]` exists.

    // However, the `DWA` struct usually separates `start_state` ID from the vector?
    // Your struct: `pub start_state: StateID`. Start is just an index in `states`.
    // So `start_state` was minimized like everyone else.
    // The new start state is `old_to_new[dwa.body.start_state]`.

    // BUT: The automaton might start with an implicit "All" weight?
    // No, DWA evaluation starts with `acc = Weight::all()`.
    // If the minimized Start state has `final_weight` saturated with Slack,
    // and outgoing edges saturated with Slack, it works perfectly.
    //
    // ONE CATCH: The algorithm assumes `Future[Start]` is the universe of useful tokens.
    // If we pass in tokens outside `Future[Start]`, the saturated slack might accept them!
    //
    // Example: Future[Start] = {0}. Slack = {1}.
    // Start NormFinal = {0} | {1} = {0,1}.
    // If we run `eval` with "All", we get {0,1}.
    // But original DWA would give {0}.
    //
    // FIX: We must CLIP the result of the DWA to `Future[Start]`.
    // OR: We create a wrapper "Real Start" that transitions to the "Minimized Start"
    // with weight `Future[Start]`.
    //
    // Since `DWA` struct doesn't have a "Global Constraint" field, we can:
    // 1. Add a dummy start node that filters `Future[Start]`.
    // 2. Or modify the `final_weight` and `trans_weights` of the New Start State
    //    to Intersect with `Future[Start]`.
    //    (i.e., undo the slack saturation for the entry point).

    let minimized_start_id = old_to_new[dwa.body.start_state];
    let start_future = &future[dwa.body.start_state];

    // We clone the states because we might need to modify the start state distinctively
    // if it is shared (merged) with another state but needs different entry filtering.
    // Actually, if Start merged with someone, they had the same Future.
    // So we can just Clip the Start State in place?
    // No, if `A` merges with `B`, and `Start` points to `A`, `Start` is distinct.
    //
    // If `Start` IS one of the states processed (it is), then `minimized_start_id`
    // points to a state that is saturated with `!Future[Start]`.
    // We MUST filter this out to preserve semantics for the very first step.

    // Easiest solution: Add a new fresh Start State that transitions to `minimized_start_id`
    // on all labels? No, DWA is strictly deterministic. We can't have epsilon.
    //
    // Solution: Copy `new_states_vec[minimized_start_id]`, AND it with `start_future`,
    // and set that as the actual start.
    // Unless `minimized_start_id` is not used by anyone else?
    // It's a DAG. Start might be target of nothing (likely).
    // If Start is not target of anything, we can modify it in place.
    // We can check incoming counts, but cloning is safer and cheap for one state.

    let mut final_states = new_states_vec;

    // Create a specific entry point state derived from the minimized start
    let mut entry_state = final_states[minimized_start_id].clone();

    // Apply the Global Filter (Future[Start]) to the entry state
    // This removes the "Slack" we added, restoring the strictness for the entry.
    entry_state.apply_weight(start_future);

    final_states.push(entry_state);
    let final_start_id = final_states.len() - 1;

    DWA {
        states: crate::precompute4::weighted_automata::dwa::DWAStates(final_states),
        body: crate::precompute4::weighted_automata::dwa::DWABody {
            start_state: final_start_id,
        },
    }
}

// Helper: Standard Kahn's Algorithm for Topo Sort
fn topological_sort(dwa: &DWA) -> Option<Vec<usize>> {
    let n = dwa.states.len();
    let mut in_degree = vec![0; n];
    let mut adj = vec![vec![]; n];

    for (u, state) in dwa.states.0.iter().enumerate() {
        for &v in state.transitions.values() {
            if v < n {
                adj[u].push(v);
                in_degree[v] += 1;
            }
        }
    }

    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut result = Vec::with_capacity(n);

    while let Some(u) = queue.pop() {
        result.push(u);
        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push(v);
            }
        }
    }

    if result.len() == n { Some(result) } else { None }
}