use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, VecDeque};

/// Resolve negative codes in a DWA by a single, high-performance, semantics-preserving NWA rewrite.
/// This function implements the delicately crafted semantics of negative codes, which represent a stack-like
/// cancellation mechanism. The transformation ensures the final DWA is free of negative codes and
/// correctly represents the language defined by these complex interactions.
///
/// The process is as follows:
/// 1.  **Convert to NWA**: The DWA is first converted to an NWA to allow for flexible graph manipulations,
///     such as adding epsilon transitions.
/// 2.  **Resolve Cancellations**: The algorithm identifies all `A --neg(c)--> B --c--> C` sequences.
///     Each such "cancellation" is resolved by adding a new epsilon transition `A --eps--> C`, which acts as a shortcut.
///     The weights are combined along the path.
/// 3.  **Propagate Finality**: Finality is propagated backward across all `neg(c)` and epsilon transitions (both
///     original and newly created ones). If a state `A` can reach a final state `F` through a path of such
///     transitions, `A` inherits finality from `F`. This is computed using a standard least fixpoint algorithm.
/// 4.  **Apply Changes & Cleanup**: The newly computed final weights are applied to the NWA states. Then, all
///     negative-code transitions are removed from the graph, as their semantic effects have been fully resolved.
/// 5.  **Determinize**: The resulting NWA, now free of negative codes, is determinized and simplified back into a
///     DWA, which is the final result.
pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let pb = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new(6);
        p.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [Resolving negative codes: {elapsed_precise}] [{wide_bar:.cyan/blue}] step {pos}/{len} ({msg})")
                .expect("progress-bar"),
        );
        Some(p)
    } else {
        None
    };

    if let Some(p) = &pb { p.set_message("DWA -> NWA"); p.set_position(1); }
    let mut nwa = NWA::from_dwa(dwa);

    if let Some(p) = &pb { p.set_message("Compute cancellations"); p.set_position(2); }
    let epsilons_to_add = compute_cancellations(&nwa.states);

    if let Some(p) = &pb { p.set_message("Propagate finality"); p.set_position(3); }
    let final_fix = compute_final_fixpoint(&nwa.states, &epsilons_to_add);

    if let Some(p) = &pb { p.set_message("Apply changes & remove negatives"); p.set_position(4); }
    // Add new epsilons
    for (from, to, w) in epsilons_to_add {
        nwa.states.add_epsilon(from, to, w);
    }
    // Add propagated final weights
    for sid in 0..nwa.states.len() {
        if !final_fix[sid].is_empty() {
            let st = &mut nwa.states.0[sid];
            if let Some(fw) = &mut st.final_weight {
                *fw |= &final_fix[sid];
            } else {
                st.final_weight = Some(final_fix[sid].clone());
            }
        }
    }
    // Remove all negative transitions
    for st in &mut nwa.states.0 {
        st.transitions.retain(|k, _| *k >= 0);
    }

    if let Some(p) = &pb { p.set_message("Determinize"); p.set_position(5); }
    let mut result = nwa.determinize_to_dwa();
    if let Some(p) = &pb { p.set_message("Simplify"); p.set_position(6); }
    result.simplify();
    *dwa = result;

    if let Some(p) = &pb { p.finish_with_message("Done"); }
}

/// Helper to compute the weighted epsilon closure of a state.
fn compute_eps_closure(
    states: &NWAStates,
    start_node: NWAStateID,
    extra_epsilons: &HashMap<NWAStateID, Vec<(NWAStateID, Weight)>>,
) -> HashMap<NWAStateID, Weight> {
    let mut closure: HashMap<NWAStateID, Weight> = HashMap::new();
    let mut q = VecDeque::new();

    closure.insert(start_node, Weight::all());
    q.push_back(start_node);

    while let Some(u) = q.pop_front() {
        let u_w = closure.get(&u).unwrap().clone();

        // Original epsilons
        for (v, w) in &states[u].epsilons {
            let new_w = &u_w & w;
            if new_w.is_empty() { continue; }
            let entry = closure.entry(*v).or_insert_with(Weight::zeros);
            let old_w = entry.clone();
            *entry |= &new_w;
            if *entry != old_w {
                q.push_back(*v);
            }
        }
        // Extra epsilons found during cancellation fixpoint
        if let Some(edges) = extra_epsilons.get(&u) {
            for (v, w) in edges {
                let new_w = &u_w & w;
                if new_w.is_empty() { continue; }
                let entry = closure.entry(*v).or_insert_with(Weight::zeros);
                let old_w = entry.clone();
                *entry |= &new_w;
                if *entry != old_w {
                    q.push_back(*v);
                }
            }
        }
    }
    closure
}

/// Finds all `A --neg(c)--> B --c--> C` patterns, including those chained via epsilon transitions,
/// by iteratively finding new cancellation shortcuts until a fixpoint is reached.
fn compute_cancellations(states: &NWAStates) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let mut all_new_epsilons: HashMap<NWAStateID, Vec<(NWAStateID, Weight)>> = HashMap::new();
    loop {
        let mut changed_in_pass = false;
        let mut pass_epsilons = Vec::new();

        for a in 0..states.len() {
            let state_a = &states[a];
            for (neg_label, (b, w_ab)) in &state_a.transitions {
                if *neg_label >= 0 { continue; }
                let c = neg_label.wrapping_sub(i16::MIN);

                if *b >= states.len() { continue; }

                let eps_closure_of_b = compute_eps_closure(states, *b, &all_new_epsilons);

                for (b_reachable, w_b_br) in eps_closure_of_b {
                    if b_reachable >= states.len() { continue; }
                    if let Some((c_target, w_br_c)) = states[b_reachable].get_transition(c) {
                        let new_w = w_ab & &w_b_br & w_br_c;
                        if !new_w.is_empty() {
                            pass_epsilons.push((a, *c_target, new_w));
                        }
                    }
                }
            }
        }

        for (from, to, w) in pass_epsilons {
            let edges = all_new_epsilons.entry(from).or_default();
            if let Some((_, existing_w)) = edges.iter_mut().find(|(t, _)| *t == to) {
                let old_w = existing_w.clone();
                *existing_w |= &w;
                if *existing_w != old_w {
                    changed_in_pass = true;
                }
            } else {
                edges.push((to, w));
                changed_in_pass = true;
            }
        }

        if !changed_in_pass {
            break;
        }
    }

    all_new_epsilons
        .into_iter()
        .flat_map(|(from, edges)| {
            edges.into_iter().map(move |(to, w)| (from, to, w))
        })
        .collect()
}

/// Compute the least fixpoint of final weights propagated backward along all epsilon edges
/// (original and new) and all negative-labeled edges.
fn compute_final_fixpoint(states: &NWAStates, new_epsilons: &[(NWAStateID, NWAStateID, Weight)]) -> Vec<Weight> {
    let n = states.len();
    let mut fut: Vec<Weight> = vec![Weight::zeros(); n];
    let mut rev: Vec<Vec<(NWAStateID, Weight)>> = vec![vec![]; n];

    // Build reverse graph for propagation
    for u in 0..n {
        // Original epsilons
        for &(v, ref w) in &states[u].epsilons {
            if v < n { rev[v].push((u, w.clone())); }
        }
        // Negative edges
        for (label, (v, w)) in &states[u].transitions {
            if *label < 0 {
                if *v < n { rev[*v].push((u, w.clone())); }
            }
        }
    }
    // New epsilons from cancellations
    for (u, v, w) in new_epsilons {
        if *v < n { rev[*v].push((*u, w.clone())); }
    }

    let mut q: VecDeque<NWAStateID> = VecDeque::new();
    for s in 0..n {
        if let Some(ref fw) = states[s].final_weight {
            if !fw.is_empty() {
                fut[s] = fw.clone();
                q.push_back(s);
            }
        }
    }

    while let Some(v) = q.pop_front() {
        let fv = fut[v].clone();
        if fv.is_empty() { continue; }
        for &(u, ref w_uv) in &rev[v] {
            let add = &fv & w_uv;
            if add.is_empty() { continue; }
            let old = &fut[u];
            if (&add & old) != add {
                fut[u] |= &add;
                q.push_back(u);
            }
        }
    }
    fut
}
