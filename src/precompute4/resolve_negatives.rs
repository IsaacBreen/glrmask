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
/// 2.  **Resolve Cancellations**: The algorithm identifies all instances of `U --c--> S --neg(c)--> T` sequences.
///     Each such "cancellation" is resolved by adding a new epsilon transition `U --eps--> T`, which acts as a shortcut.
///     The weights are combined along the path. Crucially, the `neg(c)` edges that participate in a cancellation
///     are marked and ignored in the next step.
/// 3.  **Propagate Finality**: For all remaining (non-cancelled) `neg(c)` edges, finality is propagated backward.
///     If a state can reach a final state through a path of epsilon and non-cancelled `neg(c)` transitions,
///     it also becomes final. This is computed using a standard least fixpoint algorithm.
/// 4.  **Apply Changes & Cleanup**: The newly computed final weights are applied to the NWA states. Then, all
///     negative-code transitions are removed from the graph, as their semantic effects have been fully resolved.
/// 5.  **Determinize**: The resulting NWA, now free of negative codes, is determinized and simplified back into a
///     DWA, which is the final result.
pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let pb = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new(5);
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
    let (epsilons_to_add, cancelled_negs) = compute_cancellations(&nwa.states);

    for (from, to, w) in epsilons_to_add {
        nwa.states.add_epsilon(from, to, w);
    }

    if let Some(p) = &pb { p.set_message("Propagate finality"); p.set_position(3); }
    let final_fix = compute_final_fixpoint_eps_and_neg(&nwa.states, &cancelled_negs);

    if let Some(p) = &pb { p.set_message("Apply weights & remove negatives"); p.set_position(4); }
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

    for st in &mut nwa.states.0 {
        st.transitions.retain(|k, _| *k >= 0);
    }

    if let Some(p) = &pb { p.set_message("Determinize & simplify"); p.set_position(5); }
    let mut result = nwa.determinize_to_dwa();
    result.simplify();
    *dwa = result;

    if let Some(p) = &pb { p.finish_with_message("Done"); }
}

/// Builds a map of predecessors for each state for efficient lookup.
fn build_predecessor_map(states: &NWAStates) -> HashMap<NWAStateID, Vec<(NWAStateID, i16, &Weight)>> {
    let mut preds: HashMap<NWAStateID, Vec<(NWAStateID, i16, &Weight)>> = HashMap::new();
    for u in 0..states.len() {
        for (label, (v, w)) in &states[u].transitions {
            preds.entry(*v).or_default().push((u, *label, w));
        }
    }
    preds
}

/// Finds all `U --c--> S --neg(c)--> T` patterns and returns epsilon shortcuts to add,
/// and a set of the (S, neg(c)) edges that were cancelled.
fn compute_cancellations(states: &NWAStates) -> (Vec<(NWAStateID, NWAStateID, Weight)>, HashMap<(NWAStateID, i16), bool>) {
    let mut epsilons_to_add = Vec::new();
    let mut cancelled_negs = HashMap::new();
    let preds = build_predecessor_map(states);

    for s in 0..states.len() {
        for (neg_label, (t, w_neg)) in &states[s].transitions {
            if *neg_label >= 0 { continue; }
            let c = neg_label.wrapping_sub(i16::MIN);

            if let Some(s_preds) = preds.get(&s) {
                for (u, pred_label, w_pos) in s_preds {
                    if *pred_label == c {
                        let new_w = *w_pos & w_neg;
                        if !new_w.is_empty() {
                            epsilons_to_add.push((*u, *t, new_w));
                            cancelled_negs.insert((s, *neg_label), true);
                        }
                    }
                }
            }
        }
    }
    (epsilons_to_add, cancelled_negs)
}

/// Compute the least fixpoint of final weights propagated backward along epsilon edges
/// and non-cancelled negative-labeled edges.
fn compute_final_fixpoint_eps_and_neg(states: &NWAStates, cancelled_negs: &HashMap<(NWAStateID, i16), bool>) -> Vec<Weight> {
    let n = states.len();
    let mut fut: Vec<Weight> = vec![Weight::zeros(); n];
    let mut rev: Vec<Vec<(NWAStateID, Weight)>> = vec![vec![]; n];

    for u in 0..n {
        for &(v, ref w) in &states[u].epsilons {
            if v < n { rev[v].push((u, w.clone())); }
        }
        for (label, (v, w)) in &states[u].transitions {
            if *label < 0 && !cancelled_negs.contains_key(&(u, *label)) {
                if *v < n { rev[*v].push((u, w.clone())); }
            }
        }
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
