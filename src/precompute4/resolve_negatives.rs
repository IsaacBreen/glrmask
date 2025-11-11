use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

pub fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    let pb = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new(3);
        p.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [Resolving negatives in NWA: {elapsed_precise}] [{wide_bar:.cyan/blue}] step {pos}/{len} ({msg})")
                .expect("progress-bar"),
        );
        Some(p)
    } else {
        None
    };

    if let Some(p) = &pb { p.set_message("Compute cancellations"); p.set_position(1); }
    let epsilons_to_add = compute_cancellations(&nwa.states);
    crate::debug!(4, "Computed {} new epsilon transitions from cancellations.", epsilons_to_add.len());

    if let Some(p) = &pb { p.set_message("Propagate finality"); p.set_position(2); }
    let final_fix = compute_final_fixpoint(&nwa.states, &epsilons_to_add);

    if let Some(p) = &pb { p.set_message("Apply changes & remove negatives"); p.set_position(3); }
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
    crate::debug!(4, "Applied changes to NWA.");

    if let Some(p) = &pb { p.finish_with_message("Done"); }
}

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
    let now = Instant::now();
    crate::debug!(4, "Resolving negative codes in DWA with {} states...", dwa.states.len());

    let pb = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new(4);
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
    crate::debug!(4, "Converted to NWA with {} states.", nwa.states.len());
    crate::debug!(4, "Stats for NWA from DWA:\n{}", nwa.stats());

    if let Some(p) = &pb { p.set_message("Resolve negatives in NWA"); p.set_position(2); }
    resolve_negative_codes_in_nwa(&mut nwa);
    crate::debug!(4, "Applied changes, NWA has {} states before determinization.", nwa.states.len());
    crate::debug!(4, "Stats for NWA after negative resolution:\n{}", nwa.stats());

    if let Some(p) = &pb { p.set_message("Determinize"); p.set_position(3); }
    let mut result = nwa.determinize_to_dwa();
    if let Some(p) = &pb { p.set_message("Simplify"); p.set_position(4); }
    result.simplify();
    *dwa = result;
    crate::debug!(4, "Stats for final DWA after negative resolution:\n{}", dwa.stats());

    if let Some(p) = &pb { p.finish_with_message("Done"); }
    crate::debug!(4, "resolve_negative_codes_in_dwa took: {:?}", now.elapsed());
}

/// Finds all `A --neg(c)--> B --c--> C` patterns, including those chained via epsilon transitions,
/// by iteratively finding new cancellation shortcuts until a fixpoint is reached.
fn compute_cancellations(states: &NWAStates) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let n = states.len();
    // `queries[s]` stores cancellation "searches" that have reached state `s`.
    // A search is identified by its origin state `a` and the positive code `c` it's looking for.
    // The map stores the accumulated weight of the path `a --neg(c)--> ... --eps*--> s`.
    let mut queries: Vec<HashMap<(NWAStateID, i16), Weight>> = vec![HashMap::new(); n];
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, i16, Weight)> = VecDeque::new();
    let mut new_epsilons: HashMap<(NWAStateID, NWAStateID), Weight> = HashMap::new();

    // 1. Initialize worklist with all negative transitions.
    // Each `a --neg(c)--> b` starts a search at `b` for a matching `c`.
    for a in 0..n {
        for (label, targets) in &states[a].transitions {
            if *label < 0 {
                let c = label.wrapping_sub(i16::MIN);
                for (b, w_ab) in targets {
                    if *b >= n { continue; }
                    let query_key = (a, c);
                    let query_weight = queries[*b].entry(query_key).or_default();
                    let old_w = query_weight.clone();
                    *query_weight |= w_ab;
                    if *query_weight != old_w {
                        worklist.push_back((*b, a, c, query_weight.clone()));
                    }
                }
            }
        }
    }

    // 2. Main fixpoint loop: propagate searches and generate new epsilon shortcuts.
    while let Some((s, a, c, w_as)) = worklist.pop_front() {
        // 2a. Check for cancellations at the current state `s`.
        // A cancellation occurs if `s` has an outgoing transition on `c`.
        let mut check_cancellations = |target: NWAStateID, w_st: &Weight, worklist: &mut VecDeque<_>| {
            let new_eps_w = &w_as & w_st;
            if new_eps_w.is_empty() { return; }

            let eps_key = (a, target);
            let eps_weight = new_epsilons.entry(eps_key).or_default();
            let old_eps_w = eps_weight.clone();
            *eps_weight |= &new_eps_w;

            // If this epsilon is new or stronger, it can propagate existing searches.
            if *eps_weight != old_eps_w {
                // Any search that has reached `a` can now cross this new epsilon to `target`.
                // To avoid borrow checker issues, clone the queries at state `a`.
                let queries_at_a = queries[a].clone();
                for (&(a_prime, c_prime), w_a_prime_a) in &queries_at_a {
                    let prop_w = w_a_prime_a & &*eps_weight;
                    if prop_w.is_empty() { continue; }

                    let query_key = (a_prime, c_prime);
                    let query_weight = queries[target].entry(query_key).or_default();
                    let old_qw = query_weight.clone();
                    *query_weight |= &prop_w;
                    if *query_weight != old_qw {
                        worklist.push_back((target, a_prime, c_prime, query_weight.clone()));
                    }
                }
            }
        };

        if let Some(pos_targets) = states[s].transitions.get(&c) {
            for (t, w_st) in pos_targets {
                if *t < n { check_cancellations(*t, w_st, &mut worklist); }
            }
        }
        for def in &states[s].default {
            if !def.exceptions.contains(&c) && def.target < n {
                check_cancellations(def.target, &def.weight, &mut worklist);
            }
        }

        // 2b. Propagate the current search forward over original epsilon edges.
        for (t, w_st) in &states[s].epsilons {
            if *t >= n { continue; }
            let prop_w = &w_as & w_st;
            if prop_w.is_empty() { continue; }

            let query_key = (a, c);
            let query_weight = queries[*t].entry(query_key).or_default();
            let old_qw = query_weight.clone();
            *query_weight |= &prop_w;
            if *query_weight != old_qw {
                worklist.push_back((*t, a, c, query_weight.clone()));
            }
        }
    }

    new_epsilons
        .into_iter()
        .map(|((from, to), w)| (from, to, w))
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
        for (label, targets) in &states[u].transitions {
            if *label < 0 {
                for (v, w) in targets {
                    if *v < n { rev[*v].push((u, w.clone())); }
                }
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
