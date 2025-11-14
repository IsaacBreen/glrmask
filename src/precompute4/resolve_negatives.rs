use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Convenience alias for transition labels (both positive and negative codes).
type Code = i16;

/// Key identifying an in-flight cancellation search:
/// `(origin_state, positive_code_being_cancelled)`.
type QueryKey = (NWAStateID, Code);

#[inline]
fn is_negative_symbol(label: Code) -> bool {
    label < 0 && label != DEFAULT_TRANSITION_SYMBOL
}

/// Optionally construct a configured progress bar.
fn make_progress_bar(length: u64, template: &str) -> Option<ProgressBar> {
    if !PROGRESS_BAR_ENABLED {
        return None;
    }
    let pb = ProgressBar::new(length);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(template)
            .expect("progress-bar"),
    );
    Some(pb)
}

/// Helper to advance the progress bar if it is enabled.
fn progress_step(pb: &Option<ProgressBar>, step: u64, msg: &str) {
    if let Some(p) = pb {
        p.set_message(msg.to_string());
        p.set_position(step);
    }
}

pub fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    let pb = make_progress_bar(
        3,
        "{spinner:.green} [Resolving negatives in NWA: {elapsed_precise}] \
         [{wide_bar:.cyan/blue}] step {pos}/{len} ({msg})",
    );

    progress_step(&pb, 1, "Compute cancellations");
    apply_cancellations(nwa);

    progress_step(&pb, 2, "Propagate finality");
    apply_finality_fixpoint(nwa);

    progress_step(&pb, 3, "Apply changes & remove negatives");
    remove_negative_transitions(nwa);
    crate::debug!(4, "Applied changes to NWA.");

    if let Some(p) = &pb {
        p.finish_with_message("Done");
    }
}

pub fn apply_cancellations(nwa: &mut NWA) {
    let epsilons_to_add = compute_cancellations(&nwa.states);
    crate::debug!(
        4,
        "Computed {} new epsilon transitions from cancellations.",
        epsilons_to_add.len()
    );
    for (from, to, w) in epsilons_to_add {
        nwa.states.add_epsilon(from, to, w);
    }
}

pub fn apply_finality_fixpoint(nwa: &mut NWA) {
    let final_fix = compute_finality_fixpoint(&nwa.states);
    for sid in 0..nwa.states.len() {
        if final_fix[sid].is_empty() {
            continue;
        }
        let st = &mut nwa.states.0[sid];
        if let Some(fw) = &mut st.final_weight {
            *fw |= &final_fix[sid];
        } else {
            st.final_weight = Some(final_fix[sid].clone());
        }
    }
}

pub fn remove_negative_transitions(nwa: &mut NWA) {
    for st in &mut nwa.states.0 {
        st.transitions.retain(|&label, _| !is_negative_symbol(label));
    }
}

/// Resolve negative codes in a DWA by a single, high-performance, semantics-preserving NWA rewrite.
/// This function implements the delicately crafted semantics of negative codes, which represent a stack-like
/// cancellation mechanism. The transformation ensures the final DWA is free of negative codes and
/// correctly represents the language defined by these complex interactions.
///
/// The process is as follows:
/// 1.  Convert to NWA: The DWA is first converted to an NWA to allow for flexible graph manipulations,
///     such as adding epsilon transitions.
/// 2.  Resolve cancellations: The algorithm identifies all `A --neg(c)--> B --c--> C` sequences
///     (optionally using epsilon and default transitions).
///     Each such cancellation is resolved by adding a new epsilon transition `A --eps--> C`, with
///     weights combined along the path.
/// 3.  Propagate finality: Finality is propagated backward across all negative, epsilon,
///     and default transitions. If a state `A` can reach a final state `F` through such a path,
///     `A` inherits finality from `F`.
/// 4.  Apply changes and clean up: The newly computed final weights are applied and all
///     negative-code transitions are removed.
/// 5.  Determinize: The resulting NWA, now free of negative codes, is determinized and simplified
///     back into a DWA.
pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let now = Instant::now();
    crate::debug!(
        4,
        "Resolving negative codes in DWA with {} states...",
        dwa.states.len()
    );

    let pb = make_progress_bar(
        4,
        "{spinner:.green} [Resolving negative codes: {elapsed_precise}] \
         [{wide_bar:.cyan/blue}] step {pos}/{len} ({msg})",
    );

    progress_step(&pb, 1, "DWA -> NWA");
    let mut nwa = NWA::from_dwa(dwa);
    crate::debug!(4, "Converted to NWA with {} states.", nwa.states.len());
    crate::debug!(4, "Stats for NWA from DWA:\n{}", nwa.stats());

    progress_step(&pb, 2, "Resolve negatives in NWA");
    resolve_negative_codes_in_nwa(&mut nwa);
    crate::debug!(
        4,
        "Applied changes, NWA has {} states before determinization.",
        nwa.states.len()
    );
    crate::debug!(4, "Stats for NWA after negative resolution:\n{}", nwa.stats());

    progress_step(&pb, 3, "Determinize");
    let mut result = nwa.determinize_to_dwa();

    progress_step(&pb, 4, "Simplify");
    result.simplify();
    *dwa = result;
    crate::debug!(
        4,
        "Stats for final DWA after negative resolution:\n{}",
        dwa.stats()
    );

    if let Some(p) = &pb {
        p.finish_with_message("Done");
    }
    crate::debug!(4, "resolve_negative_codes_in_dwa took: {:?}", now.elapsed());
}

/// Finds all `A --neg(c)--> B --c--> C` patterns, including those chained via epsilon transitions,
/// by iteratively finding new cancellation shortcuts until a fixpoint is reached.
fn compute_cancellations(states: &NWAStates) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let n = states.len();

    // `queries[s]` stores cancellation searches that have reached state `s`.
    // A search is identified by its origin state `a` and the positive code `c` it is looking for.
    // The map stores the accumulated weight of the path `a --neg(c)--> ... --eps*--> s`.
    let mut queries: Vec<HashMap<QueryKey, Weight>> = vec![HashMap::new(); n];
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, Code, Weight)> = VecDeque::new();
    let mut new_epsilons: HashMap<(NWAStateID, NWAStateID), Weight> = HashMap::new();

    // 1. Initialize worklist with all negative transitions.
    // Each `a --neg(c)--> b` starts a search at `b` for a matching `c`.
    for a in 0..n {
        for (&label, targets) in &states[a].transitions {
            if !is_negative_symbol(label) {
                continue;
            }
            let c = label.wrapping_sub(Code::MIN);
            for (b, w_ab) in targets {
                if *b >= n {
                    continue;
                }
                let query_key: QueryKey = (a, c);
                let query_weight = queries[*b].entry(query_key).or_default();
                let old_w = query_weight.clone();
                *query_weight |= w_ab;
                if *query_weight != old_w {
                    worklist.push_back((*b, a, c, query_weight.clone()));
                }
            }
        }
    }

    // 2. Main fixpoint loop: propagate searches and generate new epsilon shortcuts.
    while let Some((s, a, c, w_as)) = worklist.pop_front() {
        // Check for cancellations at the current state `s`.
        // A cancellation occurs if `s` has an outgoing transition on `c`,
        // or on the default symbol.
        let mut check_cancellations = |target: NWAStateID,
                                       w_st: &Weight,
                                       worklist: &mut VecDeque<(NWAStateID, NWAStateID, Code, Weight)>| {
            let new_eps_w = &w_as & w_st;
            if new_eps_w.is_empty() {
                return;
            }

            let eps_key = (a, target);
            let eps_weight = new_epsilons.entry(eps_key).or_default();
            let old_eps_w = eps_weight.clone();
            *eps_weight |= &new_eps_w;

            // If this epsilon is new or strengthened, it can propagate existing searches.
            if *eps_weight != old_eps_w {
                // Any search that has reached `a` can now cross this new epsilon to `target`.
                let queries_at_a = queries[a].clone();
                for (&(a_prime, c_prime), w_a_prime_a) in &queries_at_a {
                    let prop_w = w_a_prime_a & &*eps_weight;
                    if prop_w.is_empty() {
                        continue;
                    }

                    let query_key: QueryKey = (a_prime, c_prime);
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
                if *t < n {
                    check_cancellations(*t, w_st, &mut worklist);
                }
            }
        }
        if let Some(default_targets) = states[s].transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for (target, weight) in default_targets {
                check_cancellations(*target, weight, &mut worklist);
            }
        }

        // 2b. Propagate the current search forward over original epsilon edges.
        for (t, w_st) in &states[s].epsilons {
            if *t >= n {
                continue;
            }
            let prop_w = &w_as & w_st;
            if prop_w.is_empty() {
                continue;
            }

            let query_key: QueryKey = (a, c);
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
fn compute_finality_fixpoint(states: &NWAStates) -> Vec<Weight> {
    let n = states.len();
    let mut future_final: Vec<Weight> = vec![Weight::zeros(); n];
    let mut predecessors: Vec<Vec<(NWAStateID, Weight)>> = vec![vec![]; n];

    // Build reverse graph for propagation.
    for u in 0..n {
        // All epsilons.
        for &(v, ref w) in &states[u].epsilons {
            if v < n {
                predecessors[v].push((u, w.clone()));
            }
        }
        // Negative edges.
        for (&label, targets) in &states[u].transitions {
            if !is_negative_symbol(label) {
                continue;
            }
            for (v, w) in targets {
                if *v < n {
                    predecessors[*v].push((u, w.clone()));
                }
            }
        }
    }

    // Initialize with already-final states.
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    for s in 0..n {
        if let Some(ref fw) = states[s].final_weight {
            if !fw.is_empty() {
                future_final[s] = fw.clone();
                queue.push_back(s);
            }
        }
    }

    // Standard worklist fixpoint.
    while let Some(v) = queue.pop_front() {
        let fv = future_final[v].clone();
        if fv.is_empty() {
            continue;
        }
        for &(u, ref w_uv) in &predecessors[v] {
            let add = &fv & w_uv;
            if add.is_empty() {
                continue;
            }
            let old = &future_final[u];
            if (&add & old) != add {
                future_final[u] |= &add;
                queue.push_back(u);
            }
        }
    }

    future_final
}
