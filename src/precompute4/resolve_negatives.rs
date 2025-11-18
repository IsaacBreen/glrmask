use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

type Code = i16;
type QueryKey = (NWAStateID, Code);

#[inline]
fn is_negative_symbol(label: Code) -> bool { label < 0 && label != DEFAULT_TRANSITION_SYMBOL }

fn make_progress_bar(length: u64, template: &str) -> Option<ProgressBar> {
    if !PROGRESS_BAR_ENABLED {
        return None;
    }
    let pb = ProgressBar::new(length);
    pb.set_style(ProgressStyle::default_bar().template(template).expect("progress-bar"));
    Some(pb)
}

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
    apply_cancellations(nwa, &(0..nwa.states.len()).collect());

    progress_step(&pb, 2, "Propagate finality");
    apply_finality_fixpoint(nwa, &(0..nwa.states.len()).collect());

    progress_step(&pb, 3, "Apply changes & remove negatives");
    remove_negative_transitions(nwa, &(0..nwa.states.len()).collect());
    crate::debug!(4, "Applied changes to NWA.");

    if let Some(p) = &pb {
        p.finish_with_message("Done");
    }
}

pub fn apply_cancellations(nwa: &mut NWA, source_states_filter: &HashSet<NWAStateID>) {
    let epsilons_to_add = compute_cancellations(&nwa.states, source_states_filter);
    crate::debug!(4, "Computed {} new epsilon transitions from cancellations.", epsilons_to_add.len());
    for (from, to, w) in epsilons_to_add {
        nwa.states.add_epsilon(from, to, w);
    }
}

pub fn apply_finality_fixpoint(nwa: &mut NWA, source_states_filter: &HashSet<NWAStateID>) {
    let final_fix = compute_finality_fixpoint(&nwa.states, source_states_filter);
    for &sid in source_states_filter {
        if sid >= nwa.states.len() || final_fix[sid].is_empty() {
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

pub fn remove_negative_transitions(nwa: &mut NWA, source_states_filter: &HashSet<NWAStateID>) {
    for (stid, st) in nwa.states.0.iter_mut().enumerate() {
        if !source_states_filter.contains(&stid) {
            continue;
        }
        st.transitions.retain(|&label, _| !is_negative_symbol(label));
    }
}

/// Resolve negative codes in a DWA by a single, high-performance, semantics-preserving NWA rewrite.
pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let now = Instant::now();
    crate::debug!(4, "Resolving negative codes in DWA with {} states...", dwa.states.len());

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
    crate::debug!(4, "Applied changes, NWA has {} states before determinization.", nwa.states.len());
    crate::debug!(4, "Stats for NWA after negative resolution:\n{}", nwa.stats());

    progress_step(&pb, 3, "Determinize");
    let mut result = nwa.determinize_to_dwa();

    progress_step(&pb, 4, "Simplify");
    result.simplify();
    *dwa = result;
    crate::debug!(4, "Stats for final DWA after negative resolution:\n{}", dwa.stats());

    if let Some(p) = &pb {
        p.finish_with_message("Done");
    }
    crate::debug!(4, "resolve_negative_codes_in_dwa took: {:?}", now.elapsed());
}

fn compute_cancellations(states: &NWAStates, source_states_filter: &HashSet<NWAStateID>) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let n = states.len();

    let mut queries: Vec<HashMap<QueryKey, Weight>> = vec![HashMap::new(); n];
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, Code, Weight)> = VecDeque::new();

    // new_eps_from[from][to] = weight of the cancellation epsilon from `from` to `to`.
    let mut new_eps_from: Vec<HashMap<NWAStateID, Weight>> = vec![HashMap::new(); n];

    // Seed from negative transitions.
    for &a in source_states_filter {
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

    while let Some((s, a, c, w_as)) = worklist.pop_front() {
        // First, propagate this query through any already-known cancellation epsilons
        // originating from the current location `s`.
        if !new_eps_from[s].is_empty() {
            for (&target, eps_w) in &new_eps_from[s] {
                let prop_w = &w_as & eps_w;
                if prop_w.is_empty() {
                    continue;
                }
                let query_key: QueryKey = (a, c);
                let query_weight = queries[target].entry(query_key).or_default();
                let old_qw = query_weight.clone();
                *query_weight |= &prop_w;
                if *query_weight != old_qw {
                    worklist.push_back((target, a, c, query_weight.clone()));
                }
            }
        }

        let mut check_cancellations = |target: NWAStateID,
                                       w_st: &Weight,
                                       worklist: &mut VecDeque<(NWAStateID, NWAStateID, Code, Weight)>| {
            let new_eps_w = &w_as & w_st;
            if new_eps_w.is_empty() {
                return;
            }

            // Epsilon summarizing a cancellation from `a` (the negative's source) to `target`.
            let eps_from_a = &mut new_eps_from[a];
            let eps_weight = eps_from_a.entry(target).or_default();
            let old_eps_w = eps_weight.clone();
            *eps_weight |= &new_eps_w;

            if *eps_weight != old_eps_w {
                // When this epsilon grows, any existing queries that are currently at `a`
                // can also traverse it.
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

    let mut result = Vec::new();
    for (from, targets) in new_eps_from.into_iter().enumerate() {
        for (to, w) in targets {
            result.push((from, to, w));
        }
    }
    result
}

fn compute_finality_fixpoint(states: &NWAStates, source_states_filter: &HashSet<NWAStateID>) -> Vec<Weight> {
    let n = states.len();
    let mut future_final: Vec<Weight> = vec![Weight::zeros(); n];
    let mut predecessors: Vec<Vec<(NWAStateID, Weight)>> = vec![vec![]; n];

    for u in 0..n {
        for &(v, ref w) in &states[u].epsilons {
            predecessors[v].push((u, w.clone()));
        }
        for (&label, targets) in &states[u].transitions {
            if !is_negative_symbol(label) {
                continue;
            }
            for (v, w) in targets {
                predecessors[*v].push((u, w.clone()));
            }
        }
    }

    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    for s in 0..n {
        if let Some(ref fw) = states[s].final_weight {
            if !fw.is_empty() {
                future_final[s] = fw.clone();
                queue.push_back(s);
            }
        }
    }

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
