use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use crate::precompute4::weighted_automata::common::Label;

type Code = Label;
type QueryKey = (NWAStateID, Code);

#[inline]
fn is_negative_symbol(label: Code) -> bool { label < 0 && label != DEFAULT_TRANSITION_SYMBOL }

fn make_progress_bar(length: u64, template: &str) -> Option<ProgressBar> {
    if !PROGRESS_BAR_ENABLED { return None; }
    let pb = ProgressBar::new(length);
    pb.set_style(ProgressStyle::default_bar().template(template).expect("progress-bar"));
    Some(pb)
}
fn progress_step(pb: &Option<ProgressBar>, step: u64, msg: &str) {
    if let Some(p) = pb { p.set_message(msg.to_string()); p.set_position(step); }
}

pub fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    let pb = make_progress_bar(3, "{spinner:.green} [Resolving negatives in NWA: {elapsed_precise}] [{wide_bar:.cyan/blue}] step {pos}/{len} ({msg})");
    let all_states: HashSet<NWAStateID> = (0..nwa.states.len()).collect();
    progress_step(&pb, 1, "Compute cancellations");
    apply_cancellations(&mut nwa.states, &all_states);
    progress_step(&pb, 2, "Propagate finality");
    apply_finality_fixpoint(&mut nwa.states, &all_states);
    progress_step(&pb, 3, "Apply changes & remove negatives");
    remove_negative_transitions(&mut nwa.states, &all_states);
    if let Some(p) = &pb { p.finish_with_message("Done"); }
}

pub fn apply_cancellations(states: &mut NWAStates, source_states_filter: &HashSet<NWAStateID>) {
    let epsilons_to_add = compute_cancellations(states, source_states_filter);
    for (from, to, w) in epsilons_to_add { states.add_epsilon(from, to, w); }
}

pub fn apply_finality_fixpoint(states: &mut NWAStates, source_states_filter: &HashSet<NWAStateID>) {
    let final_fix = compute_finality_fixpoint(states, source_states_filter);
    for &sid in source_states_filter {
        if let Some(add) = final_fix.get(&sid) {
            if let Some(fw) = &mut states[sid].final_weight { *fw |= add; } else { states[sid].final_weight = Some(add.clone()); }
        }
    }
}

pub fn remove_negative_transitions(states: &mut NWAStates, source_states_filter: &HashSet<NWAStateID>) {
    for &sid in source_states_filter {
        states.0[sid].transitions.retain(|&label, _| !is_negative_symbol(label));
    }
}

pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let now = Instant::now();
    let pb = make_progress_bar(4, "{spinner:.green} [Resolving negative codes: {elapsed_precise}] [{wide_bar:.cyan/blue}] step {pos}/{len} ({msg})");
    progress_step(&pb, 1, "DWA -> NWA");
    let mut nwa = NWA::from_dwa(dwa);
    progress_step(&pb, 2, "Resolve negatives in NWA");
    resolve_negative_codes_in_nwa(&mut nwa);
    progress_step(&pb, 3, "Determinize");
    let mut result = nwa.determinize_to_dwa();
    progress_step(&pb, 4, "Simplify");
    result.simplify();
    *dwa = result;
    if let Some(p) = &pb { p.finish_with_message("Done"); }
    crate::debug!(6, "resolve_negative_codes_in_dwa took: {:?}", now.elapsed());
}

fn compute_cancellations(states: &NWAStates, source_states_filter: &HashSet<NWAStateID>) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let n = states.len();
    let mut queries: HashMap<NWAStateID, HashMap<QueryKey, Weight>> = HashMap::new();
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, Code, Weight)> = VecDeque::new();
    let mut new_eps_from: HashMap<NWAStateID, HashMap<NWAStateID, Weight>> = HashMap::new();

    for &a in source_states_filter {
        for (&label, targets) in &states[a].transitions {
            if !is_negative_symbol(label) { continue; }
            let c = label.wrapping_sub(Code::MIN);
            for (b, w_ab) in targets {
                if *b >= n { continue; }
                let query_key: QueryKey = (a, c);
                let query_weight = queries.entry(*b).or_default().entry(query_key).or_default();
                let old_w = query_weight.clone();
                *query_weight |= w_ab;
                if *query_weight != old_w { worklist.push_back((*b, a, c, query_weight.clone())); }
            }
        }
    }

    while let Some((s, a, c, w_as)) = worklist.pop_front() {
        if let Some(epsilons_from_s) = new_eps_from.get(&s) {
            for (&target, eps_w) in epsilons_from_s {
                let prop_w = &w_as & eps_w;
                if prop_w.is_empty() { continue; }
                let query_key: QueryKey = (a, c);
                let query_weight = queries.entry(target).or_default().entry(query_key).or_default();
                let old_qw = query_weight.clone();
                *query_weight |= &prop_w;
                if *query_weight != old_qw { worklist.push_back((target, a, c, query_weight.clone())); }
            }
        }

        let mut check_cancellations = |target: NWAStateID, w_st: &Weight, worklist: &mut VecDeque<(NWAStateID, NWAStateID, Code, Weight)>| {
            let new_eps_w = &w_as & w_st;
            if new_eps_w.is_empty() { return; }
            let eps_from_a = new_eps_from.entry(a).or_default();
            let eps_weight = eps_from_a.entry(target).or_default();
            let old_eps_w = eps_weight.clone();
            *eps_weight |= &new_eps_w;

            if *eps_weight != old_eps_w {
                if let Some(queries_at_a) = queries.get(&a) {
                    for (&(a_prime, c_prime), w_a_prime_a) in &queries_at_a.clone() {
                        let prop_w = w_a_prime_a & &*eps_weight;
                        if prop_w.is_empty() { continue; }
                        let query_key: QueryKey = (a_prime, c_prime);
                        let query_weight = queries.entry(target).or_default().entry(query_key).or_default();
                        let old_qw = query_weight.clone();
                        *query_weight |= &prop_w;
                        if *query_weight != old_qw { worklist.push_back((target, a_prime, c_prime, query_weight.clone())); }
                    }
                }
            }
        };

        if let Some(pos_targets) = states[s].transitions.get(&c) {
            for (t, w_st) in pos_targets { if *t < n { check_cancellations(*t, w_st, &mut worklist); } }
        }
        if let Some(default_targets) = states[s].transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for (target, weight) in default_targets { check_cancellations(*target, weight, &mut worklist); }
        }
        for (t, w_st) in &states[s].epsilons {
            if *t >= n { continue; }
            let prop_w = &w_as & w_st;
            if prop_w.is_empty() { continue; }
            let query_key: QueryKey = (a, c);
            let query_weight = queries.entry(*t).or_default().entry(query_key).or_default();
            let old_qw = query_weight.clone();
            *query_weight |= &prop_w;
            if *query_weight != old_qw { worklist.push_back((*t, a, c, query_weight.clone())); }
        }
    }

    let mut result = Vec::new();
    for (from, targets) in new_eps_from {
        for (to, w) in targets { result.push((from, to, w)); }
    }
    result
}

fn compute_finality_fixpoint(states: &NWAStates, source_states_filter: &HashSet<NWAStateID>) -> HashMap<NWAStateID, Weight> {
    let n = states.len();
    if n == 0 || source_states_filter.is_empty() { return HashMap::new(); }

    #[derive(Clone, Copy)]
    enum PredEdge { Epsilon { from: NWAStateID, eps_idx: usize }, Negative { from: NWAStateID, label: Code, trans_idx: usize } }

    let mut visited = vec![false; n];
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    let mut reachable_states: Vec<NWAStateID> = Vec::new();
    let mut preds: HashMap<NWAStateID, Vec<PredEdge>> = HashMap::new();

    for &a in source_states_filter {
        if a < n && !visited[a] { visited[a] = true; queue.push_back(a); reachable_states.push(a); }
    }

    while let Some(s) = queue.pop_front() {
        for (eps_idx, &(target, ref w)) in states[s].epsilons.iter().enumerate() {
            if target >= n || w.is_empty() { continue; }
            preds.entry(target).or_default().push(PredEdge::Epsilon { from: s, eps_idx });
            if !visited[target] { visited[target] = true; queue.push_back(target); reachable_states.push(target); }
        }
        if source_states_filter.contains(&s) {
            for (&label, targets) in &states[s].transitions {
                if !is_negative_symbol(label) { continue; }
                for (trans_idx, &(target, ref w)) in targets.iter().enumerate() {
                    if target >= n || w.is_empty() { continue; }
                    preds.entry(target).or_default().push(PredEdge::Negative { from: s, label, trans_idx });
                    if !visited[target] { visited[target] = true; queue.push_back(target); reachable_states.push(target); }
                }
            }
        }
    }

    let mut future_final_all: HashMap<NWAStateID, Weight> = HashMap::new();
    let mut worklist: VecDeque<NWAStateID> = VecDeque::new();

    for &s in &reachable_states {
        if let Some(ref fw) = states[s].final_weight {
            if !fw.is_empty() { future_final_all.insert(s, fw.clone()); worklist.push_back(s); }
        }
    }

    while let Some(s) = worklist.pop_front() {
        let f_s = match future_final_all.get(&s) { Some(w) if !w.is_empty() => w.clone(), _ => continue, };
        if let Some(pred_edges) = preds.get(&s) {
            for edge in pred_edges {
                let (pred_state, edge_w): (NWAStateID, &Weight) = match *edge {
                    PredEdge::Epsilon { from, eps_idx } => (from, &states[from].epsilons[eps_idx].1),
                    PredEdge::Negative { from, label, trans_idx } => (from, &states[from].transitions[&label][trans_idx].1),
                };
                let add = &f_s & edge_w;
                if add.is_empty() { continue; }
                let entry = future_final_all.entry(pred_state).or_insert_with(Weight::zeros);
                let old = entry.clone();
                *entry |= &add;
                if *entry != old { worklist.push_back(pred_state); }
            }
        }
    }

    let mut result: HashMap<NWAStateID, Weight> = HashMap::new();
    for &a in source_states_filter {
        if a < n {
            if let Some(w) = future_final_all.get(&a) { if !w.is_empty() { result.insert(a, w.clone()); } }
        }
    }
    result
}
