use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::precompute4::weighted_automata::common::Label;
use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;

#[inline] fn is_neg(l: Label) -> bool { l < 0 && l != DEFAULT_TRANSITION_SYMBOL }

pub fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    let pb = if PROGRESS_BAR_ENABLED {
        let pb = ProgressBar::new(3);
        pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} [Resolve Negatives] {pos}/{len} {msg}").unwrap());
        Some(pb)
    } else { None };

    let all_states: HashSet<NWAStateID> = (0..nwa.states.len()).collect();
    
    if let Some(p) = &pb { p.set_message("Cancellations"); p.set_position(1); }
    apply_cancellations(&mut nwa.states, &all_states);

    if let Some(p) = &pb { p.set_message("Finality"); p.set_position(2); }
    apply_finality_fixpoint(&mut nwa.states, &all_states);

    if let Some(p) = &pb { p.set_message("Removing negatives"); p.set_position(3); }
    remove_negative_transitions(&mut nwa.states, &all_states);

    if let Some(p) = &pb { p.finish_with_message("Done"); }
}

pub fn apply_cancellations(states: &mut NWAStates, filter: &HashSet<NWAStateID>) {
    for (from, to, w) in compute_cancellations(states, filter) { states.add_epsilon(from, to, w); }
}

pub fn apply_finality_fixpoint(states: &mut NWAStates, filter: &HashSet<NWAStateID>) {
    let final_fix = compute_finality_fixpoint(states, filter);
    for &sid in filter {
        if let Some(add) = final_fix.get(&sid) {
            let st = &mut states[sid];
            st.final_weight = Some(st.final_weight.as_ref().map_or(add.clone(), |fw| fw | add));
        }
    }
}

pub fn remove_negative_transitions(states: &mut NWAStates, filter: &HashSet<NWAStateID>) {
    for &sid in filter { states[sid].transitions.retain(|&l, _| !is_neg(l)); }
}

pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let now = Instant::now();
    crate::debug!(6, "Resolving negative codes in DWA ({} states)...", dwa.states.len());
    let mut nwa = NWA::from_dwa(dwa);
    resolve_negative_codes_in_nwa(&mut nwa);
    let mut result = nwa.determinize_to_dwa();
    result.simplify();
    *dwa = result;
    crate::debug!(6, "resolve_negative_codes_in_dwa took: {:?}", now.elapsed());
}

fn compute_cancellations(states: &NWAStates, filter: &HashSet<NWAStateID>) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let n = states.len();
    // new_eps[from][to] = weight
    let mut new_eps: HashMap<NWAStateID, HashMap<NWAStateID, Weight>> = HashMap::new();
    // queries[b][(a, c)] = weight of cancellation path a --(-c)--> ... --> b
    let mut queries: HashMap<NWAStateID, HashMap<(NWAStateID, Label), Weight>> = HashMap::new();
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, Label, Weight)> = VecDeque::new();

    // Seed from negative transitions a --(-c)--> b
    for &a in filter {
        for (&label, targets) in &states[a].transitions {
            if !is_neg(label) { continue; }
            let c = label.wrapping_sub(Label::MIN);
            for &(b, ref w) in targets {
                if b >= n { continue; }
                let qw = queries.entry(b).or_default().entry((a, c)).or_default();
                let old = qw.clone(); *qw |= w;
                if *qw != old { worklist.push_back((b, a, c, qw.clone())); }
            }
        }
    }

    while let Some((s, a, c, w_as)) = worklist.pop_front() {
        // Propagate through known cancellation epsilons s -> target
        if let Some(eps_s) = new_eps.get(&s) {
            for (&target, eps_w) in eps_s {
                let prop = &w_as & eps_w;
                if prop.is_empty() { continue; }
                let qw = queries.entry(target).or_default().entry((a, c)).or_default();
                let old = qw.clone(); *qw |= &prop;
                if *qw != old { worklist.push_back((target, a, c, qw.clone())); }
            }
        }

        // Check for matches with positive transitions s --(+c)--> t or default
        let mut check = |t: NWAStateID, w_st: &Weight| {
            let new_w = &w_as & w_st;
            if new_w.is_empty() { return; }
            let eps_a = new_eps.entry(a).or_default();
            let eps_val = eps_a.entry(t).or_default();
            let old = eps_val.clone(); *eps_val |= &new_w;
            
            if *eps_val != old {
                // Epsilon grew: a --eps--> t. Propagate existing queries at 'a' across this new epsilon.
                if let Some(qs_a) = queries.get(&a) {
                    for (&(a_p, c_p), w_pa) in qs_a.clone() {
                        let prop = w_pa & &*eps_val;
                        if prop.is_empty() { continue; }
                        let qw = queries.entry(t).or_default().entry((a_p, c_p)).or_default();
                        let old_q = qw.clone(); *qw |= &prop;
                        if *qw != old_q { worklist.push_back((t, a_p, c_p, qw.clone())); }
                    }
                }
            }
        };

        if let Some(ts) = states[s].transitions.get(&c) { for &(t, ref w) in ts { if t < n { check(t, w); } } }
        if let Some(ts) = states[s].transitions.get(&DEFAULT_TRANSITION_SYMBOL) { for &(t, ref w) in ts { check(t, w); } }
        
        // Propagate through existing epsilons s --eps--> t
        for &(t, ref w) in &states[s].epsilons {
            if t >= n { continue; }
            let prop = &w_as & w;
            if prop.is_empty() { continue; }
            let qw = queries.entry(t).or_default().entry((a, c)).or_default();
            let old = qw.clone(); *qw |= &prop;
            if *qw != old { worklist.push_back((t, a, c, qw.clone())); }
        }
    }
    
    new_eps.into_iter().flat_map(|(f, ts)| ts.into_iter().map(move |(t, w)| (f, t, w))).collect()
}

fn compute_finality_fixpoint(states: &NWAStates, filter: &HashSet<NWAStateID>) -> HashMap<NWAStateID, Weight> {
    let n = states.len();
    if n == 0 || filter.is_empty() { return HashMap::new(); }

    // Reverse graph restricted to reachable subgraph from filter via eps/neg
    let mut preds: HashMap<NWAStateID, Vec<(NWAStateID, Weight)>> = HashMap::new();
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    let mut visited = vec![false; n];
    let mut reachable = Vec::new();

    for &s in filter { if s < n { visited[s] = true; queue.push_back(s); reachable.push(s); } }

    while let Some(u) = queue.pop_front() {
        let st = &states[u];
        // Epsilons allowed everywhere
        for &(v, ref w) in &st.epsilons {
            if v < n && !w.is_empty() {
                preds.entry(v).or_default().push((u, w.clone()));
                if !visited[v] { visited[v] = true; queue.push_back(v); reachable.push(v); }
            }
        }
        // Negatives only from filtered sources
        if filter.contains(&u) {
            for (&l, ts) in &st.transitions {
                if is_neg(l) {
                    for &(v, ref w) in ts {
                        if v < n && !w.is_empty() {
                            preds.entry(v).or_default().push((u, w.clone()));
                            if !visited[v] { visited[v] = true; queue.push_back(v); reachable.push(v); }
                        }
                    }
                }
            }
        }
    }

    let mut future: HashMap<NWAStateID, Weight> = HashMap::new();
    let mut work: VecDeque<NWAStateID> = VecDeque::new();

    for &s in &reachable {
        if let Some(fw) = &states[s].final_weight {
            if !fw.is_empty() { future.insert(s, fw.clone()); work.push_back(s); }
        }
    }

    while let Some(s) = work.pop_front() {
        let f_s = match future.get(&s) { Some(w) if !w.is_empty() => w.clone(), _ => continue };
        if let Some(ps) = preds.get(&s) {
            for (p, pw) in ps {
                let add = &f_s & pw;
                if add.is_empty() { continue; }
                let ent = future.entry(*p).or_default();
                let old = ent.clone(); *ent |= &add;
                if *ent != old { work.push_back(*p); }
            }
        }
    }

    future.into_iter().filter(|(k, _)| filter.contains(k)).collect()
}
