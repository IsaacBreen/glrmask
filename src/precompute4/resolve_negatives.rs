// src/precompute4/resolve_negatives.rs

use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::precompute4::weighted_automata::common::Label;
use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

type Code = Label;
type QueryKey = (NWAStateID, Code);

#[inline]
fn is_negative_symbol(label: Code) -> bool { label < 0 && label != DEFAULT_TRANSITION_SYMBOL }

fn pb(len: u64, msg: &str) -> Option<ProgressBar> {
    if !PROGRESS_BAR_ENABLED { return None; }
    let p = ProgressBar::new(len).with_style(ProgressStyle::default_bar().template("{spinner:.green} [{elapsed}] {bar:40.cyan/blue} {pos}/{len} {msg}").unwrap());
    p.set_message(msg.to_string()); Some(p)
}

pub fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    let p = pb(3, "Resolving negatives in NWA");
    let all: HashSet<NWAStateID> = (0..nwa.states.len()).collect();

    if let Some(ref p) = p { p.set_position(1); p.set_message("Cancellations"); }
    apply_cancellations(&mut nwa.states, &all);

    if let Some(ref p) = p { p.set_position(2); p.set_message("Finality"); }
    apply_finality_fixpoint(&mut nwa.states, &all);

    if let Some(ref p) = p { p.set_position(3); p.set_message("Removing negatives"); }
    remove_negative_transitions(&mut nwa.states, &all);
    
    if let Some(ref p) = p { p.finish_with_message("Done"); }
}

pub fn apply_cancellations(states: &mut NWAStates, filter: &HashSet<NWAStateID>) {
    for (u, v, w) in compute_cancellations(states, filter) { states.add_epsilon(u, v, w); }
}

pub fn apply_finality_fixpoint(states: &mut NWAStates, filter: &HashSet<NWAStateID>) {
    for (sid, w) in compute_finality_fixpoint(states, filter) {
        let st = &mut states[sid];
        st.final_weight = Some(st.final_weight.clone().map_or(w.clone(), |fw| fw | w));
    }
}

pub fn remove_negative_transitions(states: &mut NWAStates, filter: &HashSet<NWAStateID>) {
    for &sid in filter { states[sid].transitions.retain(|&l, _| !is_negative_symbol(l)); }
}

pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let now = Instant::now();
    let p = pb(4, "Resolving DWA negatives");
    
    if let Some(ref p) = p { p.set_position(1); p.set_message("DWA->NWA"); }
    let mut nwa = NWA::from_dwa(dwa);

    if let Some(ref p) = p { p.set_position(2); p.set_message("Resolving NWA"); }
    resolve_negative_codes_in_nwa(&mut nwa);

    if let Some(ref p) = p { p.set_position(3); p.set_message("Determinizing"); }
    let mut res = nwa.determinize_to_dwa();

    if let Some(ref p) = p { p.set_position(4); p.set_message("Simplifying"); }
    res.simplify();
    *dwa = res;

    if let Some(ref p) = p { p.finish_with_message("Done"); }
    crate::debug!(6, "resolve_negative_codes_in_dwa took: {:?}", now.elapsed());
}

fn compute_cancellations(states: &NWAStates, filter: &HashSet<NWAStateID>) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let n = states.len();
    let mut queries: HashMap<NWAStateID, HashMap<QueryKey, Weight>> = HashMap::new();
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, Code, Weight)> = VecDeque::new();
    let mut new_eps: HashMap<NWAStateID, HashMap<NWAStateID, Weight>> = HashMap::new();

    for &a in filter {
        for (&l, targets) in &states[a].transitions {
            if !is_negative_symbol(l) { continue; }
            let c = l.wrapping_sub(Code::MIN);
            for (b, w) in targets {
                if *b >= n { continue; }
                let qw = queries.entry(*b).or_default().entry((a, c)).or_default();
                if (w.clone() | &*qw) != *qw { *qw |= w; worklist.push_back((*b, a, c, qw.clone())); }
            }
        }
    }

    while let Some((s, a, c, w_as)) = worklist.pop_front() {
        if let Some(eps) = new_eps.get(&s) {
            for (&t, ew) in eps {
                let prop = &w_as & ew;
                if prop.is_empty() { continue; }
                let qw = queries.entry(t).or_default().entry((a, c)).or_default();
                if (prop.clone() | &*qw) != *qw { *qw |= &prop; worklist.push_back((t, a, c, qw.clone())); }
            }
        }

        let mut check = |t: NWAStateID, w: &Weight, wl: &mut VecDeque<_>| {
            let new_ew = &w_as & w;
            if new_ew.is_empty() { return; }
            let ew_entry = new_eps.entry(a).or_default().entry(t).or_default();
            if (new_ew.clone() | &*ew_entry) != *ew_entry {
                *ew_entry |= &new_ew;
                let updates: Vec<_> = if let Some(qa) = queries.get(&a) {
                    qa.iter()
                        .filter_map(|(&(ap, cp), wap)| {
                            let p = wap & &*ew_entry;
                            if !p.is_empty() { Some((ap, cp, p)) } else { None }
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                for (ap, cp, p) in updates {
                    let qw = queries.entry(t).or_default().entry((ap, cp)).or_default();
                    if (p.clone() | &*qw) != *qw {
                        *qw |= &p;
                        wl.push_back((t, ap, cp, qw.clone()));
                    }
                }
            }
        };

        if let Some(ts) = states[s].transitions.get(&c) { for (t, w) in ts { if *t < n { check(*t, w, &mut worklist); } } }
        if let Some(ts) = states[s].transitions.get(&DEFAULT_TRANSITION_SYMBOL) { for (t, w) in ts { check(*t, w, &mut worklist); } }
        for (t, w) in &states[s].epsilons {
            if *t >= n { continue; }
            let prop = &w_as & w;
            if prop.is_empty() { continue; }
            let qw = queries.entry(*t).or_default().entry((a, c)).or_default();
            if (prop.clone() | &*qw) != *qw { *qw |= &prop; worklist.push_back((*t, a, c, qw.clone())); }
        }
    }
    new_eps.into_iter().flat_map(|(u, m)| m.into_iter().map(move |(v, w)| (u, v, w))).collect()
}

fn compute_finality_fixpoint(states: &NWAStates, filter: &HashSet<NWAStateID>) -> HashMap<NWAStateID, Weight> {
    let n = states.len();
    if n == 0 || filter.is_empty() { return HashMap::new(); }

    #[derive(Clone, Copy)]
    enum Edge { Eps(NWAStateID, usize), Neg(NWAStateID, Code, usize) }
    let mut preds: HashMap<NWAStateID, Vec<Edge>> = HashMap::new();
    let mut visited = vec![false; n];
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    let mut reachable = Vec::new();

    for &a in filter { if a < n && !visited[a] { visited[a] = true; queue.push_back(a); reachable.push(a); } }

    while let Some(s) = queue.pop_front() {
        for (i, &(t, ref w)) in states[s].epsilons.iter().enumerate() {
            if t >= n || w.is_empty() { continue; }
            preds.entry(t).or_default().push(Edge::Eps(s, i));
            if !visited[t] { visited[t] = true; queue.push_back(t); reachable.push(t); }
        }
        if filter.contains(&s) {
            for (&l, ts) in &states[s].transitions {
                if !is_negative_symbol(l) { continue; }
                for (i, &(t, ref w)) in ts.iter().enumerate() {
                    if t >= n || w.is_empty() { continue; }
                    preds.entry(t).or_default().push(Edge::Neg(s, l, i));
                    if !visited[t] { visited[t] = true; queue.push_back(t); reachable.push(t); }
                }
            }
        }
    }

    let mut final_w: HashMap<NWAStateID, Weight> = HashMap::new();
    let mut wl: VecDeque<NWAStateID> = VecDeque::new();
    for &s in &reachable {
        if let Some(ref fw) = states[s].final_weight { if !fw.is_empty() { final_w.insert(s, fw.clone()); wl.push_back(s); } }
    }

    while let Some(s) = wl.pop_front() {
        let fs = final_w[&s].clone();
        if let Some(edges) = preds.get(&s) {
            for edge in edges {
                let (p, w) = match *edge {
                    Edge::Eps(u, i) => (u, &states[u].epsilons[i].1),
                    Edge::Neg(u, l, i) => (u, &states[u].transitions[&l][i].1),
                };
                let add = &fs & w;
                if add.is_empty() { continue; }
                let ent = final_w.entry(p).or_default();
                if (add.clone() | &*ent) != *ent { *ent |= &add; wl.push_back(p); }
            }
        }
    }
    final_w.into_iter().filter(|(k, _)| filter.contains(k)).collect()
}
