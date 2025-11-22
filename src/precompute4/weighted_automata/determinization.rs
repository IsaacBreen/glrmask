// src/precompute4/weighted_automata/determinization.rs

use chrono::Local;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use super::common::{DETERMINIZE_DEBUG, Label, NWAStateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWA, NWAStates};
use crate::precompute4::test_weighted_automata;

type WeightedSubset = BTreeMap<NWAStateID, Weight>;
type ClosureMap = BTreeMap<NWAStateID, Weight>;

fn weight_union(a: Weight, b: &Weight) -> Weight { a | b.clone() }
fn weight_intersection(a: &Weight, b: &Weight) -> Weight { a & b }

fn epsilon_closure(states: &NWAStates, seed: &WeightedSubset) -> ClosureMap {
    let mut closure = ClosureMap::new();
    let mut q = VecDeque::new();
    for (sid, w) in seed {
        if !w.is_empty() {
            let prev = closure.get(sid).cloned().unwrap_or_else(Weight::zeros);
            let neww = weight_union(prev.clone(), w);
            if neww != prev { closure.insert(*sid, neww); q.push_back(*sid); }
        }
    }
    while let Some(u) = q.pop_front() {
        let uw = closure.get(&u).unwrap().clone();
        for (v, w_eps) in &states[u].epsilons {
            let cand = weight_intersection(&uw, w_eps);
            if !cand.is_empty() {
                let prev = closure.get(v).cloned().unwrap_or_else(Weight::zeros);
                let merged = weight_union(prev.clone(), &cand);
                if merged != prev { closure.insert(*v, merged); q.push_back(*v); }
            }
        }
    }
    closure
}

struct Determinizer<'a> {
    nwa: &'a NWA,
    seen: HashMap<ClosureMap, usize>,
    closures: Vec<ClosureMap>,
    queue: VecDeque<usize>,
    dwa: DWA,
    mp: Option<MultiProgress>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA, mp: Option<MultiProgress>) -> Self {
        let mut dwa = DWA::new(); dwa.states.0.clear();
        Self { nwa, seen: HashMap::new(), closures: Vec::new(), queue: VecDeque::new(), dwa, mp }
    }

    fn register(&mut self, subset: WeightedSubset) -> usize {
        let closure = epsilon_closure(&self.nwa.states, &subset);
        if let Some(&id) = self.seen.get(&closure) { return id; }
        let id = self.dwa.add_state();
        let mut fw = Weight::zeros();
        for (sid, cw) in &closure {
            if let Some(w) = &self.nwa.states[*sid].final_weight { fw |= &(cw & w); }
        }
        if !fw.is_empty() { self.dwa.set_final_weight(id, fw).unwrap(); }
        self.seen.insert(closure.clone(), id);
        self.closures.push(closure);
        self.queue.push_back(id);
        id
    }

    fn expand(&mut self, sid: usize) {
        let closure = &self.closures[sid];
        let labels: BTreeSet<Label> = closure.keys().flat_map(|&s| self.nwa.states[s].transitions.keys().cloned()).collect();
        
        let pb = self.mp.as_ref().map(|m| m.add(ProgressBar::new(labels.len() as u64).with_message(format!("State {}", sid))));

        for ch in labels {
            let mut next = WeightedSubset::new();
            for (u, cw) in closure {
                if let Some(ts) = self.nwa.states[*u].transitions.get(&ch) {
                    for (v, w) in ts {
                        let cand = weight_intersection(cw, w);
                        if !cand.is_empty() { next.entry(*v).and_modify(|x| *x |= &cand).or_insert(cand); }
                    }
                }
            }
            if !next.is_empty() {
                let w_ch = next.values().fold(Weight::zeros(), |acc, x| acc | x.clone());
                let to = self.register(next);
                self.dwa.add_transition(sid, ch, to, w_ch).ok();
            }
            if let Some(p) = &pb { p.inc(1); }
        }
        if let Some(p) = pb { p.finish_and_clear(); }
    }
}

impl NWA {
    pub fn determinize_to_dwa2(&self) -> DWA {
        // try fast path
        if self.body.start_states.len() == 1 {
            let start = self.body.start_states[0];
            if start < self.states.len() && self.states[start].transitions.is_empty() {
                let seed = WeightedSubset::from([(start, Weight::all())]);
                let closure = epsilon_closure(&self.states, &seed);
                let mut valid = true;
                let mut comps = Vec::new();
                for (s, w) in &closure {
                    if *s == start || w.is_empty() { continue; }
                    let st = &self.states[*s];
                    if !st.epsilons.is_empty() || st.transitions.values().flatten().any(|(t,_)| *t != *s) { valid = false; break; }
                    if let Some(fw) = &st.final_weight { let b = w & fw; if !b.is_empty() { comps.push((*s, b)); } }
                }
                if valid && !comps.is_empty() {
                    // pairwise check
                    for i in 0..comps.len() { for j in i+1..comps.len() { if !(comps[i].1.clone() & &comps[j].1).is_empty() { valid = false; break; } } }
                    if valid {
                        let mut dwa = DWA::new();
                        let s0 = dwa.body.start_state;
                        let mut fw = Weight::zeros();
                        for (_, b) in &comps { fw |= b; }
                        if !fw.is_empty() { dwa.set_final_weight(s0, fw).unwrap(); }
                        let mut trans: BTreeMap<Label, Weight> = BTreeMap::new();
                        for (s, b) in comps {
                            for (l, ts) in &self.states[s].transitions {
                                let mut uni = Weight::zeros(); for (_, w) in ts { uni |= w; }
                                let c = &b & &uni;
                                if !c.is_empty() { trans.entry(*l).and_modify(|x| *x |= &c).or_insert(c); }
                            }
                        }
                        for (l, w) in trans { dwa.add_transition(s0, l, s0, w).unwrap(); }
                        return dwa;
                    }
                }
            }
        }

        let show_pb = self.states.len() > 10000;
        let mp = if show_pb { Some(MultiProgress::new()) } else { None };
        let main_pb = mp.as_ref().map(|m| m.add(ProgressBar::new(0).with_message("Determinizing")));
        
        let mut det = Determinizer::new(self, mp);
        let mut start_sub = WeightedSubset::new();
        for &s in &self.body.start_states { start_sub.insert(s, Weight::all()); }
        det.dwa.body.start_state = det.register(start_sub);

        let mut proc = 0;
        while let Some(id) = det.queue.pop_front() {
            if det.seen.len() > 1_000_000 { panic!("DWA state explosion"); }
            if let Some(p) = &main_pb { p.set_length(det.seen.len() as u64); p.set_position(proc); }
            det.expand(id);
            proc += 1;
        }
        if let Some(p) = main_pb { p.finish_with_message("Done"); }
        
        let res = det.dwa;
        if DETERMINIZE_DEBUG {
             test_weighted_automata::stochastic_equivalence_test(res.clone(), self.determinize_to_dwa_with_rustfst());
        }
        res
    }
}
