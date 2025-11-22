// src/precompute4/weighted_automata/determinization.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use super::common::{Label, NWAStateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWA, NWAStates};

type WeightedSubset = BTreeMap<NWAStateID, Weight>;
type ClosureMap = BTreeMap<NWAStateID, Weight>;

fn weight_union(a: Weight, b: &Weight) -> Weight { a | b.clone() }
fn weight_union_in_place(dst: &mut Weight, src: &Weight) { *dst |= src; }
fn weight_intersection(a: &Weight, b: &Weight) -> Weight { a & b }
fn is_zero(w: &Weight) -> bool { w.is_empty() }

fn epsilon_closure(nwa_states: &NWAStates, seed: &WeightedSubset) -> ClosureMap {
    let mut closure: ClosureMap = seed.clone();
    let mut queue: VecDeque<NWAStateID> = seed.keys().cloned().collect();

    while let Some(u) = queue.pop_front() {
        let uw = closure.get(&u).cloned().unwrap_or_else(Weight::zeros);
        if is_zero(&uw) || u >= nwa_states.len() { continue; }
        
        for (v, w_eps) in &nwa_states[u].epsilons {
            let cand = weight_intersection(&uw, w_eps);
            if is_zero(&cand) { continue; }
            let prev = closure.get(v).cloned().unwrap_or_else(Weight::zeros);
            let merged = weight_union(prev.clone(), &cand);
            if merged != prev {
                closure.insert(*v, merged);
                queue.push_back(*v);
            }
        }
    }
    closure
}

fn collect_labels(nwa_states: &NWAStates, closure: &ClosureMap) -> BTreeSet<Label> {
    let mut labels = BTreeSet::new();
    for (sid, cw) in closure {
        if !is_zero(cw) && *sid < nwa_states.len() {
            labels.extend(nwa_states[*sid].transitions.keys());
        }
    }
    labels
}

fn next_subset_for_label(nwa_states: &NWAStates, closure: &ClosureMap, ch: Label) -> WeightedSubset {
    let mut next = WeightedSubset::new();
    for (sid, cw) in closure {
        if !is_zero(cw) && *sid < nwa_states.len() {
            if let Some(targets) = nwa_states[*sid].transitions.get(&ch) {
                for (to, w_edge) in targets {
                    let cand = weight_intersection(cw, w_edge);
                    if !is_zero(&cand) {
                        next.entry(*to).and_modify(|w| weight_union_in_place(w, &cand)).or_insert(cand);
                    }
                }
            }
        }
    }
    next
}

struct Determinizer<'a> {
    nwa: &'a NWA,
    seen: HashMap<ClosureMap, usize>,
    closures: Vec<ClosureMap>,
    queue: VecDeque<usize>,
    dwa: DWA,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA) -> Self {
        Determinizer {
            nwa,
            seen: HashMap::new(),
            closures: Vec::new(),
            queue: VecDeque::new(),
            dwa: DWA::new(),
        }
    }

    fn register_state(&mut self, subset: WeightedSubset) -> usize {
        let closure = epsilon_closure(&self.nwa.states, &subset);
        if let Some(&id) = self.seen.get(&closure) { return id; }

        let id = self.dwa.add_state();
        let mut finalw = Weight::zeros();
        for (sid, cw) in &closure {
            if *sid < self.nwa.states.len() {
                if let Some(fw) = &self.nwa.states[*sid].final_weight {
                    finalw |= &(cw & fw);
                }
            }
        }
        if !finalw.is_empty() { let _ = self.dwa.set_final_weight(id, finalw); }

        self.seen.insert(closure.clone(), id);
        self.closures.push(closure);
        self.queue.push_back(id);
        id
    }

    fn run(&mut self) {
        let mut start_subset = WeightedSubset::new();
        for &s in &self.nwa.body.start_states {
            start_subset.insert(s, Weight::all());
        }
        self.dwa.body.start_state = self.register_state(start_subset);

        while let Some(sid) = self.queue.pop_front() {
            let closure = &self.closures[sid].clone();
            let labels = collect_labels(&self.nwa.states, closure);
            for ch in labels {
                let next = next_subset_for_label(&self.nwa.states, closure, ch);
                if !next.is_empty() {
                    let mut w_trans = Weight::zeros();
                    for w in next.values() { w_trans |= w; }
                    if !w_trans.is_empty() {
                        let to_id = self.register_state(next);
                        let _ = self.dwa.add_transition(sid, ch, to_id, w_trans);
                    }
                }
            }
        }
    }
}

impl NWA {
    pub fn determinize_to_dwa2(&self) -> DWA {
        let mut det = Determinizer::new(self);
        det.dwa.states.0.clear();
        det.run();
        det.dwa
    }
}
