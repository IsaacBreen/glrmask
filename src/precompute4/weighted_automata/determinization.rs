use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use super::common::{Label, NWAStateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWA, NWAStates};

type WeightedSubset = BTreeMap<NWAStateID, Weight>;

fn eps_closure(states: &NWAStates, seed: &WeightedSubset) -> WeightedSubset {
    let mut closure = WeightedSubset::new();
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    for (sid, w) in seed { 
        if !w.is_empty() { closure.insert(*sid, w.clone()); queue.push_back(*sid); }
    }
    while let Some(u) = queue.pop_front() {
        let uw = closure[&u].clone();
        for (v, w_eps) in &states[u].epsilons {
            let cand = &uw & w_eps;
            if !cand.is_empty() {
                let ent = closure.entry(*v).or_default();
                let old = ent.clone(); *ent |= &cand;
                if *ent != old { queue.push_back(*v); }
            }
        }
    }
    closure
}

fn next_subset(states: &NWAStates, closure: &WeightedSubset, l: Label) -> WeightedSubset {
    let mut next = WeightedSubset::new();
    for (sid, cw) in closure {
        if let Some(ts) = states[*sid].transitions.get(&l) {
            for (to, w_edge) in ts {
                let cand = cw & w_edge;
                if !cand.is_empty() { *next.entry(*to).or_default() |= &cand; }
            }
        }
    }
    next
}

struct Determinizer<'a> {
    nwa: &'a NWA,
    seen: HashMap<WeightedSubset, usize>,
    queue: VecDeque<usize>,
    closures: Vec<WeightedSubset>,
    dwa: DWA,
    pb: Option<ProgressBar>,
}

impl<'a> Determinizer<'a> {
    fn register(&mut self, subset: WeightedSubset) -> usize {
        let closure = eps_closure(&self.nwa.states, &subset);
        if let Some(&id) = self.seen.get(&closure) { return id; }
        
        let id = self.dwa.add_state();
        let mut finalw = Weight::zeros();
        for (sid, cw) in &closure {
            if let Some(fw) = &self.nwa.states[*sid].final_weight { finalw |= &(cw & fw); }
        }
        if !finalw.is_empty() { self.dwa.set_final_weight(id, finalw).unwrap(); }

        self.seen.insert(closure.clone(), id);
        self.closures.push(closure);
        self.queue.push_back(id);
        id
    }

    fn expand(&mut self, id: usize) {
        let closure = &self.closures[id];
        let labels: BTreeSet<Label> = closure.keys().flat_map(|sid| self.nwa.states[*sid].transitions.keys().cloned()).collect();
        if let Some(pb) = &self.pb { pb.set_message(format!("St: {}, Tr: {}", id, labels.len())); }

        for l in labels {
            let sub = next_subset(&self.nwa.states, closure, l);
            let w_union = sub.values().fold(Weight::zeros(), |acc, w| acc | w);
            if !sub.is_empty() && !w_union.is_empty() {
                let target = self.register(sub);
                let _ = self.dwa.add_transition(id, l, target, w_union);
            }
        }
    }
}

impl NWA {
    pub fn determinize_to_dwa2(&self) -> DWA {
        if self.states.len() == 0 || self.body.start_states.is_empty() { return DWA::new(); }
        
        // Fast path for single-state loops (common in GLR)
        if self.body.start_states.len() == 1 {
             // ... simplified singleton logic could go here, skipping for conciseness as general path works
        }

        let mp = if self.states.len() > 5000 { Some(MultiProgress::new()) } else { None };
        let pb = mp.as_ref().map(|m| m.add(ProgressBar::new_spinner()));
        
        let mut dwa = DWA::new(); dwa.states.0.clear();
        let mut det = Determinizer { nwa: self, seen: HashMap::new(), queue: VecDeque::new(), closures: Vec::new(), dwa, pb };
        
        let mut start_sub = WeightedSubset::new();
        for &s in &self.body.start_states { start_sub.insert(s, Weight::all()); }
        det.dwa.body.start_state = det.register(start_sub);

        while let Some(id) = det.queue.pop_front() {
            det.expand(id);
            if let Some(pb) = &det.pb { 
                if id % 100 == 0 { pb.set_message(format!("States: {}", det.seen.len())); }
            }
        }
        det.dwa
    }
}
