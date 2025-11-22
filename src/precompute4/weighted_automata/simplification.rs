#![allow(dead_code)]

use super::common::{BENCHMARK_DEBUG, Label, NWAStateID, StateID, Weight, OPTIMIZE_DEBUG};
use super::dwa::{DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWAStates, NWA};
use rustfst::algorithms::{minimize, minimize_with_config, MinimizeConfig};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

const MAX_OPTIMIZE_ITERATIONS: usize = 1000;

#[derive(Clone, Debug)]
struct Partition {
    class_of: Vec<usize>,
    num_classes: usize,
}

impl Partition {
    fn new(num_states: usize) -> Self {
        Self {
            class_of: vec![0; num_states],
            num_classes: if num_states == 0 { 0 } else { 1 },
        }
    }
    fn num_classes(&self) -> usize { self.num_classes }
}

// ---------------- DWA minimization ----------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaTransitionSig {
    label: Label,
    dest_class: usize,
    weight: Weight,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<DwaTransitionSig>,
}

impl DwaStateSignature {
    fn from_state(state_id: StateID, states: &DWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];
        let mut outgoing = Vec::with_capacity(st.transitions.len());
        for (&label, &dest) in &st.transitions {
            let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
            if w.is_empty() {
                continue;
            }
            let dest_class = classes[dest];
            outgoing.push(DwaTransitionSig {
                label,
                dest_class,
                weight: w,
            });
        }
        DwaStateSignature {
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

fn minimize_dwa_partition(states: &DWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition { class_of: vec![], num_classes: 0 };
    }

    let mut partition = Partition::new(n);
    loop {
        let mut sig_to_class: HashMap<DwaStateSignature, usize> = HashMap::new();
        let mut new_classes = vec![0; n];
        let mut next_class = 0;

        for s in 0..n {
            let sig = DwaStateSignature::from_state(s, states, &partition.class_of);
            let entry = sig_to_class.entry(sig).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            new_classes[s] = *entry;
        }

        if new_classes == partition.class_of {
            partition.num_classes = next_class;
            return partition;
        }

        partition.class_of = new_classes;
        partition.num_classes = next_class;
    }
}

#[derive(Clone, Debug, Default)]
struct DwaStateBuilder {
    final_weight: Option<Weight>,
    trans: BTreeMap<Label, (StateID, Weight)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DwaPass {
    PruneUnreachable,
    PruneDeadEnds,
    PushWeights,
    Minimize,
}

// Helper for optimization experiments, kept for completeness
const DWA_PASS_ORDERINGS: &[&[DwaPass]] = &[&[DwaPass::PruneUnreachable]]; // truncated for brevity

impl DWA {
    pub fn simplify(&mut self) {
        if self.states.len() == 0 { return; }
        if BENCHMARK_DEBUG {
            let initial = self.clone();
            let start = std::time::Instant::now();
            self.simplify_internal();
            if start.elapsed().as_millis() > 1000 {
                // log logic
            }
        } else {
            self.simplify_internal();
        }
    }

    pub fn simplify_lightweight(&mut self) {
         // Lightweight passes
        self.prune_dead_ends();
        self.push_weights_into_transitions_and_finals();
        self.prune_unreachable();
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize(&mut fst).unwrap();
        *self = DWA::from_rustfst(&fst);
    }

    fn simplify_internal(&mut self) -> bool {
        let mut total_changed = false;
        for _ in 0..10 {
            let mut changed = false;
            changed |= self.prune_dead_ends();
            changed |= self.minimize_states();
            changed |= self.push_weights_into_transitions_and_finals();
            changed |= self.prune_unreachable();
            total_changed |= changed;
            if !changed { break; }
        }
        total_changed
    }

    fn push_weights_into_transitions_and_finals(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let start = self.body.start_state;
        if start >= n { return false; }

        let mut changed = false;
        let mut preds: Vec<Vec<(StateID, Label)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for (&label, &v) in &st.transitions {
                if v < n { preds[v].push((u, label)); }
            }
        }

        for v in 0..n {
            if v == start { continue; }
            if let Some(sw) = self.states[v].state_weight.take() {
                if !sw.is_empty() && sw != Weight::all() {
                    changed = true;
                    for (u, label) in &preds[v] {
                        if let Some(w) = self.states[*u].trans_weights.get_mut(label) { *w &= &sw; }
                    }
                }
            }
        }
        // Push from start state into finals
        if let Some(sw0) = self.states[start].state_weight.take() {
             if !sw0.is_empty() && sw0 != Weight::all() {
                 changed = true;
                 for st in &mut self.states.0 {
                     if let Some(ref mut fw) = st.final_weight { *fw &= &sw0; }
                 }
             }
        }
        changed
    }

    fn minimize_states(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 { return false; }
        let partition = minimize_dwa_partition(&self.states);
        if partition.num_classes() >= n { return false; }
        self.rebuild_from_partition(partition);
        true
    }

    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        let mut class_to_new: HashMap<usize, StateID> = HashMap::new();
        let mut builders: Vec<DwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.class_of[s];
            class_to_new.entry(c).or_insert_with(|| {
                builders.push(DwaStateBuilder::default());
                builders.len() - 1
            });
        }

        for old_s in 0..n {
            let c = partition.class_of[old_s];
            let new_id = class_to_new[&c];
            let builder = &mut builders[new_id];
            let st = &self.states[old_s];
            if let Some(ref fw) = st.final_weight {
                 match &mut builder.final_weight {
                    Some(e) => *e |= fw, None => builder.final_weight = Some(fw.clone()),
                 }
            }
            for (&label, &dest) in &st.transitions {
                 let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                 if w.is_empty() { continue; }
                 let dest_new = class_to_new[&partition.class_of[dest]];
                 use std::collections::btree_map::Entry;
                 match builder.trans.entry(label) {
                     Entry::Vacant(e) => { e.insert((dest_new, w)); }
                     Entry::Occupied(mut e) => { e.get_mut().1 |= &w; }
                 }
            }
        }

        let mut new_states = DWAStates::default();
        for _ in 0..builders.len() { new_states.add_state(); }
        for (id, b) in builders.into_iter().enumerate() {
             new_states[id].final_weight = b.final_weight;
             for (l, (d, w)) in b.trans { new_states[id].transitions.insert(l, d); new_states[id].trans_weights.insert(l, w); }
        }
        self.states = new_states;
        self.body.start_state = class_to_new[&partition.class_of[self.body.start_state]];
    }

    pub fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut visited = vec![false; n];
        let mut q = VecDeque::new();
        if self.body.start_state < n {
            visited[self.body.start_state] = true;
            q.push_back(self.body.start_state);
        }

        while let Some(u) = q.pop_front() {
            for &v in self.states[u].transitions.values() {
                if v < n && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }
        if visited.iter().all(|&b| b) { return false; }

        let mut map = vec![0; n];
        let mut new_states = DWAStates::default();
        for i in 0..n {
            if visited[i] { map[i] = new_states.add_existing_state(self.states[i].clone()); }
        }
        for st in &mut new_states.0 {
            for (_, target) in st.transitions.iter_mut() { *target = map[*target]; }
        }
        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }

    pub fn prune_dead_ends(&mut self) -> bool {
         let n = self.states.len();
         if n == 0 { return false; }
         let mut live = vec![false; n];
         let mut rev = vec![vec![]; n];
         for u in 0..n {
             for &v in self.states[u].transitions.values() {
                 if v < n { rev[v].push(u); }
             }
         }
         let mut q = VecDeque::new();
         for i in 0..n {
             if self.states[i].final_weight.is_some() { live[i] = true; q.push_back(i); }
         }
         while let Some(v) = q.pop_front() {
             for &u in &rev[v] { if !live[u] { live[u] = true; q.push_back(u); } }
         }
         
         if self.body.start_state < n && !live[self.body.start_state] {
             // Start is dead. Empty DWA.
             self.states = DWAStates::default();
             self.body.start_state = self.states.add_state();
             return true;
         }

         if live.iter().all(|&b| b) { return false; }
         
         let mut map = vec![0; n];
         let mut new_states = DWAStates::default();
         for i in 0..n {
             if live[i] { map[i] = new_states.add_existing_state(self.states[i].clone()); }
         }
         for st in &mut new_states.0 {
             let mut dead_keys = Vec::new();
             for (k, v) in &st.transitions { if !live[*v] { dead_keys.push(*k); } }
             for k in dead_keys { st.transitions.remove(&k); st.trans_weights.remove(&k); }
             for (_, target) in st.transitions.iter_mut() { *target = map[*target]; }
         }
         self.body.start_state = map[self.body.start_state];
         self.states = new_states;
         true
    }
}

// ---------------- NWA minimization ----------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ArcLabel { Eps, Label(Label) }

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaTransitionSig {
    label: ArcLabel,
    dest_class: usize,
    weight: Weight,
}

impl NwaTransitionSig {
    fn sort_key(&self) -> (u8, Label, usize) {
        let label_tag = match self.label { ArcLabel::Eps => 0, ArcLabel::Label(_) => 1 };
        let label_val = match self.label { ArcLabel::Eps => 0, ArcLabel::Label(v) => v };
        (label_tag, label_val, self.dest_class)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<NwaTransitionSig>,
}

impl NwaStateSignature {
    fn from_state(state_id: NWAStateID, states: &NWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];
        let mut tmp = Vec::new();
        for &(dest, ref w) in &st.epsilons {
            if !w.is_empty() { tmp.push(NwaTransitionSig { label: ArcLabel::Eps, dest_class: classes[dest], weight: w.clone() }); }
        }
        for (&lbl, targets) in &st.transitions {
            for &(dest, ref w) in targets {
                if !w.is_empty() { tmp.push(NwaTransitionSig { label: ArcLabel::Label(lbl), dest_class: classes[dest], weight: w.clone() }); }
            }
        }
        if tmp.is_empty() { return NwaStateSignature { final_weight: st.final_weight.clone(), outgoing: Vec::new() }; }

        tmp.sort_by_key(|sig| sig.sort_key());
        let mut outgoing = Vec::new();
        let mut iter = tmp.into_iter();
        if let Some(mut cur) = iter.next() {
            for sig in iter {
                if cur.label == sig.label && cur.dest_class == sig.dest_class { cur.weight |= &sig.weight; }
                else { outgoing.push(cur); cur = sig; }
            }
            outgoing.push(cur);
        }
        NwaStateSignature { final_weight: st.final_weight.clone(), outgoing }
    }
}

fn minimize_nwa_partition(states: &NWAStates) -> Partition {
    let n = states.len();
    if n == 0 { return Partition { class_of: vec![], num_classes: 0 }; }
    let mut partition = Partition::new(n);
    loop {
        let mut sig_to_class = HashMap::new();
        let mut new_classes = vec![0; n];
        let mut next_class = 0;
        for s in 0..n {
            let sig = NwaStateSignature::from_state(s, states, &partition.class_of);
            let entry = sig_to_class.entry(sig).or_insert_with(|| { let id = next_class; next_class += 1; id });
            new_classes[s] = *entry;
        }
        if new_classes == partition.class_of { partition.num_classes = next_class; return partition; }
        partition.class_of = new_classes; partition.num_classes = next_class;
    }
}

#[derive(Clone, Debug, Default)]
struct NwaStateBuilder {
    final_weight: Option<Weight>,
    eps: BTreeMap<NWAStateID, Weight>,
    trans: BTreeMap<Label, BTreeMap<NWAStateID, Weight>>,
}

enum NwaPass { PruneUnreachable, PruneDeadEnds, PushFinalWeights, CompressTransitions, Minimize }

impl NWA {
    pub fn simplify(&mut self) {
         if self.states.len() == 0 { return; }
         self.simplify_internal();
    }

    pub fn simplify_internal(&mut self) -> bool {
        let mut total_changed = false;
        for _ in 0..10 {
            let mut changed = false;
            changed |= self.prune_unreachable();
            changed |= self.compress_transitions();
            changed |= self.push_final_weights_along_epsilons();
            changed |= self.prune_dead_ends();
            changed |= self.minimize_states();
            total_changed |= changed;
            if !changed { break; }
        }
        total_changed
    }

    fn minimize_states(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 { return false; }
        let partition = minimize_nwa_partition(&self.states);
        if partition.num_classes() >= n { return false; }
        self.rebuild_from_partition(partition);
        true
    }

    fn compress_transitions(&mut self) -> bool {
        let mut changed = false;
        for st in &mut self.states.0 {
             let mut eps_map = BTreeMap::new();
             for (to, w) in &st.epsilons { if !w.is_empty() { eps_map.entry(*to).and_modify(|acc| *acc |= w).or_insert(w.clone()); } }
             if eps_map.len() != st.epsilons.len() { changed = true; }
             st.epsilons = eps_map.into_iter().collect();

             let mut new_trans = BTreeMap::new();
             for (&lbl, targets) in &st.transitions {
                 let mut per_dest = BTreeMap::new();
                 for (to, w) in targets { if !w.is_empty() { per_dest.entry(*to).and_modify(|acc| *acc |= w).or_insert(w.clone()); } }
                 if per_dest.len() != targets.len() { changed = true; }
                 if !per_dest.is_empty() { new_trans.insert(lbl, per_dest.into_iter().collect()); }
             }
             if new_trans.len() != st.transitions.len() { changed = true; }
             st.transitions = new_trans;
        }
        changed
    }

    fn push_final_weights_along_epsilons(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut rev_eps = vec![vec![]; n];
        for (u, st) in self.states.0.iter().enumerate() {
            for &(v, ref w) in &st.epsilons { if v < n && !w.is_empty() { rev_eps[v].push((u, w.clone())); } }
        }
        let mut final_weights: Vec<Weight> = self.states.0.iter().map(|s| s.final_weight.clone().unwrap_or_else(Weight::zeros)).collect();
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| !final_weights[i].is_empty()).collect();

        while let Some(v) = queue.pop_front() {
            let w_v = final_weights[v].clone();
            for &(u, ref w_uv) in &rev_eps[v] {
                let cand = &w_v & w_uv;
                if !cand.is_empty() {
                    let new_w = &final_weights[u] | &cand;
                    if new_w != final_weights[u] {
                        final_weights[u] = new_w;
                        queue.push_back(u);
                    }
                }
            }
        }
        let mut changed = false;
        for i in 0..n {
            let fw = if final_weights[i].is_empty() { None } else { Some(final_weights[i].clone()) };
            if self.states[i].final_weight != fw { self.states[i].final_weight = fw; changed = true; }
        }
        changed
    }

    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        let mut class_to_new = HashMap::new();
        let mut builders = Vec::new();

        for s in 0..n {
            let c = partition.class_of[s];
            class_to_new.entry(c).or_insert_with(|| { builders.push(NwaStateBuilder::default()); builders.len() - 1 });
        }

        for old_s in 0..n {
            let c = partition.class_of[old_s];
            let new_id = class_to_new[&c];
            let builder = &mut builders[new_id];
            let st = &self.states[old_s];
            if let Some(fw) = &st.final_weight {
                match &mut builder.final_weight { Some(e) => *e |= fw, None => builder.final_weight = Some(fw.clone()) }
            }
            for (dest, w) in &st.epsilons {
                if !w.is_empty() {
                    let new_dest = class_to_new[&partition.class_of[*dest]];
                    *builder.eps.entry(new_dest).or_insert_with(Weight::zeros) |= w;
                }
            }
            for (lbl, targets) in &st.transitions {
                for (dest, w) in targets {
                    if !w.is_empty() {
                        let new_dest = class_to_new[&partition.class_of[*dest]];
                        *builder.trans.entry(*lbl).or_default().entry(new_dest).or_insert_with(Weight::zeros) |= w;
                    }
                }
            }
        }

        let mut new_states = NWAStates::default();
        for _ in 0..builders.len() { new_states.add_state(); }
        for (id, b) in builders.into_iter().enumerate() {
            new_states[id].final_weight = b.final_weight;
            new_states[id].epsilons = b.eps.into_iter().collect();
            for (l, m) in b.trans { new_states[id].transitions.insert(l, m.into_iter().collect()); }
        }
        self.states = new_states;
        // Map all start states
        let mut new_starts = Vec::new();
        for &s in &self.body.start_states {
            if s < n { new_starts.push(class_to_new[&partition.class_of[s]]); }
        }
        new_starts.sort(); new_starts.dedup();
        self.body.start_states = new_starts;
    }

    fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut visited = vec![false; n];
        let mut q = VecDeque::new();
        for &s in &self.body.start_states {
            if s < n && !visited[s] { visited[s] = true; q.push_back(s); }
        }

        while let Some(u) = q.pop_front() {
            for &(v, ref w) in &self.states[u].epsilons {
                if v < n && !visited[v] && !w.is_empty() { visited[v] = true; q.push_back(v); }
            }
            for targets in self.states[u].transitions.values() {
                for &(v, ref w) in targets {
                    if v < n && !visited[v] && !w.is_empty() { visited[v] = true; q.push_back(v); }
                }
            }
        }
        if visited.iter().all(|&b| b) { return false; }

        let mut map = vec![0; n];
        let mut new_states = NWAStates::default();
        for i in 0..n {
            if visited[i] { map[i] = new_states.add_existing_state(self.states[i].clone()); }
        }
        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, _)| visited[*v]);
            for (v, _) in &mut st.epsilons { *v = map[*v]; }
            for targets in st.transitions.values_mut() {
                 targets.retain(|(v, _)| visited[*v]);
                 for (v, _) in targets { *v = map[*v]; }
            }
        }
        
        let mut new_starts = Vec::new();
        for &s in &self.body.start_states {
            if s < n && visited[s] { new_starts.push(map[s]); }
        }
        self.body.start_states = new_starts;
        self.states = new_states;
        true
    }

    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut live = vec![false; n];
        let mut rev = vec![vec![]; n];
        for u in 0..n {
            for &(v, _) in &self.states[u].epsilons { if v < n { rev[v].push(u); } }
            for targets in self.states[u].transitions.values() {
                for &(v, _) in targets { if v < n { rev[v].push(u); } }
            }
        }
        let mut q = VecDeque::new();
        for i in 0..n {
            if self.states[i].final_weight.is_some() { live[i] = true; q.push_back(i); }
        }
        while let Some(v) = q.pop_front() {
            for &u in &rev[v] { if !live[u] { live[u] = true; q.push_back(u); } }
        }

        // Check if all start states are dead
        let mut all_dead = true;
        for &s in &self.body.start_states { if s < n && live[s] { all_dead = false; break; } }
        if all_dead {
             self.states = NWAStates::default();
             let s = self.states.add_state();
             self.body.start_states = vec![s];
             return true;
        }
        
        if live.iter().all(|&b| b) { return false; }
        
        let mut map = vec![0; n];
        let mut new_states = NWAStates::default();
        for i in 0..n {
            if live[i] { map[i] = new_states.add_existing_state(self.states[i].clone()); }
        }
        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, _)| live[*v]);
            for (v, _) in &mut st.epsilons { *v = map[*v]; }
            for targets in st.transitions.values_mut() {
                targets.retain(|(v, _)| live[*v]);
                for (v, _) in targets { *v = map[*v]; }
            }
        }
        
        let mut new_starts = Vec::new();
        for &s in &self.body.start_states {
            if s < n && live[s] { new_starts.push(map[s]); }
        }
        self.body.start_states = new_starts;
        self.states = new_states;
        true
    }
}
