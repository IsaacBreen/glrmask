// src/precompute4/weighted_automata/simplification.rs

use super::common::{BENCHMARK_DEBUG, Label, NWAStateID, StateID, Weight, OPTIMIZE_DEBUG};
use super::dwa::{DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWAStates, NWA};
use rustfst::algorithms::{minimize, minimize_with_config, MinimizeConfig};
use std::collections::{BTreeMap, HashMap, VecDeque};

const MAX_ITERS: usize = 1000;

#[derive(Clone, Debug)]
struct Partition { class: Vec<usize>, count: usize }
impl Partition { fn new(n: usize) -> Self { Self { class: vec![0; n], count: if n == 0 { 0 } else { 1 } } } }

fn minimize_partition<F, K>(n: usize, key_fn: F) -> Partition where F: Fn(usize, &[usize]) -> K, K: std::hash::Hash + Eq {
    if n == 0 { return Partition { class: vec![], count: 0 }; }
    let mut p = Partition::new(n);
    loop {
        let mut map = HashMap::new();
        let mut next = vec![0; n];
        let mut c = 0;
        for i in 0..n {
            let k = key_fn(i, &p.class);
            next[i] = *map.entry(k).or_insert_with(|| { let v = c; c += 1; v });
        }
        if next == p.class { p.count = c; return p; }
        p.class = next; p.count = c;
    }
}

impl DWA {
    pub fn simplify(&mut self) {
        if self.states.len() == 0 { return; }
        if OPTIMIZE_DEBUG { self.run_opt_exp(); return; }
        if BENCHMARK_DEBUG {
             let mut c = self.clone(); c.simplify_internal();
             let mut r = self.clone(); r.minimize_with_rustfst();
             if c.states.len() > r.states.len() { crate::debug!(5, "Internal simplify worse than rustfst: {} vs {}", c.states.len(), r.states.len()); }
             *self = c;
             return;
        }
        self.simplify_internal();
    }

    pub fn simplify_lightweight(&mut self) {
        for _ in 0..10 {
            if !(self.prune_dead_ends() | self.push_weights() | self.prune_unreachable()) { break; }
        }
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize(&mut fst).unwrap();
        *self = DWA::from_rustfst(&fst);
    }

    fn run_opt_exp(&mut self) { self.simplify_internal(); }

    fn simplify_internal(&mut self) -> bool {
        let mut changed = false;
        for _ in 0..MAX_ITERS {
            let c = self.prune_dead_ends() | self.minimize_states() | self.push_weights() | self.prune_unreachable();
            changed |= c;
            if !c { break; }
        }
        changed
    }

    fn push_weights(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 || self.body.start_state >= n { return false; }
        let mut changed = false;
        let mut preds = vec![vec![]; n];
        for (u, st) in self.states.0.iter().enumerate() { for (&l, &v) in &st.transitions { if v < n { preds[v].push((u, l)); } } }

        for v in 0..n {
            if v == self.body.start_state { continue; }
            if let Some(sw) = self.states[v].state_weight.take() {
                changed = true;
                for (u, l) in &preds[v] { if let Some(w) = self.states[*u].trans_weights.get_mut(l) { *w &= &sw; } }
            }
        }
        if let Some(sw) = self.states[self.body.start_state].state_weight.take() {
            changed = true;
            for st in &mut self.states.0 { if let Some(fw) = &mut st.final_weight { *fw &= &sw; } }
        }
        changed
    }

    fn minimize_states(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 { return false; }
        let p = minimize_partition(n, |i, cls| {
            let st = &self.states[i];
            let mut out: Vec<_> = st.transitions.iter().filter(|(_, &d)| !st.trans_weights.get(0).map_or(false, |w| w.is_empty())).map(|(&l, &d)| (l, cls[d], st.trans_weights[&l].clone())).collect();
            (st.final_weight.clone(), out)
        });
        if p.count >= n { return false; }
        
        let mut map = HashMap::new();
        let mut builders = vec![DWAState::default(); p.count];
        for i in 0..p.count { map.insert(i, i); } // direct map if sequential

        for i in 0..n {
            let c = p.class[i];
            let b = &mut builders[c];
            let st = &self.states[i];
            if let Some(fw) = &st.final_weight { b.final_weight = Some(b.final_weight.clone().map_or(fw.clone(), |x| x | fw)); }
            for (&l, &d) in &st.transitions {
                let w = st.trans_weights[&l].clone();
                if !w.is_empty() {
                    b.transitions.insert(l, p.class[d]); // Assuming deterministic, map to class
                    b.trans_weights.entry(l).and_modify(|x| *x |= &w).or_insert(w);
                }
            }
        }
        self.states = DWAStates(builders);
        self.body.start_state = p.class[self.body.start_state];
        true
    }

    pub fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        if self.body.start_state >= n { return self.reset_empty(); }

        let mut vis = vec![false; n];
        let mut q = VecDeque::from([self.body.start_state]);
        vis[self.body.start_state] = true;
        while let Some(u) = q.pop_front() {
            for &v in self.states[u].transitions.values() { if v < n && !vis[v] { vis[v] = true; q.push_back(v); } }
        }
        if vis.iter().all(|&b| b) { return false; }
        self.remap_states(&vis)
    }

    pub fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut live = vec![false; n];
        let mut rev = vec![vec![]; n];
        for u in 0..n { for &v in self.states[u].transitions.values() { if v < n { rev[v].push(u); } } }
        let mut q = VecDeque::new();
        for i in 0..n { if self.states[i].final_weight.as_ref().map_or(false, |w| !w.is_empty()) { live[i] = true; q.push_back(i); } }
        while let Some(v) = q.pop_front() { for &u in &rev[v] { if !live[u] { live[u] = true; q.push_back(u); } } }
        if self.body.start_state >= n || !live[self.body.start_state] { return self.reset_empty(); }
        if live.iter().all(|&b| b) { return false; }
        self.remap_states(&live)
    }

    fn reset_empty(&mut self) -> bool {
        if self.states.len() > 0 && self.states.0 != vec![DWAState::default()] {
            self.states = DWAStates::default(); self.states.add_state(); self.body.start_state = 0; true
        } else { false }
    }

    fn remap_states(&mut self, keep: &[bool]) -> bool {
        let n = self.states.len();
        let mut map = vec![0; n];
        let mut new_states = DWAStates(Vec::new());
        for i in 0..n { if keep[i] { map[i] = new_states.add_state(); new_states[map[i]] = self.states[i].clone(); } }
        for st in &mut new_states.0 {
            let mut trans = BTreeMap::new(); let mut ws = BTreeMap::new();
            for (&l, &d) in &st.transitions { if d < n && keep[d] { trans.insert(l, map[d]); ws.insert(l, st.trans_weights[&l].clone()); } }
            st.transitions = trans; st.trans_weights = ws;
        }
        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }
}

impl NWA {
    pub fn simplify(&mut self) {
        if self.states.len() == 0 { return; }
        if OPTIMIZE_DEBUG { self.run_opt_exp(); return; }
        self.simplify_internal();
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, MinimizeConfig::default().with_allow_nondet(true)).unwrap();
        *self = NWA::from_rustfst(&fst);
    }

    fn run_opt_exp(&mut self) { self.simplify_internal(); }

    fn simplify_internal(&mut self) -> bool {
        let mut changed = false;
        for _ in 0..MAX_ITERS {
            let c = self.prune_unreachable() | self.compress_transitions() | self.push_final_weights() | self.prune_dead_ends() | self.minimize_states();
            changed |= c;
            if !c { break; }
        }
        changed
    }

    fn minimize_states(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 { return false; }
        // Sig: final_weight, sorted list of (label/eps, dest_class, weight)
        let p = minimize_partition(n, |i, cls| {
            let st = &self.states[i];
            let mut edges = Vec::new();
            for &(t, ref w) in &st.epsilons { if !w.is_empty() { edges.push((0, 0, cls[t], w.clone())); } }
            for (&l, ts) in &st.transitions { for &(t, ref w) in ts { if !w.is_empty() { edges.push((1, l, cls[t], w.clone())); } } }
            edges.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
            // merge
            let mut merged = Vec::new();
            if !edges.is_empty() {
                let mut curr = edges[0].clone();
                for next in edges.into_iter().skip(1) {
                    if curr.0 == next.0 && curr.1 == next.1 && curr.2 == next.2 { curr.3 |= &next.3; }
                    else { merged.push(curr); curr = next; }
                }
                merged.push(curr);
            }
            (st.final_weight.clone(), merged)
        });
        if p.count >= n { return false; }

        let mut builders = vec![NWAState::default(); p.count];
        for i in 0..n {
            let c = p.class[i];
            let b = &mut builders[c];
            let st = &self.states[i];
            if let Some(fw) = &st.final_weight { b.final_weight = Some(b.final_weight.clone().map_or(fw.clone(), |x| x | fw)); }
            for &(t, ref w) in &st.epsilons { if !w.is_empty() { b.epsilons.push((p.class[t], w.clone())); } }
            for (&l, ts) in &st.transitions { for &(t, ref w) in ts { if !w.is_empty() { b.transitions.entry(l).or_default().push((p.class[t], w.clone())); } } }
        }
        // clean up builders
        for b in &mut builders {
            // compress
            let mut new_eps = BTreeMap::new();
            for (t, w) in &b.epsilons { new_eps.entry(*t).and_modify(|x| *x |= w).or_insert(w.clone()); }
            b.epsilons = new_eps.into_iter().collect();
            for ts in b.transitions.values_mut() {
                let mut m = BTreeMap::new();
                for (t, w) in ts.iter() { m.entry(*t).and_modify(|x| *x |= w).or_insert(w.clone()); }
                *ts = m.into_iter().collect();
            }
        }
        self.states = NWAStates(builders);
        self.body.start_states = self.body.start_states.iter().map(|s| p.class[*s]).collect::<std::collections::HashSet<_>>().into_iter().collect();
        true
    }

    fn compress_transitions(&mut self) -> bool {
        let mut changed = false;
        for st in &mut self.states.0 {
            let ne = st.epsilons.len();
            let mut em = BTreeMap::new();
            for (t, w) in &st.epsilons { if !w.is_empty() { em.entry(*t).and_modify(|x| *x |= w).or_insert(w.clone()); } }
            st.epsilons = em.into_iter().collect();
            if st.epsilons.len() != ne { changed = true; }

            let mut new_trans = BTreeMap::new();
            for (&l, ts) in &st.transitions {
                let nt = ts.len();
                let mut tm = BTreeMap::new();
                for (t, w) in ts { if !w.is_empty() { tm.entry(*t).and_modify(|x| *x |= w).or_insert(w.clone()); } }
                let vec: Vec<_> = tm.into_iter().collect();
                if vec.len() != nt { changed = true; }
                if !vec.is_empty() { new_trans.insert(l, vec); }
            }
            if new_trans.len() != st.transitions.len() { changed = true; }
            st.transitions = new_trans;
        }
        changed
    }

    fn push_final_weights(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut rev_eps = vec![vec![]; n];
        for (u, st) in self.states.0.iter().enumerate() { for &(v, ref w) in &st.epsilons { if v < n && !w.is_empty() { rev_eps[v].push((u, w.clone())); } } }
        
        let mut fws: Vec<_> = self.states.0.iter().map(|s| s.final_weight.clone().unwrap_or_else(Weight::zeros)).collect();
        let mut q: VecDeque<_> = (0..n).filter(|&i| !fws[i].is_empty()).collect();
        let mut changed = false;

        while let Some(v) = q.pop_front() {
            let w_v = fws[v].clone();
            for &(u, ref w_uv) in &rev_eps[v] {
                let add = &w_v & w_uv;
                if !add.is_empty() {
                     let nw = &fws[u] | &add;
                     if nw != fws[u] { fws[u] = nw; q.push_back(u); }
                }
            }
        }
        for i in 0..n {
            let nw = if fws[i].is_empty() { None } else { Some(fws[i].clone()) };
            if self.states[i].final_weight != nw { self.states[i].final_weight = nw; changed = true; }
        }
        changed
    }

    fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut vis = vec![false; n];
        let mut q = VecDeque::new();
        for &s in &self.body.start_states { if s < n && !vis[s] { vis[s] = true; q.push_back(s); } }
        while let Some(u) = q.pop_front() {
            for &(v, ref w) in &self.states[u].epsilons { if v < n && !vis[v] && !w.is_empty() { vis[v] = true; q.push_back(v); } }
            for ts in self.states[u].transitions.values() { for &(v, ref w) in ts { if v < n && !vis[v] && !w.is_empty() { vis[v] = true; q.push_back(v); } } }
        }
        if vis.iter().all(|&b| b) { return false; }
        self.remap_states(&vis)
    }

    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut live = vec![false; n];
        let mut rev = vec![vec![]; n];
        for u in 0..n {
            for &(v, ref w) in &self.states[u].epsilons { if v < n && !w.is_empty() { rev[v].push(u); } }
            for ts in self.states[u].transitions.values() { for &(v, ref w) in ts { if v < n && !w.is_empty() { rev[v].push(u); } } }
        }
        let mut q = VecDeque::new();
        for i in 0..n { if self.states[i].final_weight.as_ref().map_or(false, |w| !w.is_empty()) { live[i] = true; q.push_back(i); } }
        while let Some(v) = q.pop_front() { for &u in &rev[v] { if !live[u] { live[u] = true; q.push_back(u); } } }
        if !self.body.start_states.iter().any(|&s| s < n && live[s]) { self.states = NWAStates::default(); self.body.start_states.clear(); return true; }
        if live.iter().all(|&b| b) { return false; }
        self.remap_states(&live)
    }

    fn remap_states(&mut self, keep: &[bool]) -> bool {
        let n = self.states.len();
        let mut map = vec![0; n];
        let mut new_states = NWAStates(Vec::new());
        for i in 0..n { if keep[i] { map[i] = new_states.add_state(); new_states[map[i]] = self.states[i].clone(); } }
        for st in &mut new_states.0 {
            st.epsilons = st.epsilons.iter().filter(|(v, _)| *v < n && keep[*v]).map(|(v, w)| (map[*v], w.clone())).collect();
            let mut nt = BTreeMap::new();
            for (&l, ts) in &st.transitions {
                let v: Vec<_> = ts.iter().filter(|(v, _)| *v < n && keep[*v]).map(|(v, w)| (map[*v], w.clone())).collect();
                if !v.is_empty() { nt.insert(l, v); }
            }
            st.transitions = nt;
        }
        self.body.start_states = self.body.start_states.iter().filter(|&&s| s < n && keep[s]).map(|&s| map[s]).collect();
        self.states = new_states;
        true
    }
}
