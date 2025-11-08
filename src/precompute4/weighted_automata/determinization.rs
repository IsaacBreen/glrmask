#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::SimpleBitset as Weight;
use super::dwa::{DWAState, DWAStates, DWA, DWABody};
use super::nwa::{NWA, NWAStates};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use range_set_blaze::RangeSetBlaze;

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::RangeInclusive;
use std::time::Instant;

/// This determinization follows a principled construction using an on-the-fly product:
/// 1) Extract a minimal disjoint partition of the weight space (atoms).
/// 2) For each atom, build and minimize a component DFA.
/// 3) Construct the final DWA by exploring the reachable state space of the product of these
///    component DFAs. A state in the DWA corresponds to a tuple of states `(q_0, ..., q_{k-1})`
///    from the component DFAs.
/// 4) This exploration is done "on-the-fly", creating DWA states and their weighted transitions
///    directly, avoiding a bloated intermediate unweighted automaton.
/// 5) The resulting DWA is then passed to simplification passes, which can further reduce it by
///    merging structurally equivalent states.
impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now_total = Instant::now();

        let nwa = self.clone();
        debug_log(3, || format!("Starting determinization for NWA with {} states", nwa.states.len()));

        let fut = nwa.compute_future_weights();
        let atoms = WeightPartition::from_nwa(&nwa);
        debug_log(4, || format!("Built weight partition with {} atoms in {:?}", atoms.intervals.len(), now_total.elapsed()));

        if atoms.intervals.is_empty() {
            return DWA::new();
        }

        let sigma = Alphabet::from_nwa_with_future(&nwa, &fut);
        debug_log(4, || format!("Built alphabet with {} labels", sigma.labels.len()));

        // 1. Build and minimize a DFA for each atom.
        let pb_atoms = if PROGRESS_BAR_ENABLED {
            Some(
                ProgressBar::new(atoms.intervals.len() as u64).with_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (per-atom DFAs)")
                        .unwrap(),
                ),
            )
        } else { None };

        let mut comp_dfas: Vec<DetDFA> = Vec::with_capacity(atoms.intervals.len());
        for (i, atom) in atoms.intervals.iter().enumerate() {
            let nfa = PerAtomNFA::from_nwa(&nwa.states, nwa.body.start_state, &sigma, atom, &fut);
            let mut dfa = nfa.determinize(&sigma);
            dfa.minimize(&sigma);
            crate::debug!(5, "Atom {}: interval={:?}, DFA states={}", i, atom, dfa.n_states);
            comp_dfas.push(dfa);
            if let Some(p) = &pb_atoms { p.inc(1); }
        }
        if let Some(p) = pb_atoms { p.finish_with_message("Per-atom DFAs built & minimized"); }

        // 2. Build the DWA directly from the components via on-the-fly product construction.
        let now_dwa = Instant::now();
        let mut dwa = DWA::from_components(&comp_dfas, &sigma, &atoms);
        debug_log(4, || format!("Constructed initial DWA from components ({} states) in {:?}", dwa.states.len(), now_dwa.elapsed()));

        debug_log(3, || format!("NWA::determinize_to_dwa total time: {:?}", now_total.elapsed()));

        dwa
    }
}

/* ------------------------------
   Utilities and support structs
   ------------------------------ */

fn debug_log(level: usize, msg: impl FnOnce() -> String) {
    crate::debug!(level, "{}", msg());
}

/// Alphabet = all labels that appear as exceptions, plus a special OTHER symbol.
#[derive(Clone, Debug)]
struct Alphabet {
    labels: Vec<i16>,
    other_index: usize,
}
impl Alphabet {
    fn from_nwa_with_future(nwa: &NWA, fut: &Vec<Weight>) -> Self {
        let mut set = BTreeSet::new();
        for (_s, st) in nwa.states.0.iter().enumerate() {
            for (&lbl, targets) in &st.transitions {
                if targets.iter().any(|(t, w)| !(&fut[*t] & w).is_empty()) {
                    set.insert(lbl);
                }
            }
            for def in &st.default {
                if !(&fut[def.target] & &def.weight).is_empty() {
                    set.extend(&def.exceptions);
                }
            }
        }
        let labels: Vec<i16> = set.into_iter().collect();
        let other_index = labels.len();
        Alphabet { labels, other_index }
    }
    #[inline]
    fn size(&self) -> usize { self.labels.len() + 1 }
    #[inline]
    fn is_other(&self, sym: usize) -> bool { sym == self.other_index }
    #[inline]
    fn label_at(&self, sym: usize) -> Option<i16> { self.labels.get(sym).copied() }
}

/// WeightPartition: disjoint contiguous atoms that cover the union of all weights.
#[derive(Clone, Debug)]
struct WeightPartition {
    intervals: Vec<RangeInclusive<usize>>,
    atoms: Vec<Weight>,
}
impl WeightPartition {
    fn from_nwa(nwa: &NWA) -> Self {
        let mut breaks = BTreeSet::new();
        let mut has_tail_to_max = false;

        let mut feed_weight = |w: &Weight| {
            for r in w.rsb.ranges() {
                breaks.insert(*r.start());
                if *r.end() == usize::MAX {
                    has_tail_to_max = true;
                } else {
                    breaks.insert(r.end().saturating_add(1));
                }
            }
        };

        for st in &nwa.states.0 {
            if let Some(w) = &st.final_weight { feed_weight(w); }
            for (_, w) in &st.epsilons { feed_weight(w); }
            for def in &st.default { feed_weight(&def.weight); }
            for (_, targets) in &st.transitions {
                for (_, w) in targets { feed_weight(w); }
            }
        }

        let mut breaks: Vec<usize> = breaks.into_iter().collect();
        if breaks.is_empty() && !has_tail_to_max {
            return WeightPartition { intervals: vec![], atoms: vec![] };
        }
        if breaks.is_empty() && has_tail_to_max {
            breaks.push(0);
        }

        let mut intervals: Vec<RangeInclusive<usize>> = Vec::new();
        for window in breaks.windows(2) {
            let (start, end_plus_1) = (window[0], window[1]);
            if start < end_plus_1 {
                intervals.push(start..=end_plus_1 - 1);
            }
        }
        if has_tail_to_max {
            if let Some(&last_break) = breaks.last() {
                intervals.push(last_break..=usize::MAX);
            }
        }

        let atoms = intervals.iter().map(|r| std::iter::once(r.clone()).collect()).collect();
        WeightPartition { intervals, atoms }
    }
}

/* ------------------------------
   Per-atom NFA and DFA
   ------------------------------ */

#[derive(Clone, Debug)]
struct PerAtomNFA {
    n: usize,
    start: usize,
    finals: Vec<bool>,
    ex_by_state: Vec<BTreeMap<i16, Vec<usize>>>,
    def_by_state: Vec<Vec<(usize, BTreeSet<i16>)>>,
    eps_by_state: Vec<Vec<usize>>,
}
impl PerAtomNFA {
    fn from_nwa(states: &NWAStates, start: usize, _sigma: &Alphabet, atom: &RangeInclusive<usize>, fut: &[Weight]) -> Self {
        let n_total = states.len();
        let atom_w: Weight = std::iter::once(atom.clone()).collect();

        let live: Vec<bool> = (0..n_total).map(|s| !(&fut[s] & &atom_w).is_empty()).collect();

        let trivial_nfa = || PerAtomNFA { n: 1, start: 0, finals: vec![false], ex_by_state: vec![BTreeMap::new()], def_by_state: vec![], eps_by_state: vec![] };

        if start >= n_total || !live[start] {
            return trivial_nfa();
        }

        let mut q = VecDeque::new();
        let mut visited = vec![false; n_total];
        let mut order = Vec::new();

        visited[start] = true;
        q.push_back(start);
        order.push(start);

        let mut head = 0;
        while head < q.len() {
            let u = q[head];
            head += 1;

            let check_and_add = |v: usize, q: &mut VecDeque<usize>, visited: &mut [bool], order: &mut Vec<usize>| {
                if v < n_total && live[v] && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                    order.push(v);
                }
            };

            for (v, w) in &states[u].epsilons { if !(&atom_w & w).is_empty() { check_and_add(*v, &mut q, &mut visited, &mut order); } }
            for (_, targets) in &states[u].transitions { for (v, w) in targets { if !(&atom_w & w).is_empty() { check_and_add(*v, &mut q, &mut visited, &mut order); } } }
            for def in &states[u].default { if !(&atom_w & &def.weight).is_empty() { check_and_add(def.target, &mut q, &mut visited, &mut order); } }
        }

        if order.is_empty() {
            return trivial_nfa();
        }

        let m = order.len();
        let mut id_of = vec![usize::MAX; n_total];
        for (i, &old) in order.iter().enumerate() { id_of[old] = i; }

        let mut new_self = PerAtomNFA {
            n: m,
            start: id_of[start],
            finals: vec![false; m],
            ex_by_state: vec![BTreeMap::new(); m],
            def_by_state: vec![Vec::new(); m],
            eps_by_state: vec![Vec::new(); m],
        };

        for (new_s, &old_s) in order.iter().enumerate() {
            if let Some(w) = &states[old_s].final_weight { if !(&atom_w & w).is_empty() { new_self.finals[new_s] = true; } }

            let mut local_ex = BTreeMap::new();
            for (&lbl, targets) in &states[old_s].transitions {
                let kept: Vec<usize> = targets.iter().filter_map(|(to, w)| {
                    if !(&atom_w & w).is_empty() && id_of[*to] != usize::MAX { Some(id_of[*to]) } else { None }
                }).collect();
                if !kept.is_empty() { local_ex.insert(lbl, kept); }
            }
            new_self.ex_by_state[new_s] = local_ex;

            let ex_for_def: BTreeSet<i16> = new_self.ex_by_state[new_s].keys().copied().collect();
            for def in &states[old_s].default {
                if !(&atom_w & &def.weight).is_empty() && id_of[def.target] != usize::MAX {
                    new_self.def_by_state[new_s].push((id_of[def.target], ex_for_def.clone()));
                }
            }

            for (to, w) in &states[old_s].epsilons {
                if !(&atom_w & w).is_empty() && id_of[*to] != usize::MAX {
                    new_self.eps_by_state[new_s].push(id_of[*to]);
                }
            }
        }
        new_self
    }

    fn eps_closure_per_state(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.n];
        for s in 0..self.n {
            let mut visited = vec![false; self.n];
            let mut stack = vec![s];
            visited[s] = true;
            while let Some(u) = stack.pop() {
                out[s].push(u);
                for &v in &self.eps_by_state[u] {
                    if !visited[v] {
                        visited[v] = true;
                        stack.push(v);
                    }
                }
            }
            out[s].sort_unstable();
        }
        out
    }

    fn eps_closure_set(&self, base: &[usize], per_state_closures: &Vec<Vec<usize>>) -> Vec<usize> {
        let mut closure_set = BTreeSet::new();
        for &s in base {
            if s < per_state_closures.len() {
                closure_set.extend(&per_state_closures[s]);
            }
        }
        closure_set.into_iter().collect()
    }

    fn determinize(&self, sigma: &Alphabet) -> DetDFA {
        let per_state_closures = self.eps_closure_per_state();
        let start_set = self.eps_closure_set(&[self.start], &per_state_closures);

        let mut map: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut dfa_states: Vec<(bool, Vec<Option<usize>>)> = Vec::new();
        let mut q = VecDeque::new();

        let intern = |subset: Vec<usize>, map: &mut HashMap<Vec<usize>, usize>, dfa_states: &mut Vec<(bool, Vec<Option<usize>>)>, q: &mut VecDeque<usize>| -> Option<usize> {
            if subset.is_empty() { return None; }
            if let Some(&id) = map.get(&subset) { return Some(id); }

            let id = dfa_states.len();
            let is_final = subset.iter().any(|&s| self.finals[s]);
            dfa_states.push((is_final, vec![None; sigma.size()]));
            map.insert(subset, id);
            q.push_back(id);
            Some(id)
        };

        let start_id = intern(start_set, &mut map, &mut dfa_states, &mut q).unwrap_or(0);

        let mut head = 0;
        while head < q.len() {
            let u_id = q[head];
            head += 1;
            let subset = map.iter().find(|(_, &v)| v == u_id).map(|(k, _)| k.clone()).unwrap();

            for sym in 0..sigma.size() {
                let mut next_raw = BTreeSet::new();
                match sigma.label_at(sym) {
                    Some(lbl) => {
                        for &s in &subset {
                            if let Some(ts) = self.ex_by_state[s].get(&lbl) { next_raw.extend(ts); }
                            for (target, exceptions) in &self.def_by_state[s] { if !exceptions.contains(&lbl) { next_raw.insert(*target); } }
                        }
                    }
                    None => { // OTHER
                        for &s in &subset { for (target, _) in &self.def_by_state[s] { next_raw.insert(*target); } }
                    }
                }
                let next_vec: Vec<usize> = next_raw.into_iter().collect();
                let next_subset = self.eps_closure_set(&next_vec, &per_state_closures);
                dfa_states[u_id].1[sym] = intern(next_subset, &mut map, &mut dfa_states, &mut q);
            }
        }

        let n = dfa_states.len();
        let sink = n;
        let mut finals = vec![false; n + 1];
        let mut trans = vec![vec![sink; sigma.size()]; n + 1];

        for i in 0..n {
            finals[i] = dfa_states[i].0;
            for sym in 0..sigma.size() {
                trans[i][sym] = dfa_states[i].1[sym].unwrap_or(sink);
            }
        }

        DetDFA { n_states: n + 1, start: start_id, finals, trans }
    }
}

#[derive(Clone, Debug)]
struct DetDFA {
    n_states: usize,
    start: usize,
    finals: Vec<bool>,
    trans: Vec<Vec<usize>>,
}
impl DetDFA {
    fn minimize(&mut self, sigma: &Alphabet) {
        if self.n_states == 0 { return; }
        let mut q = VecDeque::new();
        if self.start < self.n_states {
            q.push_back(self.start);
        }
        let mut reachable = vec![false; self.n_states];
        if self.start < self.n_states {
            reachable[self.start] = true;
        }

        let mut head = 0;
        while head < q.len() {
            let u = q[head];
            head += 1;
            for &v in &self.trans[u] {
                if v < self.n_states && !reachable[v] {
                    reachable[v] = true;
                    q.push_back(v);
                }
            }
        }

        let mut old_to_new = vec![usize::MAX; self.n_states];
        let mut new_to_old = Vec::new();
        for i in 0..self.n_states {
            if reachable[i] {
                old_to_new[i] = new_to_old.len();
                new_to_old.push(i);
            }
        }

        if new_to_old.is_empty() {
            self.n_states = 1; self.start = 0; self.finals = vec![false]; self.trans = vec![vec![0; sigma.size()]];
            return;
        }

        let n = new_to_old.len();
        let pruned_dfa = DetDFA {
            n_states: n,
            start: old_to_new[self.start],
            finals: new_to_old.iter().map(|&old| self.finals[old]).collect(),
            trans: new_to_old.iter().map(|&old| self.trans[old].iter().map(|&v| old_to_new[v]).collect()).collect(),
        };

        let (quotient_dfa, _) = Self::hopcroft_minimize(&pruned_dfa, sigma);
        *self = quotient_dfa;
    }

    fn find_sink_index(&self, sigma: &Alphabet) -> Option<usize> {
        (0..self.n_states).find(|&s| !self.finals[s] && self.trans[s].iter().all(|&t| t == s))
    }

    fn hopcroft_minimize(dfa: &DetDFA, sigma: &Alphabet) -> (DetDFA, Vec<usize>) {
        let n = dfa.n_states;
        if n <= 1 { return (dfa.clone(), (0..n).collect()); }

        let a = sigma.size();
        let mut part_id = vec![0; n];
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        let (accepting, non_accepting): (Vec<_>, Vec<_>) = (0..n).partition(|&s| dfa.finals[s]);

        if accepting.is_empty() || non_accepting.is_empty() {
            let mut new_dfa = dfa.clone();
            if n > 0 {
                new_dfa.start = 0;
            }
            return (new_dfa, vec![0; n]);
        }
        blocks.push(accepting);
        blocks.push(non_accepting);
        for (pid, block) in blocks.iter().enumerate() { for &s in block { part_id[s] = pid; } }

        let mut inv: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); n]; a];
        for s in 0..n { for sym in 0..a { let v = dfa.trans[s][sym]; inv[sym][v].push(s); } }

        let mut worklist: BTreeSet<(usize, usize)> = BTreeSet::new();
        let smaller_set = if blocks[0].len() <= blocks[1].len() { 0 } else { 1 };
        for sym in 0..a { worklist.insert((smaller_set, sym)); }

        while let Some(&(b, sym)) = worklist.iter().next() {
            worklist.remove(&(b, sym));
            let mut pre = BTreeSet::new();
            for &v in &blocks[b] { pre.extend(&inv[sym][v]); }
            if pre.is_empty() { continue; }

            let mut affected: HashMap<usize, Vec<usize>> = HashMap::new();
            for &s in &pre {
                let pid = part_id[s];
                affected.entry(pid).or_insert_with(Vec::new).push(s);
            }

            for (pid, in_pre) in affected {
                if in_pre.len() == blocks[pid].len() { continue; }

                let pid2 = blocks.len();
                let (mut p1, mut p2) = (Vec::new(), Vec::new());
                let in_pre_set: BTreeSet<_> = in_pre.into_iter().collect();
                for &s in &blocks[pid] {
                    if in_pre_set.contains(&s) { p1.push(s); } else { p2.push(s); }
                }

                if p1.is_empty() || p2.is_empty() { continue; }

                blocks.push(p2);
                blocks[pid] = p1;
                for &s in &blocks[pid] { part_id[s] = pid; }
                for &s in &blocks[pid2] { part_id[s] = pid2; }

                for sym2 in 0..a {
                    if worklist.remove(&(pid, sym2)) {
                        worklist.insert((pid, sym2));
                        worklist.insert((pid2, sym2));
                    } else {
                        if blocks[pid].len() <= blocks[pid2].len() {
                            worklist.insert((pid, sym2));
                        } else {
                            worklist.insert((pid2, sym2));
                        }
                    }
                }
            }
        }

        let num_parts = blocks.len();
        let mut repr = vec![0; num_parts];
        for (pid, block) in blocks.iter().enumerate() { if !block.is_empty() { repr[pid] = block[0]; } }

        let start_part = if dfa.start < n { part_id[dfa.start] } else { 0 };
        let finals2 = (0..num_parts).map(|pid| dfa.finals[repr[pid]]).collect();
        let mut trans2 = vec![vec![0; a]; num_parts];
        for pid in 0..num_parts {
            let s = repr[pid];
            for sym in 0..a {
                trans2[pid][sym] = part_id[dfa.trans[s][sym]];
            }
        }

        (DetDFA { n_states: num_parts, start: start_part, finals: finals2, trans: trans2 }, part_id)
    }
}

/* ------------------------------
   Direct DWA Construction from Components
   ------------------------------ */

impl DWA {
    fn from_components(comps: &[DetDFA], sigma: &Alphabet, atoms: &WeightPartition) -> Self {
        let k = comps.len();
        let mut dwa = DWA::new();
        dwa.states.0.clear();

        let mut map: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut q = VecDeque::new();

        let start_tuple: Vec<usize> = comps.iter().map(|c| c.start).collect();

        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new_spinner();
            p.set_style(ProgressStyle::default_spinner().template("{spinner:.green} [Determinize: {elapsed_precise}] Building DWA... States found: {pos}").unwrap());
            Some(p)
        } else { None };

        let start_id = dwa.add_state();
        map.insert(start_tuple.clone(), start_id);
        q.push_back((start_id, start_tuple));
        dwa.body.start_state = start_id;
        if let Some(p) = &pb { p.set_position(dwa.states.len() as u64); }

        let atom_weights = &atoms.atoms;
        let comp_sinks: Vec<_> = comps.iter().map(|c| c.find_sink_index(sigma)).collect();

        while let Some((u_id, u_tuple)) = q.pop_front() {
            let mut w_final = Weight::zeros();
            let mut w_active = Weight::zeros();
            for (i, &comp_state) in u_tuple.iter().enumerate() {
                if comps[i].finals[comp_state] { w_final |= &atom_weights[i]; }
                if comp_sinks[i].map_or(true, |s| comp_state != s) { w_active |= &atom_weights[i]; }
            }

            if !w_final.is_empty() { dwa.states[u_id].final_weight = Some(w_final); }
            let edge_weight = if w_active.is_empty() { Weight::zeros() } else { w_active };

            for sym in 0..sigma.size() {
                let next_tuple: Vec<usize> = u_tuple.iter().enumerate().map(|(i, &s)| comps[i].trans[s][sym]).collect();

                let v_id = *map.entry(next_tuple.clone()).or_insert_with(|| {
                    let new_id = dwa.add_state();
                    q.push_back((new_id, next_tuple));
                    if let Some(p) = &pb { p.set_position(dwa.states.len() as u64); }
                    new_id
                });

                if sigma.is_other(sym) {
                    dwa.set_default_transition(u_id, v_id, edge_weight.clone()).unwrap();
                } else {
                    let lbl = sigma.label_at(sym).unwrap();
                    dwa.add_transition(u_id, lbl, v_id, edge_weight.clone()).unwrap();
                }
            }
        }

        if let Some(p) = pb { p.finish_with_message(format!("DWA construction done, {} states", dwa.states.len())); }
        dwa
    }
}