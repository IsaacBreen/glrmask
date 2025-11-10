#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::SimpleBitset as Weight;
use super::dwa::{DWAState, DWAStates, DWA, DWABody};
use super::nwa::{NWA, NWAStates};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque, HashSet};
use std::ops::RangeInclusive;
use std::time::Instant;

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now_total = Instant::now();

        let mut nwa = self.clone();
        debug_log(3, || format!("Starting determinization for NWA with {} states", nwa.states.len()));

        let fut = nwa.compute_future_weights();

        let now_atoms = Instant::now();
        let atoms = WeightPartition::from_nwa(&nwa);
        debug_log(4, || {
            format!("Built weight partition with {} atoms in {:?}", atoms.intervals.len(), now_atoms.elapsed())
        });

        let now_sigma = Instant::now();
        let sigma = Alphabet::from_nwa_with_future(&nwa, &fut);
        debug_log(4, || {
            format!("Built alphabet with {} labels in {:?}", sigma.labels.len(), now_sigma.elapsed())
        });

        let pb_atoms = if PROGRESS_BAR_ENABLED {
            Some(
                ProgressBar::new(atoms.intervals.len() as u64).with_style(
                    ProgressStyle::default_bar()
                        .template(
                            "{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (per-atom DFAs)",
                        )
                        .unwrap(),
                ),
            )
        } else {
            None
        };

        let mut comp_dfas: Vec<DetDFA> = Vec::with_capacity(atoms.intervals.len());
        for (i, atom) in atoms.intervals.iter().enumerate() {
            let nfa = PerAtomNFA::from_nwa(&nwa.states, nwa.body.start_state, &sigma, atom, &fut);
            let mut dfa = nfa.determinize(&sigma);
            dfa.minimize(&sigma);
            crate::debug!(4, "Atom {}: interval={:?}, DFA states={}", i, atom, dfa.n_states);
            comp_dfas.push(dfa);
            if let Some(p) = &pb_atoms {
                p.inc(1);
            }
        }
        if let Some(p) = pb_atoms {
            p.finish_with_message("Per-atom DFAs built & minimized");
        }

        if atoms.intervals.is_empty() {
            let mut dwa = DWA::new();
            return dwa;
        }

        let now_product = Instant::now();
        let product = ProductDFA::from_components(&comp_dfas, &sigma, &atoms);
        debug_log(4, || {
            format!(
                "Product DFA: states={}, alphabet_size={}, time={:?}",
                product.n_states, sigma.size(), now_product.elapsed()
            )
        });

        let now_convert = Instant::now();
        let dwa = product.to_dwa(&sigma, &atoms, &comp_dfas);
        debug_log(4, || {
            format!("Product->DWA conversion took {:?}", now_convert.elapsed())
        });

        let dwa = dwa;

        debug_log(3, || {
            format!("NWA::determinize_to_dwa total time: {:?}", now_total.elapsed())
        });

        dwa
    }
}

fn debug_log(level: usize, msg: impl FnOnce() -> String) {
    crate::debug!(level, "{}", msg());
}

#[derive(Clone, Debug)]
struct Alphabet {
    labels: Vec<i16>,
    other_index: usize,
}

impl Alphabet {
    fn from_nwa_with_future(nwa: &NWA, fut: &Vec<Weight>) -> Self {
        let mut set = BTreeSet::new();
        for (s, st) in nwa.states.0.iter().enumerate() {
            for (&lbl, targets) in &st.transitions {
                let mut relevant = false;
                for (t, w) in targets {
                    if !(&fut[*t] & w).is_empty() {
                        relevant = true;
                        break;
                    }
                }
                if relevant {
                    set.insert(lbl);
                }
            }
            for def in &st.default {
                if !(&fut[def.target] & &def.weight).is_empty() {
                    for &lbl in &def.exceptions {
                        set.insert(lbl);
                    }
                }
            }
        }
        let labels: Vec<i16> = set.into_iter().collect();
        let other_index = labels.len();
        Alphabet { labels, other_index }
    }
    
    #[inline]
    fn size(&self) -> usize {
        self.labels.len() + 1
    }
    
    #[inline]
    fn index_of_label(&self, l: i16) -> Option<usize> {
        self.labels.binary_search(&l).ok()
    }
    
    #[inline]
    fn is_other(&self, sym: usize) -> bool {
        sym == self.other_index
    }
    
    #[inline]
    fn label_at(&self, sym: usize) -> Option<i16> {
        if sym < self.labels.len() {
            Some(self.labels[sym])
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct WeightPartition {
    intervals: Vec<RangeInclusive<usize>>,
    atoms: Vec<Weight>,
}

impl WeightPartition {
    fn from_nwa(nwa: &NWA) -> Self {
        let mut starts: BTreeSet<usize> = BTreeSet::new();
        let mut ends_plus: BTreeSet<usize> = BTreeSet::new();
        let mut has_tail_to_max = false;

        let mut feed_weight = |w: &Weight| {
            let mut it = w.rsb.ranges();
            while let Some(r) = it.next() {
                let s = *r.start();
                let e = *r.end();
                starts.insert(s);
                if e == usize::MAX {
                    has_tail_to_max = true;
                } else {
                    ends_plus.insert(e.saturating_add(1));
                }
            }
        };

        for st in &nwa.states.0 {
            if let Some(w) = &st.final_weight {
                if !w.is_empty() {
                    feed_weight(w);
                }
            }
            for (_, w) in &st.epsilons {
                if !w.is_empty() {
                    feed_weight(w);
                }
            }
            for def in &st.default {
                if !def.weight.is_empty() {
                    feed_weight(&def.weight);
                }
            }
            for (_, targets) in &st.transitions {
                for (_, w) in targets {
                    if !w.is_empty() {
                        feed_weight(w);
                    }
                }
            }
        }

        if starts.is_empty() && ends_plus.is_empty() && !has_tail_to_max {
            return WeightPartition { intervals: vec![], atoms: vec![] };
        }

        let mut breaks: Vec<usize> = starts.union(&ends_plus).copied().collect();
        breaks.sort_unstable();
        breaks.dedup();
        
        if breaks.is_empty() {
            let singleton = 0usize..=usize::MAX;
            let atom_w: Weight = std::iter::once(singleton.clone()).collect();
            return WeightPartition { intervals: vec![singleton], atoms: vec![atom_w] };
        }

        let mut intervals: Vec<RangeInclusive<usize>> = Vec::new();
        for i in 0..breaks.len().saturating_sub(1) {
            let a = breaks[i];
            let b_excl = breaks[i + 1];
            if a < b_excl {
                let b = b_excl - 1;
                intervals.push(a..=b);
            }
        }
        if has_tail_to_max {
            if let Some(&last) = breaks.last() {
                if last <= usize::MAX {
                    intervals.push(last..=usize::MAX);
                }
            }
        }

        let atoms: Vec<Weight> = intervals
            .iter()
            .map(|r| {
                let r2: RangeInclusive<usize> = (*r.start())..=(*r.end());
                std::iter::once(r2).collect()
            })
            .collect();

        WeightPartition { intervals, atoms }
    }
}

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
        let atom_w: Weight = std::iter::once((*atom.start())..=(*atom.end())).collect();

        let mut live = vec![false; n_total];
        for s in 0..n_total {
            if !(&fut[s] & &atom_w).is_empty() {
                live[s] = true;
            }
        }

        if start >= n_total || !live[start] {
            return PerAtomNFA {
                n: 1,
                start: 0,
                finals: vec![false],
                ex_by_state: vec![BTreeMap::new()],
                def_by_state: vec![Vec::new()],
                eps_by_state: vec![Vec::new()],
            };
        }

        let mut visited = vec![false; n_total];
        let mut order: Vec<usize> = Vec::new();
        let mut q = VecDeque::new();
        visited[start] = true;
        q.push_back(start);
        order.push(start);

        while let Some(u) = q.pop_front() {
            for (v, w) in &states[u].epsilons {
                if *v < n_total && live[*v] && !(&atom_w & w).is_empty() && !visited[*v] {
                    visited[*v] = true;
                    q.push_back(*v);
                    order.push(*v);
                }
            }
            for (_lbl, targets) in &states[u].transitions {
                for (v, w) in targets {
                    if *v < n_total && live[*v] && !(&atom_w & w).is_empty() && !visited[*v] {
                        visited[*v] = true;
                        q.push_back(*v);
                        order.push(*v);
                    }
                }
            }
            for def in &states[u].default {
                let v = def.target;
                if v < n_total && live[v] && !(&atom_w & &def.weight).is_empty() && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                    order.push(v);
                }
            }
        }

        let m = order.len();
        let mut id_of = vec![usize::MAX; n_total];
        for (i, &old) in order.iter().enumerate() {
            id_of[old] = i;
        }

        let mut finals = vec![false; m];
        let mut ex_by_state: Vec<BTreeMap<i16, Vec<usize>>> = vec![BTreeMap::new(); m];
        let mut def_by_state: Vec<Vec<(usize, BTreeSet<i16>)>> = vec![Vec::new(); m];
        let mut eps_by_state: Vec<Vec<usize>> = vec![Vec::new(); m];

        for (new_s, &old_s) in order.iter().enumerate() {
            if let Some(w) = &states[old_s].final_weight {
                if !(&atom_w & w).is_empty() {
                    finals[new_s] = true;
                }
            }
            
            let mut local_ex: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
            for (&lbl, targets) in &states[old_s].transitions {
                let mut kept: Vec<usize> = Vec::new();
                for (to_old, w) in targets {
                    if *to_old < n_total && live[*to_old] && !(&atom_w & w).is_empty() {
                        let to_new = id_of[*to_old];
                        if to_new != usize::MAX {
                            kept.push(to_new);
                        }
                    }
                }
                if !kept.is_empty() {
                    kept.sort_unstable();
                    kept.dedup();
                    local_ex.insert(lbl, kept);
                }
            }
            ex_by_state[new_s] = local_ex.clone();

            let ex_for_def: BTreeSet<i16> = local_ex.keys().copied().collect();
            for def in &states[old_s].default {
                let to_old = def.target;
                if to_old < n_total && live[to_old] && !(&atom_w & &def.weight).is_empty() {
                    let to_new = id_of[to_old];
                    if to_new != usize::MAX {
                        def_by_state[new_s].push((to_new, ex_for_def.clone()));
                    }
                }
            }

            for (to_old, w) in &states[old_s].epsilons {
                if *to_old < n_total && live[*to_old] && !(&atom_w & w).is_empty() {
                    let to_new = id_of[*to_old];
                    if to_new != usize::MAX {
                        eps_by_state[new_s].push(to_new);
                    }
                }
            }
        }

        let new_start = id_of[start];
        debug_assert!(new_start != usize::MAX, "start must be visited when live[start] is true");
        Self { n: m, start: new_start, finals, ex_by_state, def_by_state, eps_by_state }
    }

    fn eps_closure_per_state(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::<usize>::new(); self.n];
        for s in 0..self.n {
            let mut visited = vec![false; self.n];
            let mut stack = vec![s];
            visited[s] = true;
            let mut closure = Vec::new();
            while let Some(u) = stack.pop() {
                closure.push(u);
                for &v in &self.eps_by_state[u] {
                    if v < self.n && !visited[v] {
                        visited[v] = true;
                        stack.push(v);
                    }
                }
            }
            closure.sort_unstable();
            out[s] = closure;
        }
        out
    }

    fn eps_closure_set(&self, base: &[usize], per: &Vec<Vec<usize>>) -> Vec<usize> {
        let mut mark = vec![false; self.n];
        let mut result = Vec::new();
        for &s in base {
            if s >= self.n {
                continue;
            }
            for &u in &per[s] {
                if !mark[u] {
                    mark[u] = true;
                    result.push(u);
                }
            }
        }
        result.sort_unstable();
        result
    }

    fn determinize(&self, sigma: &Alphabet) -> DetDFA {
        let per = self.eps_closure_per_state();
        let start_set = self.eps_closure_set(&[self.start], &per);

        let mut map: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut states: Vec<Vec<usize>> = Vec::new();
        let mut finals: Vec<bool> = Vec::new();
        let mut trans: Vec<Vec<Option<usize>>> = Vec::new();

        let pb_subset = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new_spinner();
            p.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [Determinize/Subset: {elapsed_precise}] States found: {pos}")
                    .unwrap(),
            );
            Some(p)
        } else {
            None
        };

        let mut push_state = |subset: Vec<usize>,
                              states: &mut Vec<Vec<usize>>,
                              finals: &mut Vec<bool>,
                              trans: &mut Vec<Vec<Option<usize>>>,
                              map: &mut HashMap<Vec<usize>, usize>| {
            if let Some(&id) = map.get(&subset) {
                return id;
            }
            let id = states.len();
            let is_final = subset.iter().any(|&s| self.finals[s]);
            states.push(subset);
            finals.push(is_final);
            trans.push(vec![None; sigma.size()]);
            map.insert(states[id].clone(), id);
            id
        };

        let start_id = push_state(start_set, &mut states, &mut finals, &mut trans, &mut map);
        if let Some(p) = &pb_subset {
            p.set_position(states.len() as u64);
        }

        let mut q = VecDeque::new();
        q.push_back(start_id);

        while let Some(u) = q.pop_front() {
            let subset = states[u].clone();

            for sym in 0..sigma.size() {
                let mut next_raw: Vec<usize> = Vec::new();

                match sigma.label_at(sym) {
                    Some(lbl) => {
                        for &s in &subset {
                            if let Some(ts) = self.ex_by_state[s].get(&lbl) {
                                next_raw.extend_from_slice(ts);
                            }
                            for (target, exceptions) in &self.def_by_state[s] {
                                if !exceptions.contains(&lbl) {
                                    next_raw.push(*target);
                                }
                            }
                        }
                    }
                    None => {
                        for &s in &subset {
                            for (target, _) in &self.def_by_state[s] {
                                next_raw.push(*target);
                            }
                        }
                    }
                }

                if next_raw.is_empty() {
                    continue;
                }

                next_raw.sort_unstable();
                next_raw.dedup();

                let next = self.eps_closure_set(&next_raw, &per);
                if next.is_empty() {
                    continue;
                }

                let v = if let Some(&id) = map.get(&next) {
                    id
                } else {
                    let id = states.len();
                    let is_final = next.iter().any(|&s| self.finals[s]);
                    states.push(next.clone());
                    finals.push(is_final);
                    trans.push(vec![None; sigma.size()]);
                    map.insert(next, id);
                    if let Some(p) = &pb_subset {
                        p.set_position(states.len() as u64);
                    }
                    q.push_back(id);
                    id
                };

                trans[u][sym] = Some(v);
            }
        }

        if let Some(p) = pb_subset {
            p.finish_with_message(format!("Subset construction done, {} states", states.len()));
        }

        DetDFA {
            n_states: states.len(),
            start: start_id,
            finals,
            trans,
        }
    }
}

#[derive(Clone, Debug)]
struct DetDFA {
    n_states: usize,
    start: usize,
    finals: Vec<bool>,
    trans: Vec<Vec<Option<usize>>>,
}

impl DetDFA {
    fn minimize(&mut self, sigma: &Alphabet) { // TODO: This is complex, check if it's correct
        debug_log(4, || format!("Minimizing DFA with {} states", self.n_states));
        
        let reachable = {
            let mut visited = vec![false; self.n_states];
            let mut q = VecDeque::new();
            visited[self.start] = true;
            q.push_back(self.start);
            while let Some(u) = q.pop_front() {
                for &v_opt in &self.trans[u] {
                    if v_opt.is_none() { continue; }
                    let v = v_opt.unwrap();
                    if !visited[v] {
                        visited[v] = true;
                        q.push_back(v);
                    }
                }
            }
            visited
        };

        let mut map = vec![usize::MAX; self.n_states];
        let mut new_states = 0usize;
        for i in 0..self.n_states {
            if reachable[i] {
                map[i] = new_states;
                new_states += 1;
            }
        }

        if new_states == 0 {
            self.n_states = 1;
            self.start = 0;
            self.finals = vec![false];
            self.trans = vec![vec![Some(0); sigma.size()]];
            return;
        }

        let mut finals = vec![false; new_states];
        let mut trans: Vec<Vec<Option<usize>>> = vec![vec![None; sigma.size()]; new_states];

        for i in 0..self.n_states {
            if !reachable[i] {
                continue;
            }
            let ni = map[i];
            finals[ni] = self.finals[i];
            for a in 0..sigma.size() {
                if let Some(v) = self.trans[i][a] {
                    trans[ni][a] = Some(map[v]);
                }
                // None remains None
            }
        }

        self.n_states = new_states;
        self.start = map[self.start];
        self.finals = finals;
        self.trans = trans;

        if self.n_states <= 1 {
            return;
        }

        // Hopcroft requires a complete DFA. We add a temporary sink state.
        let has_none = self.trans.iter().any(|row| row.iter().any(|x| x.is_none()));
        let (mut n, sink_id) = if has_none {
            (self.n_states + 1, Some(self.n_states))
        } else {
            (self.n_states, None)
        };

        let mut trans_complete = vec![vec![0; sigma.size()]; n];
        for i in 0..self.n_states {
            for j in 0..sigma.size() {
                trans_complete[i][j] = self.trans[i][j].unwrap_or(sink_id.unwrap());
            }
        }
        if let Some(sid) = sink_id {
            for j in 0..sigma.size() {
                trans_complete[sid][j] = sid;
            }
        }

        let mut finals_complete = self.finals.clone();
        if sink_id.is_some() {
            finals_complete.push(false);
        }

        let a = sigma.size();

        let mut part_id = vec![0usize; n];
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        let (accepting_block, non_accepting_block): (Vec<_>, Vec<_>) = (0..n).partition(|&s| finals_complete[s]);

        if accepting_block.is_empty() || non_accepting_block.is_empty() {
            return;
        }
        blocks.push(accepting_block);
        blocks.push(non_accepting_block);
        for (pid, block) in blocks.iter().enumerate() { 
            for &s in block { 
                part_id[s] = pid; 
            } 
        }

        let mut inv: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); n]; a];
        for s in 0..n { 
            for sym in 0..a {
                let v = trans_complete[s][sym];
                inv[sym][v].push(s); 
            } 
        }

        let mut worklist: BTreeSet<(usize, usize)> = BTreeSet::new();
        let smaller_initial_set = if blocks[0].len() <= blocks[1].len() { 0 } else { 1 };
        for sym in 0..a {
            worklist.insert((smaller_initial_set, sym));
        }

        let pb_hopcroft = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new_spinner();
            p.set_style(
                ProgressStyle::default_spinner()
                    .template(
                        "{spinner:.green} [Determinize/Minimize: {elapsed_precise}] Pass {pos}, worklist size: {msg}",
                    )
                    .unwrap(),
            );
            Some(p)
        } else {
            None
        };
        let mut passes = 0u64;

        while let Some(&(b, sym)) = worklist.iter().next() {
            worklist.remove(&(b, sym));

            if let Some(p) = &pb_hopcroft {
                passes += 1;
                p.set_position(passes);
                p.set_message(format!("{}", worklist.len()));
            }

            let mut pre: Vec<usize> = Vec::new();
            for &v in &blocks[b] { 
                pre.extend_from_slice(&inv[sym][v]); 
            }
            if pre.is_empty() {
                continue;
            }
            pre.sort_unstable();
            pre.dedup();

            let mut affected: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
            for &s in &pre { 
                let pid = part_id[s]; 
                affected.entry(pid).or_default().0.push(s); 
            }
            for (pid, (ref mut in_pre, ref mut not_in_pre)) in affected.iter_mut() {
                for &s in &blocks[*pid] { 
                    if !in_pre.binary_search(&s).is_ok() { 
                        not_in_pre.push(s); 
                    } 
                }
            }

            let mut to_replace: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();

            for (pid, (in_pre, not_in_pre)) in affected.into_iter() {
                if in_pre.is_empty() || not_in_pre.is_empty() {
                    continue;
                }
                to_replace.push((pid, in_pre, not_in_pre));
            }

            if to_replace.is_empty() {
                continue;
            }

            for (pid, mut in_pre, mut not_in_pre) in to_replace {
                in_pre.sort_unstable();
                not_in_pre.sort_unstable();

                let pid2 = blocks.len();
                blocks.push(not_in_pre);
                blocks[pid] = in_pre;

                for &s in &blocks[pid] { 
                    part_id[s] = pid; 
                }
                for &s in &blocks[pid2] { 
                    part_id[s] = pid2; 
                }

                for sym2 in 0..a {
                    if worklist.remove(&(pid, sym2)) {
                        worklist.insert((pid, sym2));
                        worklist.insert((pid2, sym2));
                    } else {
                        let (smaller_pid, _) = if blocks[pid].len() <= blocks[pid2].len() { 
                            (pid, pid2) 
                        } else { 
                            (pid2, pid) 
                        };
                        worklist.insert((smaller_pid, sym2));
                    }
                }
            }
        }

        if let Some(p) = pb_hopcroft {
            p.finish_with_message(format!("Hopcroft done, {} partitions", blocks.len()));
        }

        let num_parts = blocks.len();
        let mut repr: Vec<usize> = vec![0; num_parts];
        for (pid, block) in blocks.iter().enumerate() { 
            repr[pid] = block[0]; 
        }

        let start_part = part_id[self.start];
        let sink_part = sink_id.map(|sid| part_id[sid]);

        let mut finals2 = vec![false; num_parts];
        for pid in 0..num_parts {
            finals2[pid] = finals_complete[repr[pid]];
        }

        let mut trans2 = vec![vec![None; a]; num_parts];
        for pid in 0..num_parts {
            let s = repr[pid];
            for sym in 0..a {
                let v_part = part_id[trans_complete[s][sym]];
                if Some(v_part) == sink_part {
                    trans2[pid][sym] = None;
                } else {
                    trans2[pid][sym] = Some(v_part);
                }
            }
        }

        self.n_states = num_parts;
        self.start = start_part;
        self.finals = finals2;
        self.trans = trans2;
    }
}

#[derive(Clone, Debug)]
struct ProductDFA {
    n_states: usize,
    start: usize,
    finals: Vec<bool>,
    trans: Vec<Vec<usize>>,
    tuples: Vec<Vec<Option<usize>>>,
    active_weights: Vec<Weight>,
}

impl ProductDFA {
    fn from_components(comps: &[DetDFA], sigma: &Alphabet, atoms: &WeightPartition) -> Self {
        let k = comps.len();

        let pb_product = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new_spinner();
            p.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] Building product DFA... Tuples found: {pos}")
                    .unwrap(),
            );
            Some(p)
        } else {
            None
        };

        // Step 1: Collect all reachable tuples
        let start_tuple: Vec<Option<usize>> = comps.iter().map(|c| Some(c.start)).collect();
        let mut visited: HashSet<Vec<Option<usize>>> = HashSet::new();
        let mut queue: VecDeque<Vec<Option<usize>>> = VecDeque::new();

        visited.insert(start_tuple.clone());
        queue.push_back(start_tuple.clone());

        while let Some(current) = queue.pop_front() {
            for sym in 0..sigma.size() {
                let next_tuple: Vec<Option<usize>> = (0..k)
                    .map(|i| current[i].and_then(|s| comps[i].trans[s][sym]))
                    .collect();
                if visited.insert(next_tuple.clone()) {
                    queue.push_back(next_tuple);
                    if let Some(p) = &pb_product {
                        p.set_position(visited.len() as u64);
                    }
                }
            }
        }

        let all_tuples: Vec<Vec<Option<usize>>> = visited.into_iter().collect();
        let n = all_tuples.len();

        if let Some(p) = &pb_product {
            p.set_message(format!("Collected {} tuples, now merging...", n));
        }

        // Step 2: Group tuples by their canonical key
        let group_ids = Self::group_product_states(&all_tuples);
        let num_groups = group_ids.iter().max().map_or(0, |&max| max + 1);
        let mut classes: Vec<Vec<usize>> = vec![vec![]; num_groups];
        for (tuple_idx, group_id) in group_ids.iter().enumerate() {
            classes[*group_id].push(tuple_idx);
        }


        // Step 4: For each class, pick representative and compute active weight
        let mut class_list: Vec<(Vec<usize>, Vec<Option<usize>>, Weight)> = Vec::new();
        // (members, representative, active_weight)

        crate::debug!(5, "ProductDFA: Atoms: {}", atoms.intervals.len());
        for (i, a) in atoms.atoms.iter().enumerate() {
            crate::debug!(5, " Atom {}: {:?}", i, a);
        }

        for members in classes.into_iter() {
            if members.is_empty() {
                continue;
            }
            // Pick representative (tuple with fewest sinks)
            let best_idx = members
                .iter()
                .min_by_key(|&&idx| {
                    all_tuples[idx]
                        .iter()
                        .filter(|s| s.is_none())
                        .count()
                })
                .unwrap();
            let repr = all_tuples[*best_idx].clone();

            // Compute active weight: union of atoms for all non-sink positions across all members
            let mut active = Weight::zeros();
            for &member_idx in &members {
                let tuple = &all_tuples[member_idx];
                for (comp_idx, state) in tuple.iter().enumerate() {
                    if state.is_some() {
                        active |= &atoms.atoms[comp_idx];
                    }
                }
            }

            class_list.push((members, repr, active));
        }

        // Step 5: Build mappings
        let n_states = class_list.len();
        let mut tuple_to_id: HashMap<Vec<Option<usize>>, usize> = HashMap::new();

        for (id, (ref members, _repr, _active)) in class_list.iter().enumerate() {
            for &member_idx in members {
                tuple_to_id.insert(all_tuples[member_idx].clone(), id);
            }
        }

        crate::debug!(5, "ProductDFA: Merged {} tuples into {} states", n, n_states);
        for (id, (ref members, ref repr, active)) in class_list.iter().enumerate() {
            crate::debug!(5, " State {}:", id);
            crate::debug!(5, "  Representative: {:?}", repr);
            crate::debug!(5, "  Members ({}):", members.len());
            for &member_idx in members {
                crate::debug!(5, "   {:?}", all_tuples[member_idx]);
            }
            crate::debug!(5, "  Active weight: {:?}", active);
        }

        let start_id = tuple_to_id[&start_tuple];

        // Step 6: Build Product DFA structures
        let mut finals = vec![false; n_states];
        let mut trans = vec![vec![0; sigma.size()]; n_states];
        let mut tuples = vec![vec![]; n_states]; // Vec<Vec<Option<usize>>>
        let mut active_weights = vec![Weight::zeros(); n_states];

        for (id, (members, repr, active_weight)) in class_list.into_iter().enumerate() {
            tuples[id] = repr.clone();
            active_weights[id] = active_weight;

            // Check if any member is final (has at least one accepting component)
            finals[id] = members.iter().any(|&idx| {
                let tuple = &all_tuples[idx];
                tuple.iter().enumerate().any(|(i, s_opt)| {
                    if let Some(s) = s_opt { comps[i].finals[*s] } else { false }
                })
            });

            // Build transitions from representative
            for sym in 0..sigma.size() {
                let next_tuple: Vec<Option<usize>> = (0..k)
                    .map(|i| repr[i].and_then(|s| comps[i].trans[s][sym]))
                    .collect();
                let next_id = tuple_to_id[&next_tuple];
                trans[id][sym] = next_id;
            }
        }

        if let Some(p) = pb_product {
            p.finish_with_message(format!("Product DFA built with {} states (merged from {} tuples)", n_states, n));
        }

        ProductDFA {
            n_states,
            start: start_id,
            finals,
            trans,
            tuples,
            active_weights,
        }
    }

    fn to_dwa(&self, sigma: &Alphabet, atoms: &WeightPartition, comps: &[DetDFA]) -> DWA {
        let mut dwa_states = DWAStates::default();
        for _ in 0..self.n_states {
            dwa_states.add_state();
        }
        let mut dwa = DWA {
            states: dwa_states,
            body: DWABody {
                start_state: self.start,
            },
        };

        let pb_convert = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(self.n_states as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Attaching DWA weights)",
                    )
                    .unwrap(),
            );
            Some(p)
        } else {
            None
        };

        // Final weights
        for sid in 0..self.n_states {
            if !self.finals[sid] {
                continue;
            }
            let mut w_final = Weight::zeros();
            for (i, s_opt) in self.tuples[sid].iter().enumerate() {
                if let Some(s_i) = s_opt {
                    if comps[i].finals[*s_i] {
                        w_final |= &atoms.atoms[i];
                    }
                }
            }
            if !w_final.is_empty() {
                let _ = dwa.set_final_weight(sid, w_final);
            }
        }

        // Transitions
        for sid in 0..self.n_states {
            if let Some(p) = &pb_convert {
                p.inc(1);
            }

            let edge_weight = &self.active_weights[sid];

            // Default (OTHER)
            let dst_def = self.trans[sid][sigma.other_index];
            if sid < dwa.states.len() && dst_def < dwa.states.len() {
                let _ = dwa.set_default_transition(sid, dst_def, edge_weight.clone());
            }

            // Exceptions for each explicit label
            for (li, &lbl) in sigma.labels.iter().enumerate() {
                let dst = self.trans[sid][li];
                if sid < dwa.states.len() && dst < dwa.states.len() {
                    let _ = dwa.add_transition(sid, lbl, dst, edge_weight.clone());
                }
            }
        }

        if let Some(p) = pb_convert {
            p.finish_with_message("Attached DWA weights");
        }
        dwa
    }

    /// Groups product state tuples into equivalence classes.
    /// Returns a vector where `result[i]` is the group ID for `all_tuples[i]`.
    fn group_product_states(all_tuples: &[Vec<Option<usize>>]) -> Vec<usize> {
        // This is a greedy, order-dependent implementation for clique partitioning.
        // Two tuples can be in the same group if for any component, their states are
        // either the same, or one of them is a sink (None).
        let n = all_tuples.len();
        let mut group_ids = vec![usize::MAX; n];
        // Each group is represented by its canonical "core", which is the superposition of the cores of its members.
        let mut group_cores: Vec<BTreeMap<usize, usize>> = Vec::new();
        let mut next_group_id = 0;

        for (i, tuple) in all_tuples.iter().enumerate() {
            let core: BTreeMap<usize, usize> = tuple
                .iter()
                .enumerate()
                .filter_map(|(comp_idx, s_opt)| s_opt.map(|s| (comp_idx, s)))
                .collect();

            let mut assigned_gid = None;
            // Find the first existing group that this tuple is compatible with.
            for (gid, g_core) in group_cores.iter_mut().enumerate() {
                let compatible = core.iter().all(|(k, v)| g_core.get(k).map_or(true, |v2| v == v2))
                    && g_core.iter().all(|(k, v)| core.get(k).map_or(true, |v2| v == v2));

                if compatible {
                    // Add to this group and update the group's canonical core.
                    g_core.extend(core.clone());
                    assigned_gid = Some(gid);
                    break;
                }
            }

            let gid = assigned_gid.unwrap_or_else(|| {
                // No compatible group found, create a new one.
                let new_gid = next_group_id;
                next_group_id += 1;
                group_cores.push(core);
                new_gid
            });
            group_ids[i] = gid;
        }
        group_ids
    }
}
