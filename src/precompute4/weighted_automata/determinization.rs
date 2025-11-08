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

/// This determinization follows a principled construction using a shared DFA representation:
/// 1) Extract a minimal disjoint partition of the weight space (atoms).
/// 2) For each atom, build and minimize a component DFA.
/// 3) Combine all component DFAs into a single large DFA via disjoint union.
/// 4) Minimize this "shared DFA". This crucial step merges any structurally identical states,
///    even if they originated from different atoms, dramatically reducing the state space.
/// 5) Build the final DWA by simulating a product construction over this compact, shared DFA.
///    A DWA state represents a tuple of pointers into the shared DFA, one for each atom.
///    This avoids the state space explosion of a traditional product construction by leveraging
///    the shared structure and encoding atom-specific information (finality, liveness) in weights.
impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now_total = Instant::now();

        let mut nwa = self.clone();
        debug_log(3, || format!("Starting determinization for NWA with {} states", nwa.states.len()));

        let fut = nwa.compute_future_weights();
        let now_atoms = Instant::now();
        let atoms = WeightPartition::from_nwa(&nwa);
        debug_log(4, || format!("Built weight partition with {} atoms in {:?}", atoms.intervals.len(), now_atoms.elapsed()));

        let now_sigma = Instant::now();
        let sigma = Alphabet::from_nwa_with_future(&nwa, &fut);
        debug_log(4, || format!("Built alphabet with {} labels in {:?}", sigma.labels.len(), now_sigma.elapsed()));

        // 1. Build and minimize a DFA for each atom.
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
            return DWA::new();
        }

        // 2. Combine all component DFAs into a single, shared DFA and minimize it.
        let now_shared = Instant::now();
        let shared_dfa = SharedDFA::from_components(&comp_dfas, &sigma);
        debug_log(4, || {
            format!(
                "Built and minimized shared DFA ({} states) in {:?}",
                shared_dfa.dfa.n_states, now_shared.elapsed()
            )
        });

        // 3. Build the final DWA by simulating the product construction on the SHARED DFA.
        let now_dwa = Instant::now();
        let mut dwa = DWA::from_shared_dfa(&shared_dfa, &sigma, &atoms);
        debug_log(4, || {
            format!(
                "Constructed DWA from shared DFA ({} states) in {:?}",
                dwa.states.len(), now_dwa.elapsed()
            )
        });

        debug_log(3, || {
            format!("NWA::determinize_to_dwa total time: {:?}", now_total.elapsed())
        });

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
    // ... (Implementation of PerAtomNFA is unchanged, it's correct) ...
    fn from_nwa(states: &NWAStates, start: usize, _sigma: &Alphabet, atom: &RangeInclusive<usize>, fut: &[Weight]) -> Self {
        let n_total = states.len();
        let atom_w: Weight = std::iter::once((*atom.start())..=(*atom.end())).collect();

        // Live states for this atom: those with F[s] ∧ atom ≠ ∅
        let mut live = vec![false; n_total];
        for s in 0..n_total {
            if !(&fut[s] & &atom_w).is_empty() {
                live[s] = true;
            }
        }

        // If start is not live, return a trivial 1-state NFA (non-final, no edges).
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

        // BFS to collect only states reachable from start via edges that intersect atom and lead to live targets.
        let mut visited = vec![false; n_total];
        let mut order: Vec<usize> = Vec::new();
        let mut q = VecDeque::new();
        visited[start] = true;
        q.push_back(start);
        order.push(start);

        while let Some(u) = q.pop_front() {
            // Epsilons
            for (v, w) in &states[u].epsilons {
                if *v < n_total && live[*v] && !(&atom_w & w).is_empty() && !visited[*v] {
                    visited[*v] = true;
                    q.push_back(*v);
                    order.push(*v);
                }
            }
            // Labeled
            for (_lbl, targets) in &states[u].transitions {
                for (v, w) in targets {
                    if *v < n_total && live[*v] && !(&atom_w & w).is_empty() && !visited[*v] {
                        visited[*v] = true;
                        q.push_back(*v);
                        order.push(*v);
                    }
                }
            }
            // Defaults
            for def in &states[u].default {
                let v = def.target;
                if v < n_total && live[v] && !(&atom_w & &def.weight).is_empty() && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                    order.push(v);
                }
            }
        }

        // Compact IDs
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
            // final
            if let Some(w) = &states[old_s].final_weight {
                if !(&atom_w & w).is_empty() {
                    finals[new_s] = true;
                }
            }
            // exceptions for this atom
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

            // default(s) for this atom: use per-atom exceptions = keys of local_ex
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

            // epsilons
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
        // ... (Implementation is unchanged) ...
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
        // ... (Implementation is unchanged) ...
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
        // ... (Implementation is unchanged) ...
        let per = self.eps_closure_per_state();

        // Start set: closure({start})
        let start_set = self.eps_closure_set(&[self.start], &per);

        // Subset-to-id interning
        let mut map: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut states: Vec<Vec<usize>> = Vec::new();
        let mut finals: Vec<bool> = Vec::new();
        let mut trans: Vec<Vec<Option<usize>>> = Vec::new(); // Option for next state; sink if None later

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

            // For each symbol in Sigma', compute the next subset (then epsilon closure)
            for sym in 0..sigma.size() {
                let mut next_raw: Vec<usize> = Vec::new();

                match sigma.label_at(sym) {
                    Some(lbl) => {
                        // Label 'lbl': for each s in subset: take explicit transitions on lbl,
                        // and any default transitions for which lbl is not an exception.
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
                        // OTHER: for any symbol not in sigma.labels, no state has an exception,
                        // so we take all default transitions.
                        for &s in &subset {
                            for (target, _) in &self.def_by_state[s] {
                                next_raw.push(*target);
                            }
                        }
                    }
                }

                if next_raw.is_empty() {
                    // Will lead to sink; leave as None for now
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

        // Build sink state if any transition is None
        let needs_sink = trans.iter().any(|row| row.iter().any(|x| x.is_none()));
        let mut sink_index: Option<usize> = None;

        if needs_sink {
            sink_index = Some(trans.len());
            trans.push(vec![None; sigma.size()]);
            finals.push(false);
            states.push(Vec::new()); // empty subset for sink
        }

        // Fill None transitions with sink (or self-loop if no sink needed)
        let mut out_trans: Vec<Vec<usize>> = Vec::with_capacity(trans.len());
        for (i, row) in trans.iter().enumerate() {
            let mut new_row = Vec::with_capacity(row.len());
            for (_sym, dst) in row.iter().enumerate() {
                match dst {
                    Some(v) => new_row.push(*v),
                    None => {
                        if let Some(sink) = sink_index {
                            new_row.push(sink);
                        } else {
                            new_row.push(i); // complete with self (won't happen unless needs_sink=false)
                        }
                    }
                }
            }
            out_trans.push(new_row);
        }

        DetDFA {
            n_states: out_trans.len(),
            start: start_id,
            finals,
            trans: out_trans,
        }
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
    // ... (Implementation of minimize and find_sink_index are unchanged) ...
    fn minimize(&mut self, sigma: &Alphabet) {
        // ... (Implementation is unchanged) ...
        debug_log(4, || format!("Minimizing DFA with {} states", self.n_states));
        // Remove states unreachable from start first
        let reachable = {
            let mut visited = vec![false; self.n_states];
            let mut q = VecDeque::new();
            if self.start < self.n_states {
                visited[self.start] = true;
                q.push_back(self.start);
            }
            while let Some(u) = q.pop_front() {
                for &v in &self.trans[u] {
                    if !visited[v] {
                        visited[v] = true;
                        q.push_back(v);
                    }
                }
            }
            visited
        };

        // Map reachable states to compact ids
        let mut map = vec![usize::MAX; self.n_states];
        let mut new_states = 0usize;
        for i in 0..self.n_states {
            if reachable[i] {
                map[i] = new_states;
                new_states += 1;
            }
        }

        if new_states == 0 {
            // No reachable states? Create a single dead state.
            self.n_states = 1;
            self.start = 0;
            self.finals = vec![false];
            self.trans = vec![vec![0; sigma.size()]];
            return;
        }

        let mut finals = vec![false; new_states];
        let mut trans = vec![vec![0usize; sigma.size()]; new_states];

        for i in 0..self.n_states {
            if !reachable[i] {
                continue;
            }
            let ni = map[i];
            finals[ni] = self.finals[i];
            for a in 0..sigma.size() {
                let v = self.trans[i][a];
                trans[ni][a] = map[v];
            }
        }

        self.n_states = new_states;
        self.start = map[self.start];
        self.finals = finals;
        self.trans = trans;

        // Hopcroft minimization on complete DFA
        if self.n_states <= 1 {
            return;
        }

        let n = self.n_states;
        let a = sigma.size();

        // Initial partition: accepting vs non-accepting
        let mut part_id = vec![0usize; n];
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        let (accepting_block, non_accepting_block): (Vec<_>, Vec<_>) = (0..n).partition(|&s| self.finals[s]);

        if accepting_block.is_empty() || non_accepting_block.is_empty() {
            // All accepting or all non-accepting -> nothing to split
            return;
        }
        blocks.push(accepting_block);
        blocks.push(non_accepting_block);
        for (pid, block) in blocks.iter().enumerate() { for &s in block { part_id[s] = pid; } }

        // Build inverse transitions for each symbol
        let mut inv: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); n]; a];
        for s in 0..n { for sym in 0..a { let v = self.trans[s][sym]; inv[sym][v].push(s); } }

        // Worklist of (block id, symbol). Using BTreeSet for efficient presence checks and removal.
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

            // Compute preimage of block b under symbol sym
            let mut pre: Vec<usize> = Vec::new();
            for &v in &blocks[b] { pre.extend_from_slice(&inv[sym][v]); }
            if pre.is_empty() {
                continue;
            }
            pre.sort_unstable();
            pre.dedup();

            // For each block, split by intersection with pre
            let mut affected: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
            for &s in &pre { let pid = part_id[s]; affected.entry(pid).or_default().0.push(s); }
            for (pid, (ref mut in_pre, ref mut not_in_pre)) in affected.iter_mut() {
                // Fill not_in_pre
                for &s in &blocks[*pid] { if !in_pre.binary_search(&s).is_ok() { not_in_pre.push(s); } }
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

            // Apply replacements (block splits)
            for (pid, mut in_pre, mut not_in_pre) in to_replace {
                in_pre.sort_unstable();
                not_in_pre.sort_unstable();

                let pid2 = blocks.len();
                blocks.push(not_in_pre);
                blocks[pid] = in_pre;

                // Update part_id map for all states in the newly created blocks
                for &s in &blocks[pid] { part_id[s] = pid; }
                for &s in &blocks[pid2] { part_id[s] = pid2; }

                // Update worklist according to Hopcroft's algorithm
                for sym2 in 0..a {
                    if worklist.remove(&(pid, sym2)) {
                        // The original block was on the worklist. Replace it with both new blocks.
                        worklist.insert((pid, sym2));
                        worklist.insert((pid2, sym2));
                    } else {
                        // The original block was not on the worklist. Add the smaller of the two new blocks.
                        let (smaller_pid, _) = if blocks[pid].len() <= blocks[pid2].len() { (pid, pid2) } else { (pid2, pid) };
                        worklist.insert((smaller_pid, sym2));
                    }
                }
            }
        }

        if let Some(p) = pb_hopcroft {
            p.finish_with_message(format!("Hopcroft done, {} partitions", blocks.len()));
        }

        // Build the quotient automaton
        let num_parts = blocks.len();
        let mut repr: Vec<usize> = vec![0; num_parts];
        for (pid, block) in blocks.iter().enumerate() { repr[pid] = block[0]; }

        let start_part = part_id[self.start];
        let mut finals2 = vec![false; num_parts];
        for pid in 0..num_parts {
            finals2[pid] = self.finals[repr[pid]];
        }

        let mut trans2 = vec![vec![0usize; a]; num_parts];
        for pid in 0..num_parts {
            let s = repr[pid];
            for sym in 0..a {
                let v = self.trans[s][sym];
                trans2[pid][sym] = part_id[v];
            }
        }

        self.n_states = num_parts;
        self.start = start_part;
        self.finals = finals2;
        self.trans = trans2;
    }
    fn find_sink_index(&self, sigma: &Alphabet) -> Option<usize> {
        // ... (Implementation is unchanged) ...
        'outer: for s in 0..self.n_states {
            if self.finals[s] {
                continue;
            }
            for sym in 0..sigma.size() {
                if self.trans[s][sym] != s {
                    continue 'outer;
                }
            }
            return Some(s);
        }
        None
    }
}

/* ------------------------------
   Shared DFA Representation
   ------------------------------ */

struct SharedDFA {
    dfa: DetDFA,
    state_info: Vec<Vec<(usize, bool, bool)>>, // (atom_index, is_final, is_sink)
    atom_starts: Vec<usize>,
}

impl SharedDFA {
    fn from_components(comps: &[DetDFA], sigma: &Alphabet) -> Self {
        if comps.is_empty() {
            return Self {
                dfa: DetDFA { n_states: 0, start: 0, finals: vec![], trans: vec![] },
                state_info: vec![],
                atom_starts: vec![],
            };
        }

        // 1. Disjoint union of all components
        let mut combined_trans = Vec::new();
        let mut combined_finals = Vec::new();
        let mut state_info_pre: Vec<(usize, bool, bool)> = Vec::new();
        let mut atom_starts_pre = Vec::with_capacity(comps.len());
        let mut offset = 0;

        for (atom_idx, comp) in comps.iter().enumerate() {
            atom_starts_pre.push(comp.start + offset);
            let comp_sink = comp.find_sink_index(sigma);

            for i in 0..comp.n_states {
                let new_trans_row: Vec<usize> = comp.trans[i].iter().map(|&t| t + offset).collect();
                combined_trans.push(new_trans_row);
                combined_finals.push(comp.finals[i]);
                let is_sink = comp_sink.map_or(false, |s| i == s);
                state_info_pre.push((atom_idx, comp.finals[i], is_sink));
            }
            offset += comp.n_states;
        }

        let mut shared_dfa = DetDFA {
            n_states: combined_trans.len(),
            start: 0, // Placeholder
            finals: combined_finals,
            trans: combined_trans,
        };

        // 2. Minimize the combined DFA and get the partition map.
        let (minimized_dfa, partition_map) = Self::minimize_and_get_partition_map(&mut shared_dfa, sigma, &atom_starts_pre);

        // 3. Rebuild state_info and atom_starts based on the partition map.
        let num_new_states = minimized_dfa.n_states;
        let mut new_state_info: Vec<Vec<(usize, bool, bool)>> = vec![Vec::new(); num_new_states];
        for old_id in 0..shared_dfa.n_states {
            let new_id = partition_map[old_id];
            let (atom_idx, is_final, is_sink) = state_info_pre[old_id];
            new_state_info[new_id].push((atom_idx, is_final, is_sink));
        }

        for info in &mut new_state_info {
            info.sort_unstable();
            info.dedup();
        }

        let new_atom_starts = atom_starts_pre.iter().map(|&s| partition_map[s]).collect();

        Self {
            dfa: minimized_dfa,
            state_info: new_state_info,
            atom_starts: new_atom_starts,
        }
    }

    // This is a helper that performs minimization and correctly returns the partition map.
    fn minimize_and_get_partition_map(dfa: &mut DetDFA, sigma: &Alphabet, starts: &[usize]) -> (DetDFA, Vec<usize>) {
        // The key is to first prune unreachable states from ALL potential start states.
        let mut q = VecDeque::from_iter(starts.iter().copied());
        let mut reachable = vec![false; dfa.n_states];
        for &s in starts {
            reachable[s] = true;
        }

        let mut head = 0;
        while head < q.len() {
            let u = q[head];
            head += 1;
            for &v in &dfa.trans[u] {
                if !reachable[v] {
                    reachable[v] = true;
                    q.push_back(v);
                }
            }
        }

        let mut old_to_new_map = vec![usize::MAX; dfa.n_states];
        let mut new_to_old_map = Vec::new();
        for i in 0..dfa.n_states {
            if reachable[i] {
                old_to_new_map[i] = new_to_old_map.len();
                new_to_old_map.push(i);
            }
        }

        if new_to_old_map.is_empty() {
            let empty_dfa = DetDFA { n_states: 0, start: 0, finals: vec![], trans: vec![] };
            return (empty_dfa, vec![0; dfa.n_states]);
        }

        let mut pruned_dfa = DetDFA {
            n_states: new_to_old_map.len(),
            start: 0, // Placeholder
            finals: new_to_old_map.iter().map(|&old| dfa.finals[old]).collect(),
            trans: new_to_old_map.iter().map(|&old| {
                dfa.trans[old].iter().map(|&v| old_to_new_map[v]).collect()
            }).collect(),
        };

        // Now, run Hopcroft on the pruned DFA.
        let (quotient_dfa, pruned_partition_map) = Self::hopcroft_minimize(&pruned_dfa, sigma);

        // Combine the maps: old -> pruned -> quotient
        let mut final_partition_map = vec![0; dfa.n_states];
        for old_id in 0..dfa.n_states {
            if reachable[old_id] {
                let pruned_id = old_to_new_map[old_id];
                final_partition_map[old_id] = pruned_partition_map[pruned_id];
            }
        }

        (quotient_dfa, final_partition_map)
    }

    fn hopcroft_minimize(dfa: &DetDFA, sigma: &Alphabet) -> (DetDFA, Vec<usize>) {
        if dfa.n_states <= 1 {
            return (dfa.clone(), (0..dfa.n_states).collect());
        }
        // ... (This is the main logic of Hopcroft's algorithm, refactored from DetDFA::minimize) ...
        // It should return the final minimized DFA and the partition map.
        let n = dfa.n_states;
        let a = sigma.size();

        let mut part_id = vec![0; n];
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        let (accepting, non_accepting): (Vec<_>, Vec<_>) = (0..n).partition(|&s| dfa.finals[s]);

        if accepting.is_empty() || non_accepting.is_empty() {
            return (dfa.clone(), (0..n).collect());
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
            // ... (rest of Hopcroft splitting logic) ...
            let mut pre: Vec<usize> = Vec::new();
            for &v in &blocks[b] { pre.extend_from_slice(&inv[sym][v]); }
            if pre.is_empty() { continue; }
            pre.sort_unstable();
            pre.dedup();

            let mut affected: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
            for &s in &pre { let pid = part_id[s]; affected.entry(pid).or_default().0.push(s); }
            for (pid, (ref mut in_pre, ref mut not_in_pre)) in affected.iter_mut() {
                let mut in_pre_set = BTreeSet::from_iter(in_pre.iter().copied());
                for &s in &blocks[*pid] { if !in_pre_set.contains(&s) { not_in_pre.push(s); } }
            }

            for (pid, (in_pre, not_in_pre)) in affected.into_iter() {
                if in_pre.is_empty() || not_in_pre.is_empty() { continue; }

                let pid2 = blocks.len();
                blocks.push(not_in_pre);
                blocks[pid] = in_pre;

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
        let mut repr: Vec<usize> = vec![0; num_parts];
        for (pid, block) in blocks.iter().enumerate() { if !block.is_empty() { repr[pid] = block[0]; } }

        let finals2 = (0..num_parts).map(|pid| dfa.finals[repr[pid]]).collect();
        let mut trans2 = vec![vec![0; a]; num_parts];
        for pid in 0..num_parts {
            let s = repr[pid];
            for sym in 0..a {
                let v = dfa.trans[s][sym];
                trans2[pid][sym] = part_id[v];
            }
        }

        let quotient_dfa = DetDFA {
            n_states: num_parts,
            start: 0, // Placeholder
            finals: finals2,
            trans: trans2,
        };
        (quotient_dfa, part_id)
    }
}

/* ------------------------------
   DWA Construction from Shared DFA
   ------------------------------ */

impl DWA {
    fn from_shared_dfa(shared: &SharedDFA, sigma: &Alphabet, atoms: &WeightPartition) -> Self {
        let k = shared.atom_starts.len();
        let mut dwa = DWA::new();
        dwa.states.0.clear();

        let mut map: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut q = VecDeque::new();

        let start_tuple = shared.atom_starts.clone();

        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new_spinner();
            p.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] Building DWA from shared graph... States found: {pos}")
                    .unwrap(),
            );
            Some(p)
        } else {
            None
        };

        let start_id = dwa.add_state();
        map.insert(start_tuple.clone(), start_id);
        q.push_back((start_id, start_tuple));
        dwa.body.start_state = start_id;
        if let Some(p) = &pb { p.set_position(dwa.states.len() as u64); }

        let atom_weights = &atoms.atoms;

        while let Some((u_id, u_tuple)) = q.pop_front() {
            let mut w_final = Weight::zeros();
            let mut w_active = Weight::zeros();

            for atom_idx in 0..k {
                let shared_state_id = u_tuple[atom_idx];
                // Find the specific info for this atom_idx within the shared state's info vec
                if let Some(&(_, is_final, is_sink)) = shared.state_info[shared_state_id].iter().find(|&&(orig_atom, _, _)| orig_atom == atom_idx) {
                    if is_final { w_final |= &atom_weights[atom_idx]; }
                    if !is_sink { w_active |= &atom_weights[atom_idx]; }
                }
            }

            if !w_final.is_empty() {
                dwa.states[u_id].final_weight = Some(w_final);
            }
            let edge_weight = if w_active.is_empty() { Weight::zeros() } else { w_active };

            for sym in 0..sigma.size() {
                let next_tuple: Vec<usize> = u_tuple.iter().map(|&s| shared.dfa.trans[s][sym]).collect();

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

        if let Some(p) = pb {
            p.finish_with_message(format!("DWA construction from shared graph done, {} states", dwa.states.len()));
        }

        dwa
    }
}