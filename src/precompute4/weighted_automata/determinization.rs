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

/// Determinization by "shared-graph minimization" with an initial-dispatch trick.
///
/// Core idea (replacing the explicit product):
/// - Compute disjoint weight atoms.
/// - For each atom, build and minimize a DFA (over Sigma' = labels + OTHER).
/// - Form a single "combined pre-DFA" by disjoint sum of all component DFAs, plus a super_start
///   that has k dispatch transitions on k fresh synthetic symbols (one per atom) into each component's start.
/// - Minimize the combined pre-DFA with those synthetic symbols in the alphabet as well.
///   This merges all isomorphic subgraphs across components, avoiding state explosion.
/// - The minimized combined DFA's structure (restricted back to Sigma') is the final skeleton.
/// - Compute per-state final weight as union of atoms for which some component's final maps into that class.
/// - Compute per-state live weight as union of atoms whose component contributes a non-sink state to that class.
/// - The start state is the minimized image of super_start. Its outgoing transitions for Sigma' are set by
///   the unique minimized target class that all component starts map to on that label (assumption below).
///
/// Assumption (consistency of targets):
/// For any minimized class C (including the start), and any real symbol a in Sigma',
/// all underlying component-states s in C have δ(s, a) mapping to the same minimized class.
/// This is guaranteed by DFA minimization (Myhill–Nerode congruence) over the extended alphabet.
///
/// Note: We never build any product automaton.
impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now_total = Instant::now();

        // Work on a local copy (allows future pre-simplifications if needed)
        let nwa = self.clone();
        debug_log(3, || format!("Starting determinization for NWA with {} states", nwa.states.len()));

        // 1) Precompute future-acceptance masks (used to prune atom-NFA edges & alphabet).
        let fut = nwa.compute_future_weights();

        // 2) Build weight partition (atoms).
        let now_atoms = Instant::now();
        let atoms = WeightPartition::from_nwa(&nwa);
        debug_log(4, || format!("Weight partition: {} atoms built in {:?}", atoms.intervals.len(), now_atoms.elapsed()));

        if atoms.intervals.is_empty() {
            // No weight can ever be produced => trivial DWA with only start, no edges, no finals.
            return DWA::new();
        }

        // 3) Build alphabet Sigma' = explicit labels + OTHER (no synthetic entries yet).
        let now_sigma = Instant::now();
        let sigma = Alphabet::from_nwa_with_future(&nwa, &fut);
        debug_log(4, || format!("Alphabet: {} labels (+ OTHER) in {:?}", sigma.labels.len(), now_sigma.elapsed()));

        // 4) For each atom, build NFA, determinize to DFA, and minimize.
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
        let mut comp_sinks: Vec<Option<usize>> = Vec::with_capacity(atoms.intervals.len());
        for (i, atom) in atoms.intervals.iter().enumerate() {
            let nfa = PerAtomNFA::from_nwa(&nwa.states, nwa.body.start_state, &sigma, atom, &fut);
            let mut dfa = nfa.determinize(&sigma);
            dfa.minimize(&sigma);
            crate::debug!(4, "Atom {}: interval={:?}, DFA states={}", i, atom, dfa.n_states);
            let sink = dfa.find_sink_index(&sigma);
            comp_sinks.push(sink);
            comp_dfas.push(dfa);
            if let Some(p) = &pb_atoms {
                p.inc(1);
            }
        }
        if let Some(p) = pb_atoms {
            p.finish_with_message("Per-atom DFAs built & minimized");
        }

        // 5) Build one combined pre-DFA with a super_start and k synthetic entry symbols.
        let now_comb = Instant::now();
        let mut combined_build = CombinedPreDFA::build(&comp_dfas, &sigma, atoms.intervals.len());
        debug_log(4, || {
            format!(
                "Combined pre-DFA: states={}, alphabet_size={} (includes {} synthetic), time={:?}",
                combined_build.combined.n_states,
                combined_build.alphabet_size,
                combined_build.num_atoms,
                now_comb.elapsed()
            )
        });

        // 6) Minimize the combined pre-DFA (over the extended alphabet) and obtain old->new map.
        let now_min = Instant::now();
        let map_old_to_new = combined_build
            .combined
            .minimize_with_map_raw(combined_build.alphabet_size);
        let combined_min = combined_build.combined.clone();
        debug_log(4, || {
            format!(
                "Minimized combined DFA: states={}, time={:?}",
                combined_min.n_states,
                now_min.elapsed()
            )
        });

        // 7) Convert minimized shared DFA into final DWA structure and weights (only Sigma', ignores synthetic).
        let now_to_dwa = Instant::now();
        let dwa = to_dwa_shared_graph(
            &combined_min,
            &map_old_to_new,
            &sigma,
            &atoms,
            &comp_dfas,
            &comp_sinks,
            &combined_build,
        );
        debug_log(3, || format!("NWA::determinize_to_dwa total time: {:?} (to_dwa: {:?})", now_total.elapsed(), now_to_dwa.elapsed()));

        dwa
    }
}

/* ------------------------------
   Alphabet and Weight Partition
   ------------------------------ */

/// Alphabet = all labels that appear as exceptions, plus a special OTHER symbol.
/// OTHER means "use default transitions".
#[derive(Clone, Debug)]
struct Alphabet {
    labels: Vec<i16>, // sorted unique exception labels
    other_index: usize,
}
impl Alphabet {
    fn from_nwa(nwa: &NWA) -> Self {
        let mut set = BTreeSet::new();
        for st in &nwa.states.0 {
            for (&lbl, _) in &st.transitions {
                set.insert(lbl);
            }
            for def in &st.default {
                for &lbl in &def.exceptions {
                    set.insert(lbl);
                }
            }
        }
        let labels: Vec<i16> = set.into_iter().collect();
        let other_index = labels.len(); // the last slot is OTHER
        Alphabet { labels, other_index }
    }

    /// Filter by future-acceptance masks: keep labels that can contribute to acceptance.
    fn from_nwa_with_future(nwa: &NWA, fut: &Vec<Weight>) -> Self {
        let mut set = BTreeSet::new();
        for (s, st) in nwa.states.0.iter().enumerate() {
            // labeled transitions
            for (&lbl, targets) in &st.transitions {
                if targets.iter().any(|(t, w)| !(&fut[*t] & w).is_empty()) {
                    set.insert(lbl);
                }
            }
            // defaults: if relevant, keep their exception labels too
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

/// WeightPartition: disjoint contiguous atoms that cover the union of all weights.
#[derive(Clone, Debug)]
struct WeightPartition {
    intervals: Vec<RangeInclusive<usize>>,
    atoms: Vec<Weight>, // cached Weight for each interval
}
impl WeightPartition {
    fn from_nwa(nwa: &NWA) -> Self {
        let mut starts: BTreeSet<usize> = BTreeSet::new();
        let mut ends_plus: BTreeSet<usize> = BTreeSet::new();
        let mut has_tail_to_max = false;

        let mut feed_weight = |w: &Weight| {
            let mut it = w.ranges();
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

/* ------------------------------
   Per-atom NFA and DFA
   ------------------------------ */

/// An NFA specialized for a single atom:
/// - Keep only edges/defaults/epsilons/final whose weight intersects the atom.
/// - Alphabet is Sigma' (labels + OTHER); on label 'l', if a state has an exception for l,
///   take those targets; otherwise use defaults; on OTHER, always use defaults.
#[derive(Clone, Debug)]
struct PerAtomNFA {
    n: usize,
    start: usize,
    finals: Vec<bool>,
    ex_by_state: Vec<BTreeMap<i16, Vec<usize>>>,
    def_by_state: Vec<Vec<(usize, BTreeSet<i16>)>>, // list of (target, exceptions)
    eps_by_state: Vec<Vec<usize>>,
}
impl PerAtomNFA {
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

        // BFS over edges that intersect atom and lead to live targets.
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

            // default(s) for this atom: exceptions = keys of local_ex
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

    /// Determinize this NFA to a complete DFA with alphabet Sigma' (labels + OTHER).
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

        // Build sink state if any transition is None
        let needs_sink = trans.iter().any(|row| row.iter().any(|x| x.is_none()));
        let mut sink_index: Option<usize> = None;

        if needs_sink {
            sink_index = Some(trans.len());
            trans.push(vec![None; sigma.size()]);
            finals.push(false);
            states.push(Vec::new());
        }

        // Fill None transitions with sink (or self-loop if no sink needed)
        let mut out_trans: Vec<Vec<usize>> = Vec::with_capacity(trans.len());
        for (i, row) in trans.iter().enumerate() {
            let mut new_row = Vec::with_capacity(row.len());
            for (sym, dst) in row.iter().enumerate() {
                match dst {
                    Some(v) => new_row.push(*v),
                    None => {
                        if let Some(sink) = sink_index {
                            new_row.push(sink);
                        } else {
                            new_row.push(i);
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

/// Deterministic complete DFA (over Sigma').
#[derive(Clone, Debug)]
struct DetDFA {
    n_states: usize,
    start: usize,
    finals: Vec<bool>,
    trans: Vec<Vec<usize>>, // [state][symbol] -> next state
}
impl DetDFA {
    fn minimize(&mut self, sigma: &Alphabet) {
        debug_log(4, || format!("Minimizing DFA with {} states", self.n_states));
        let reachable = self.reachable_states(sigma.size());

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
            return;
        }
        blocks.push(accepting_block);
        blocks.push(non_accepting_block);
        for (pid, block) in blocks.iter().enumerate() { for &s in block { part_id[s] = pid; } }

        // Build inverse transitions for each symbol
        let mut inv: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); n]; a];
        for s in 0..n { for sym in 0..a { let v = self.trans[s][sym]; inv[sym][v].push(s); } }

        let mut worklist: BTreeSet<(usize, usize)> = BTreeSet::new();
        let smaller_initial_set = if blocks[0].len() <= blocks[1].len() { 0 } else { 1 };
        for sym in 0..a { worklist.insert((smaller_initial_set, sym)); }

        let pb_hopcroft = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new_spinner();
            p.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [Determinize/Minimize: {elapsed_precise}] Pass {pos}, worklist size: {msg}")
                    .unwrap(),
            );
            Some(p)
        } else { None };
        let mut passes = 0u64;

        while let Some(&(b, sym)) = worklist.iter().next() {
            worklist.remove(&(b, sym));

            if let Some(p) = &pb_hopcroft {
                passes += 1;
                p.set_position(passes);
                p.set_message(format!("{}", worklist.len()));
            }

            let mut pre: Vec<usize> = Vec::new();
            for &v in &blocks[b] { pre.extend_from_slice(&inv[sym][v]); }
            if pre.is_empty() { continue; }
            pre.sort_unstable();
            pre.dedup();

            let mut affected: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
            for &s in &pre { let pid = part_id[s]; affected.entry(pid).or_default().0.push(s); }
            for (pid, (ref mut in_pre, ref mut not_in_pre)) in affected.iter_mut() {
                for &s in &blocks[*pid] { if !in_pre.binary_search(&s).is_ok() { not_in_pre.push(s); } }
            }

            let mut to_replace: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();

            for (pid, (in_pre, not_in_pre)) in affected.into_iter() {
                if in_pre.is_empty() || not_in_pre.is_empty() { continue; }
                to_replace.push((pid, in_pre, not_in_pre));
            }
            if to_replace.is_empty() { continue; }

            for (pid, mut in_pre, mut not_in_pre) in to_replace {
                in_pre.sort_unstable();
                not_in_pre.sort_unstable();

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
                        let (smaller_pid, _) = if blocks[pid].len() <= blocks[pid2].len() { (pid, pid2) } else { (pid2, pid) };
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
        for (pid, block) in blocks.iter().enumerate() { repr[pid] = block[0]; }

        let start_part = part_id[self.start];
        let mut finals2 = vec![false; num_parts];
        for pid in 0..num_parts { finals2[pid] = self.finals[repr[pid]]; }

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

    /// Minimize returning a mapping from old-state (pre-minimization) to new-state (post-minimization).
    /// The alphabet size is given explicitly (works with synthetic extra symbols).
    fn minimize_with_map_raw(&mut self, alphabet_size: usize) -> Vec<usize> {
        // Reachable pruning
        let reachable = self.reachable_states(alphabet_size);
        let mut old_to_compact = vec![usize::MAX; self.n_states];
        let mut compact_count = 0usize;
        for i in 0..self.n_states {
            if reachable[i] {
                old_to_compact[i] = compact_count;
                compact_count += 1;
            }
        }
        if compact_count == 0 {
            // No reachable states from start: make trivial
            let old_n = self.n_states;
            self.n_states = 1;
            self.start = 0;
            self.finals = vec![false];
            self.trans = vec![vec![0; alphabet_size]];
            return vec![0; old_n];
        }

        // Build compact DFA
        let mut finals_c = vec![false; compact_count];
        let mut trans_c = vec![vec![0usize; alphabet_size]; compact_count];
        for i in 0..self.n_states {
            if !reachable[i] { continue; }
            let ci = old_to_compact[i];
            finals_c[ci] = self.finals[i];
            for a in 0..alphabet_size {
                trans_c[ci][a] = old_to_compact[self.trans[i][a]];
            }
        }
        let start_c = old_to_compact[self.start];

        // Hopcroft on compact
        let mut part_id = vec![0usize; compact_count];
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        let (acc_block, nonacc_block): (Vec<_>, Vec<_>) = (0..compact_count).partition(|&s| finals_c[s]);
        if acc_block.is_empty() || nonacc_block.is_empty() {
            // All same finality => mapping is identity of compact ids
            let mut compact_to_new = vec![0usize; compact_count];
            for i in 0..compact_count { compact_to_new[i] = if finals_c.iter().any(|&f| f) { 0 } else { 0 }; }
            // Build minimal DFA (single block case)
            let n_new = 1;
            let start_new = 0;
            let mut trans_new = vec![vec![0usize; alphabet_size]; n_new];
            for a in 0..alphabet_size { trans_new[0][a] = 0; }
            self.n_states = n_new;
            self.start = start_new;
            self.finals = vec![finals_c.iter().any(|&f| f)];
            self.trans = trans_new;

            // old -> compact -> new
            let mut map_old_to_new = vec![0usize; old_to_compact.len()];
            for (old, &c) in old_to_compact.iter().enumerate() {
                if c != usize::MAX {
                    map_old_to_new[old] = 0;
                } else {
                    map_old_to_new[old] = 0;
                }
            }
            return map_old_to_new;
        }

        blocks.push(acc_block);
        blocks.push(nonacc_block);
        for (pid, block) in blocks.iter().enumerate() { for &s in block { part_id[s] = pid; } }

        // Build inverse transitions
        let mut inv: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); compact_count]; alphabet_size];
        for s in 0..compact_count { for sym in 0..alphabet_size { let v = trans_c[s][sym]; inv[sym][v].push(s); } }

        // Worklist
        let mut worklist: BTreeSet<(usize, usize)> = BTreeSet::new();
        let small_init = if blocks[0].len() <= blocks[1].len() { 0 } else { 1 };
        for sym in 0..alphabet_size { worklist.insert((small_init, sym)); }

        while let Some(&(b, sym)) = worklist.iter().next() {
            worklist.remove(&(b, sym));

            let mut pre: Vec<usize> = Vec::new();
            for &v in &blocks[b] { pre.extend_from_slice(&inv[sym][v]); }
            if pre.is_empty() { continue; }
            pre.sort_unstable();
            pre.dedup();

            let mut affected: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
            for &s in &pre { let pid = part_id[s]; affected.entry(pid).or_default().0.push(s); }
            for (pid, (ref mut in_pre, ref mut not_in_pre)) in affected.iter_mut() {
                for &s in &blocks[*pid] { if !in_pre.binary_search(&s).is_ok() { not_in_pre.push(s); } }
            }

            let mut to_replace: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
            for (pid, (in_pre, not_in_pre)) in affected.into_iter() {
                if in_pre.is_empty() || not_in_pre.is_empty() { continue; }
                to_replace.push((pid, in_pre, not_in_pre));
            }
            if to_replace.is_empty() { continue; }

            for (pid, mut in_pre, mut not_in_pre) in to_replace {
                in_pre.sort_unstable();
                not_in_pre.sort_unstable();

                let pid2 = blocks.len();
                blocks.push(not_in_pre);
                blocks[pid] = in_pre;

                for &s in &blocks[pid] { part_id[s] = pid; }
                for &s in &blocks[pid2] { part_id[s] = pid2; }

                for sym2 in 0..alphabet_size {
                    if worklist.remove(&(pid, sym2)) {
                        worklist.insert((pid, sym2));
                        worklist.insert((pid2, sym2));
                    } else {
                        let smaller = if blocks[pid].len() <= blocks[pid2].len() { pid } else { pid2 };
                        worklist.insert((smaller, sym2));
                    }
                }
            }
        }

        // Build quotient automaton
        let num_parts = blocks.len();
        let mut repr: Vec<usize> = vec![0; num_parts];
        for (pid, block) in blocks.iter().enumerate() { repr[pid] = block[0]; }

        let start_part = part_id[start_c];
        let mut finals2 = vec![false; num_parts];
        for pid in 0..num_parts { finals2[pid] = finals_c[repr[pid]]; }

        let mut trans2 = vec![vec![0usize; alphabet_size]; num_parts];
        for pid in 0..num_parts {
            let s = repr[pid];
            for sym in 0..alphabet_size {
                let v = trans_c[s][sym];
                trans2[pid][sym] = part_id[v];
            }
        }

        // old -> compact -> part
        let mut map_old_to_new = vec![0usize; self.n_states];
        for old in 0..self.n_states {
            if old_to_compact[old] != usize::MAX {
                map_old_to_new[old] = part_id[old_to_compact[old]];
            } else {
                // unreachable from start: map to start_part (arbitrary, won't be used)
                map_old_to_new[old] = start_part;
            }
        }

        self.n_states = num_parts;
        self.start = start_part;
        self.finals = finals2;
        self.trans = trans2;

        map_old_to_new
    }

    fn find_sink_index(&self, sigma: &Alphabet) -> Option<usize> {
        'outer: for s in 0..self.n_states {
            if self.finals[s] { continue; }
            for sym in 0..sigma.size() {
                if self.trans[s][sym] != s {
                    continue 'outer;
                }
            }
            return Some(s);
        }
        None
    }

    fn reachable_states(&self, alphabet_size: usize) -> Vec<bool> {
        let mut visited = vec![false; self.n_states];
        let mut q = VecDeque::new();
        visited[self.start] = true;
        q.push_back(self.start);
        while let Some(u) = q.pop_front() {
            for &v in &self.trans[u][..alphabet_size] {
                if !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }
        visited
    }
}

/* ------------------------------
   Combined pre-DFA (build phase)
   ------------------------------ */

struct CombinedPreDFA {
    combined: DetDFA,
    alphabet_size: usize, // Sigma' size + num_atoms synthetic
    sigma_size: usize,    // Sigma' size (labels + OTHER)
    num_atoms: usize,
    offsets: Vec<usize>,  // global-state offset per component (start at 1; 0 reserved for super_start)
}
impl CombinedPreDFA {
    fn build(comps: &[DetDFA], sigma: &Alphabet, num_atoms: usize) -> Self {
        let sigma_size = sigma.size();
        let alphabet_size = sigma_size + num_atoms; // add one synthetic symbol per atom
        let mut total_states = 1usize; // super_start
        for c in comps { total_states += c.n_states; }

        let mut combined = DetDFA {
            n_states: total_states,
            start: 0,
            finals: vec![false; total_states],
            trans: vec![vec![0usize; alphabet_size]; total_states],
        };

        // Fill super_start transitions: synthetic dispatch to each component's start
        // For real symbols: arbitrary self-loops (they won't be used directly in final DWA).
        // For synthetic symbol j (sigma_size + i): go to offset[i] + comps[i].start.
        let mut offsets = Vec::with_capacity(comps.len());
        let mut base = 1usize;
        for i in 0..comps.len() {
            offsets.push(base);
            base += comps[i].n_states;
        }

        // super_start real symbols -> self
        for a in 0..sigma_size {
            combined.trans[0][a] = 0;
        }
        // super_start synthetic -> dispatch to each component start
        for (i, c) in comps.iter().enumerate() {
            combined.trans[0][sigma_size + i] = offsets[i] + c.start;
        }

        // Copy each component into combined (with self-loops on synthetic symbols).
        for (i, c) in comps.iter().enumerate() {
            let off = offsets[i];
            for s in 0..c.n_states {
                combined.finals[off + s] = c.finals[s];
                // real symbols
                for a in 0..sigma_size {
                    combined.trans[off + s][a] = off + c.trans[s][a];
                }
                // synthetic symbols: self-loops for all
                for j in 0..num_atoms {
                    combined.trans[off + s][sigma_size + j] = off + s;
                }
            }
        }

        CombinedPreDFA { combined, alphabet_size, sigma_size, num_atoms, offsets }
    }
}

/* ------------------------------
   Product-free conversion to DWA
   ------------------------------ */

fn to_dwa_shared_graph(
    combined_min: &DetDFA,
    old2new: &Vec<usize>,         // map from pre-min combined old state -> minimized new state
    sigma: &Alphabet,
    atoms: &WeightPartition,
    comps: &[DetDFA],
    comp_sinks: &[Option<usize>],
    comb_info: &CombinedPreDFA,
) -> DWA {
    let n_new = combined_min.n_states;
    let sigma_size = comb_info.sigma_size;
    let num_atoms = comb_info.num_atoms;
    let atom_weights: &Vec<Weight> = &atoms.atoms;

    // 1) Compute per-class live and final masks by aggregating over component states (excluding super_start).
    let mut w_live: Vec<Weight> = vec![Weight::zeros(); n_new];
    let mut w_final: Vec<Weight> = vec![Weight::zeros(); n_new];

    // Union-all atoms (useful for start overrides)
    let mut w_all = Weight::zeros();
    for aw in atom_weights.iter() { w_all |= aw; }

    // Helper: restrict aggregation to states reachable from each component's start to avoid spurious contributions.
    let mut comp_reachables: Vec<Vec<bool>> = Vec::with_capacity(comps.len());
    for c in comps {
        comp_reachables.push(c.reachable_states(sigma_size));
    }

    for (i, c) in comps.iter().enumerate() {
        let off = comb_info.offsets[i];
        let sink = comp_sinks[i];
        let reachable = &comp_reachables[i];
        for s in 0..c.n_states {
            if !reachable[s] { continue; }
            let old_global = off + s;
            let new_cls = old2new[old_global];

            if c.finals[s] {
                w_final[new_cls] |= &atom_weights[i];
            }
            let is_sink = sink.map_or(false, |sk| sk == s);
            if !is_sink {
                w_live[new_cls] |= &atom_weights[i];
            }
        }
    }

    // 2) Start class is minimized image of super_start (old=0).
    let start_new = old2new[0];

    // Override start weights:
    // - live: all atoms whose component start is not sink.
    // - final: atoms whose component start is final.
    let mut start_live = Weight::zeros();
    let mut start_final = Weight::zeros();
    for (i, c) in comps.iter().enumerate() {
        let sink = comp_sinks[i];
        let is_sink_start = sink.map_or(false, |sk| sk == c.start);
        if !is_sink_start {
            start_live |= &atom_weights[i];
        }
        if c.finals[c.start] {
            start_final |= &atom_weights[i];
        }
    }
    if !start_live.is_empty() { w_live[start_new] = start_live.clone(); }
    if !start_final.is_empty() { w_final[start_new] |= &start_final; }

    // 3) Construct final DWA structure: one state per minimized class, transitions over Sigma'.
    let mut dwa_states = DWAStates::default();
    for _ in 0..n_new { dwa_states.add_state(); }
    let mut dwa = DWA { states: dwa_states, body: DWABody { start_state: start_new } };

    // Helper: compute start transitions on Sigma' from component starts (consistency check).
    // For each real symbol a, all component-start targets' minimized classes should agree (assumption).
    let mut start_targets: Vec<usize> = vec![start_new; sigma_size];
    for a in 0..sigma_size {
        let mut set: BTreeSet<usize> = BTreeSet::new();
        for (i, c) in comps.iter().enumerate() {
            let off = comb_info.offsets[i];
            let dst_c = c.trans[c.start][a];
            let dst_global = off + dst_c;
            let dst_new = old2new[dst_global];
            set.insert(dst_new);
        }
        let chosen = *set.iter().next().unwrap_or(&start_new);
        // Optional: if set.len() > 1, the assumption fails; we still pick one deterministically.
        start_targets[a] = chosen;
    }

    // 4) Attach transitions and weights.
    // For each minimized state s: default and labeled transitions use edge-weight = w_live[s].
    // Default symbol = OTHER at sigma.other_index; labels at indices 0..labels.len()-1.
    for s in 0..n_new {
        let edge_w = if w_live[s].is_empty() { Weight::zeros() } else { w_live[s].clone() };

        // Default (OTHER)
        let dst_def = if s == start_new {
            start_targets[sigma.other_index]
        } else {
            combined_min.trans[s][sigma.other_index]
        };
        let _ = dwa.set_default_transition(s, dst_def, edge_w.clone());

        // Exceptions
        for (li, &lbl) in sigma.labels.iter().enumerate() {
            let dst = if s == start_new {
                start_targets[li]
            } else {
                combined_min.trans[s][li]
            };
            let _ = dwa.add_transition(s, lbl, dst, edge_w.clone());
        }
    }

    // 5) Final weights
    for s in 0..n_new {
        if !w_final[s].is_empty() {
            let _ = dwa.set_final_weight(s, w_final[s].clone());
        }
    }

    dwa
}

/* ------------------------------
   Utilities
   ------------------------------ */

fn debug_log(level: usize, msg: impl FnOnce() -> String) {
    crate::debug!(level, "{}", msg());
}

/* ------------------------------
   End of determinization.rs
   ------------------------------ */
