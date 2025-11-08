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

/// This determinization is a complete rewrite. It follows a principled construction:
/// 1) Extract a minimal disjoint partition of the weight space (atoms), by splitting at
///    all range boundaries appearing in any edge/default/epsilon/final weight of the NWA.
///    Each "atom" is a contiguous usize interval; every original weight is a union of atoms.
/// 2) For each atom, build an unweighted NFA by keeping only those edges/defaults/epsilons/finals
///    whose weights intersect that atom. Determinize this NFA to a DFA over a finite alphabet
///    Sigma' = { all labels appearing anywhere in the NWA } ∪ { OTHER }, where OTHER means:
///      "use state's default targets when there is no exception for that label."
///    Then minimize the DFA (Hopcroft).
/// 3) Combine all atom-DFAs into a single deterministic product DFA by parallel composition
///    across components (one component per atom). The product DFA has a single start state
///    (tuple of per-atom starts) and reads the original word labels (from Sigma').
/// 4) Convert the product DFA into a DWA:
///    - Each product state P = (q_0, q_1, ..., q_{k-1}) has an "active weight" equal to the union of
///      atom-weights for components i where q_i is not the component's sink state (if any).
///      We attach this active weight to all outgoing edges from P.
///    - Each labeled edge (exception label) carries the active weight; the transition target is deterministic.
///    - The default edge (OTHER) also carries the active weight.
///    - The final weight of P is the union of atom-weights for components in which q_i is accepting.
///    This construction yields a correct DWA: for any input word, each atom either remains "alive"
///    (never falls into its component sink) and ends in a final component, contributing that atom's range
///    to the final weight; otherwise it is filtered out by the final state's weight.
///
/// This DWA is then simplified (minimized structurally and normalized) using the existing passes.
impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now_total = Instant::now();

        // Always work on a simplified copy of the NWA to avoid redundant structure.
        let mut nwa = self.clone();
        debug_log(3, || format!("Starting determinization for NWA with {} states", nwa.states.len()));

        // Compute future-acceptance masks once. We will use them to prune atoms and shrink the alphabet.
        let fut = nwa.compute_future_weights();

        // 1) Build the atomic partition of weights (disjoint contiguous ranges).
        let now_atoms = Instant::now();
        let atoms = WeightPartition::from_nwa(&nwa);
        debug_log(4, || {
            format!("Built weight partition with {} atoms in {:?}", atoms.intervals.len(), now_atoms.elapsed())
        });

        // 2) Build global alphabet (all labels used anywhere) + OTHER
        let now_sigma = Instant::now();
        // Filter the alphabet using the future-acceptance masks:
        // we only keep labels that can contribute to acceptance along some path
        let sigma = Alphabet::from_nwa_with_future(&nwa, &fut);
        debug_log(4, || {
            format!("Built alphabet with {} labels in {:?}", sigma.labels.len(), now_sigma.elapsed())
        });

        // 3) For each atom, build and minimize a DFA.
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

        // Edge case: If no atoms (i.e., union of all weights is empty), return an empty DWA quickly.
        if atoms.intervals.is_empty() {
            // No weight can ever be produced: produce a trivial 1-state DWA (non-final, no edges).
            let mut dwa = DWA::new();
            return dwa;
        }

        // 4) Build product DFA over components (parallel composition).
        let now_product = Instant::now();
        let product = ProductDFA::from_components(&comp_dfas, &sigma);
        debug_log(4, || {
            format!(
                "Product DFA: states={}, alphabet_size={}, time={:?}",
                product.n_states, sigma.size(), now_product.elapsed()
            )
        });

        // 5) Convert product DFA to DWA (attach weights).
        let now_convert = Instant::now();
        let dwa = product.to_dwa(&sigma, &atoms, &comp_dfas, &comp_sinks);
        debug_log(4, || {
            format!("Product->DWA conversion took {:?}", now_convert.elapsed())
        });

        // 6) Simplify DWA
        let mut dwa = dwa;

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
/// OTHER means "use default transitions"
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
        let other_index = labels.len(); // last slot is OTHER
        Alphabet { labels, other_index }
    }

    /// Build an alphabet filtered by future-acceptance masks:
    /// only keep labels that can actually contribute to acceptance along some path.
    /// A label is kept if there exists some edge s --lbl/w--> t with (w ∧ F[t]) ≠ ∅,
    /// or a default s --wdef--> t with (wdef ∧ F[t]) ≠ ∅, in which case we keep the defaults'
    /// exception labels too (they must be separated from OTHER).
    fn from_nwa_with_future(nwa: &NWA, fut: &Vec<Weight>) -> Self {
        let mut set = BTreeSet::new();
        for (s, st) in nwa.states.0.iter().enumerate() {
            // labeled transitions
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
            // defaults: if relevant, keep their exception labels
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
/// Each atom is a RangeInclusive<usize>. Every original weight is a union of some subset of atoms.
///
/// Construction:
/// - Gather all start points s and all (end+1) boundaries for each range in every weight.
/// - Sort and dedup.
/// - Create intervals [b[i], b[i+1]-1] for i=0..len-2.
/// - If any weight has an interval ending at usize::MAX, track that the final tail exists
///   and create the last atom [b_last, usize::MAX].
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
            // final weight
            if let Some(w) = &st.final_weight {
                if !w.is_empty() {
                    feed_weight(w);
                }
            }
            // epsilons
            for (_, w) in &st.epsilons {
                if !w.is_empty() {
                    feed_weight(w);
                }
            }
            // defaults
            for def in &st.default {
                if !def.weight.is_empty() {
                    feed_weight(&def.weight);
                }
            }
            // exceptions
            for (_, targets) in &st.transitions {
                for (_, w) in targets {
                    if !w.is_empty() {
                        feed_weight(w);
                    }
                }
            }
        }

        // If there are no weights at all, the partition is empty.
        if starts.is_empty() && ends_plus.is_empty() && !has_tail_to_max {
            return WeightPartition { intervals: vec![], atoms: vec![] };
        }

        // Combine and sort all "breakpoints" (start and end+1).
        let mut breaks: Vec<usize> = starts.union(&ends_plus).copied().collect();
        breaks.sort_unstable();
        breaks.dedup();
        if breaks.is_empty() {
            // Only possible if there was at least one ALL weight: single atom [0..=usize::MAX]
            let singleton = 0usize..=usize::MAX;
            let atom_w: Weight = std::iter::once(singleton.clone()).collect();
            return WeightPartition { intervals: vec![singleton], atoms: vec![atom_w] };
        }

        // Build atoms between consecutive breakpoints; if tail-to-max exists, include final segment.
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

        // Cache atom Weights
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
/// - Alphabet is Sigma' (all labels + OTHER); on label 'l', if a state has an exception for l,
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

    /// Epsilon closure precomputation for each single state.
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

    /// Given a set of states, compute epsilon closure (using per-state closures).
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
            for (sym, dst) in row.iter().enumerate() {
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
        // Remove states unreachable from start first
        let reachable = {
            let mut visited = vec![false; self.n_states];
            let mut q = VecDeque::new();
            visited[self.start] = true;
            q.push_back(self.start);
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

    /// Find an index of a sink state (if any), defined as a non-accepting state whose transitions on all symbols loop to itself.
    fn find_sink_index(&self, sigma: &Alphabet) -> Option<usize> {
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
   Product DFA and conversion to DWA
   ------------------------------ */

#[derive(Clone, Debug)]
struct ProductDFA {
    n_states: usize,
    start: usize,
    finals: Vec<bool>,
    trans: Vec<Vec<usize>>, // [state][symbol] -> next state
    // Additionally keep the tuple of component states for each product state:
    tuples: Vec<Vec<usize>>, // tuples[state] = [q0, q1, ..., q_{k-1}]
}
impl ProductDFA {
    fn from_components(comps: &[DetDFA], sigma: &Alphabet) -> Self {
        let k = comps.len();
        // Starting tuple: [start0, start1, ..., start_{k-1}]
        let mut start_tuple = Vec::with_capacity(k);
        for c in comps {
            start_tuple.push(c.start);
        }

        let mut map: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut tuples: Vec<Vec<usize>> = Vec::new();
        let mut finals: Vec<bool> = Vec::new();
        let mut trans: Vec<Vec<usize>> = Vec::new();

        let pb_product = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new_spinner();
            p.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] Building product DFA... States found: {pos}")
                    .unwrap(),
            );
            Some(p)
        } else {
            None
        };

        let mut intern = |tuple: Vec<usize>,
                          map: &mut HashMap<Vec<usize>, usize>,
                          tuples: &mut Vec<Vec<usize>>,
                          finals: &mut Vec<bool>,
                          trans: &mut Vec<Vec<usize>>| {
            if let Some(&id) = map.get(&tuple) {
                return id;
            }
            let id = tuples.len();
            let is_final = tuple.iter().enumerate().any(|(i, &s)| comps[i].finals[s]);
            tuples.push(tuple.clone());
            finals.push(is_final);
            trans.push(vec![0usize; sigma.size()]);
            map.insert(tuple, id);
            id
        };

        let start = intern(start_tuple, &mut map, &mut tuples, &mut finals, &mut trans);
        if let Some(p) = &pb_product {
            p.set_position(tuples.len() as u64);
        }

        let mut q = VecDeque::new();
        q.push_back(start);

        while let Some(u) = q.pop_front() {
            let tuple = tuples[u].clone();
            for sym in 0..sigma.size() {
                // build next tuple by composing per-component transitions
                let mut next_tuple = Vec::with_capacity(k);
                for (i, comp) in comps.iter().enumerate() {
                    let s = tuple[i];
                    let v = comp.trans[s][sym];
                    next_tuple.push(v);
                }
                let v = if let Some(&id) = map.get(&next_tuple) {
                    id
                } else {
                    let id = tuples.len();
                    let is_final = next_tuple.iter().enumerate().any(|(i, &s)| comps[i].finals[s]);
                    tuples.push(next_tuple.clone());
                    finals.push(is_final);
                    trans.push(vec![0usize; sigma.size()]);
                    map.insert(next_tuple, id);
                    if let Some(p) = &pb_product {
                        p.set_position(tuples.len() as u64);
                    }
                    q.push_back(id);
                    id
                };
                trans[u][sym] = v;
            }
        }

        if let Some(p) = pb_product {
            p.finish_with_message(format!("Product DFA built with {} states", tuples.len()));
        }

        ProductDFA {
            n_states: tuples.len(),
            start,
            finals,
            trans,
            tuples,
        }
    }

    /// Convert this product DFA into a DWA:
    /// - One DWA state per product DFA state.
    /// - Edge weights: active indices at source state (union over atoms whose component is not sink).
    /// - Final weights: union over atoms whose component is accepting at that state.
    fn to_dwa(&self, sigma: &Alphabet, atoms: &WeightPartition, comps: &[DetDFA], comp_sinks: &[Option<usize>]) -> DWA {
        let mut dwa_states = DWAStates::default();
        for _ in 0..self.n_states {
            dwa_states.add_state();
        }
        let mut dwa = DWA { states: dwa_states, body: DWABody { start_state: self.start } };

        // Precompute per-atom weight
        let atom_weights: &Vec<Weight> = &atoms.atoms;

        let pb_convert = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(self.n_states as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Attaching DWA weights)")
                    .unwrap(),
            );
            Some(p)
        } else {
            None
        };

        // Helper: compute W_live(state_id) = union of atoms for components that are not in sink at this state.
        let mut w_live_cache: Vec<Weight> = Vec::with_capacity(self.n_states);
        for sid in 0..self.n_states {
            let mut w = Weight::zeros();
            for (i, &s_i) in self.tuples[sid].iter().enumerate() {
                if let Some(sink_idx) = comp_sinks[i] {
                    if s_i == sink_idx {
                        continue;
                    }
                }
                // add atom i
                w |= &atom_weights[i];
            }
            w_live_cache.push(w);
        }

        // Final weights
        for sid in 0..self.n_states {
            if !self.finals[sid] {
                continue;
            }
            let mut w_final = Weight::zeros();
            for (i, &s_i) in self.tuples[sid].iter().enumerate() {
                if comps[i].finals[s_i] {
                    w_final |= &atom_weights[i];
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
            // Always emit transitions to keep the DWA total. Use zero weight if no atom is live.
            let edge_weight = if w_live_cache[sid].is_empty() {
                Weight::zeros()
            } else {
                w_live_cache[sid].clone()
            };

            // Default (OTHER)
            let dst_def = self.trans[sid][sigma.other_index];
            if sid < dwa.states.len() && dst_def < dwa.states.len() {
                // Create/overwrite default transition
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
}

/* ------------------------------
   End of determinization.rs
   ------------------------------ */
