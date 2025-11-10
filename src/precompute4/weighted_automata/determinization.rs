#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::tuple_merger;
use super::bitset::SimpleBitset as Weight;
use super::dwa::{DWAState, DWAStates, DWA, DWABody};
use super::nwa::{NWA, NWAStates};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};


use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::RangeInclusive;
use std::time::Instant;
use crate::precompute4::weighted_automata::format_i16_char;

/// This determinization implements a tuple-based merging construction:
/// 1) Partition weight space into atoms.
/// 2) For each atom, determinize the corresponding unweighted NFA to a DFA.
/// 3) Enumerate the set of reachable product tuples T = Vec<Option<usize>> of component DFA states,
///    where None represents the component's sink state.
/// 4) Merge tuples greedily into product states by unifying on non-sink positions:
///    two tuples are mergeable iff for each index either both are Some with equal values or at least one is None.
///    The representative tuple of a merged state is the pointwise "most-specific" unify.
/// 5) For each merged state, compute:
///    - final weight = union of atom-weights for indices whose representative component is accepting.
///    - transitions from representative tuple on the global alphabet (labels + OTHER).
///      Store per-label destination tuple, and for OTHER store a default destination tuple.
/// 6) Convert merged product directly to a DWA:
///    - For a transition with destination tuple U, the edge weight is the union of atom-weights for i with U[i] != None.
///    - Final weight as computed in (5).
impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        self.determinize_to_dwa_internal(false)
    }

    pub fn determinize_to_dwa_with_final_state_simplification(&self) -> DWA {
        self.determinize_to_dwa_internal(true)
    }

    fn determinize_to_dwa_internal(&self, simplify_final_states: bool) -> DWA {
        let now_total = Instant::now();

        // Work on a copy to avoid side effects.
        let mut nwa = self.clone();
        crate::debug!(3, "Starting determinization for NWA with {} states", nwa.states.len());

        // Compute future-acceptance masks (used to filter alphabet).
        let fut = nwa.compute_future_weights();

        // 1) Weight atoms
        let now_atoms = Instant::now();
        let atoms = WeightPartition::from_nwa(&nwa);
        crate::debug!(
            4,
            "Built weight partition with {} atoms in {:?}",
            atoms.intervals.len(),
            now_atoms.elapsed()
        );

        // 2) Alphabet (labels + OTHER), filtered by future weights
        let now_sigma = Instant::now();
        let sigma = Alphabet::from_nwa_with_future(&nwa, &fut);
        crate::debug!(4, "Built alphabet with {} labels in {:?}", sigma.labels.len(), now_sigma.elapsed());

        // 3) Per-atom DFAs
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
            let mut dfa = nfa.determinize(&sigma, simplify_final_states);
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

        if 5 <= crate::r#macro::get_macro_debug_level() {
            println!("\n--- Atomic DFAs ({} total) ---", comp_dfas.len());
            for (i, dfa) in comp_dfas.iter().enumerate() {
                println!("  DFA {} (for atom {:?}):", i, atoms.intervals[i]);
                println!("    - States: {}, Start: {}, Sink: {:?}", dfa.n_states, dfa.start, comp_sinks[i]);
                for s in 0..dfa.n_states {
                    let final_marker = if dfa.finals[s] { " (final)" } else { "" };
                    let mut trans_parts = vec![];
                    for (sym_idx, &label) in sigma.labels.iter().enumerate() {
                        let target = dfa.trans[s][sym_idx];
                        trans_parts.push(format!("{}->{}", super::common::format_i16_char(label), target));
                    }
                    let other_target = dfa.trans[s][sigma.other_index];
                    trans_parts.push(format!("*->{}", other_target));
                    println!("    - State {}{}: {}", s, final_marker, trans_parts.join(", "));
                }
            }
        }

        // Edge case: No atoms => no weight can be produced
        if atoms.intervals.is_empty() {
            return DWA::new();
        }

        // 4) Enumerate reachable product tuples (Vec<Option<usize>>)
        let now_enum = Instant::now();
        let k = comp_dfas.len();
        let mut start_tuple = Vec::with_capacity(k);
        for i in 0..k {
            let s = comp_dfas[i].start;
            if let Some(sink) = comp_sinks[i] {
                if s == sink {
                    start_tuple.push(None);
                } else {
                    start_tuple.push(Some(s));
                }
            } else {
                start_tuple.push(Some(s));
            }
        }
        crate::debug!(
            4,
            "Computed start tuple in {:?}",
            now_enum.elapsed()
        );

        // 5) Merge tuples greedily and attach transitions/finals from representative tuples
        let now_merge = Instant::now();
        let merger_components: Vec<tuple_merger::Component> = comp_dfas.iter().zip(comp_sinks.iter()).map(|(dfa, sink)| {
            let mut sparse_trans = Vec::with_capacity(dfa.n_states);
            for s in 0..dfa.n_states {
                let mut exceptions = BTreeMap::new();
                if let Some(sink_idx) = sink {
                    for (sym, &target) in dfa.trans[s].iter().enumerate() {
                        if target != *sink_idx {
                            exceptions.insert(sym, target);
                        }
                    }
                } else {
                    // No sink, so all transitions are explicit.
                    for (sym, &target) in dfa.trans[s].iter().enumerate() {
                        exceptions.insert(sym, target);
                    }
                }
                sparse_trans.push(exceptions);
            }
            tuple_merger::Component { transitions: sparse_trans }
        }).collect();

        let merged_automaton = tuple_merger::merge_and_build_automaton(
            start_tuple,
            &merger_components,
            sigma.size(),
        );
        crate::debug!(
            4,
            "Merged into {} product states in {:?}",
            merged_automaton.states.len(),
            now_merge.elapsed()
        );

        // 6) Convert merged product to a DWA
        let now_convert = Instant::now();
        let dwa = build_dwa_from_merged_automaton(&merged_automaton, &comp_dfas, &atoms.atoms, &sigma);
        crate::debug!(
            4,
            "Merged-product -> DWA conversion completed in {:?}",
            now_convert.elapsed()
        );

        crate::debug!(3, "NWA::determinize_to_dwa total time: {:?}", now_total.elapsed());

        dwa
    }
}

/* ------------------------------
   Utilities and support structs
   ------------------------------ */

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
        for (_s, st) in nwa.states.0.iter().enumerate() {
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
/// Each atom is a RangeInclusive<usize>.
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
/// - Keep only edges/defaults/epsilons/finals whose weight intersects the atom.
/// - Alphabet is Sigma' (labels + OTHER); on label 'l': exceptions for l (if any) and defaults where l not in exceptions.
/// - On OTHER: use defaults.
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

            // default(s) for this atom
            for def in &states[old_s].default {
                let to_old = def.target;
                if to_old < n_total && live[to_old] && !(&atom_w & &def.weight).is_empty() {
                    let to_new = id_of[to_old];
                    if to_new != usize::MAX {
                        def_by_state[new_s].push((to_new, def.exceptions.clone()));
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
    fn determinize(&self, sigma: &Alphabet, simplify_final_states: bool) -> DetDFA {
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
                        // or if none, any default transitions for which lbl is not an exception.
                        for &s in &subset {
                            if let Some(ts) = self.ex_by_state[s].get(&lbl) {
                                next_raw.extend_from_slice(ts);
                            } else {
                                for (target, exceptions) in &self.def_by_state[s] {
                                    if !exceptions.contains(&lbl) {
                                        next_raw.push(*target);
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        // OTHER: take all default transitions
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

        if simplify_final_states {
            // From final states, all transitions go to a sink.
            for s in 0..states.len() {
                if finals[s] {
                    trans[s] = vec![None; sigma.size()];
                }
            }
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
        crate::debug!(4, "Minimizing DFA with {} states", self.n_states);
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

        // Worklist of (block id, symbol).
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
   Tuple enumeration and merging
   ------------------------------ */

/* ------------------------------
   Build DWA from merged product
   ------------------------------ */

fn edge_weight_from_tuple(atom_weights: &Vec<Weight>, tuple: &tuple_merger::ProductTuple) -> Weight {
    let mut w = Weight::zeros();
    for (i, pos) in tuple.iter().enumerate() {
        if pos.is_some() {
            w |= &atom_weights[i];
        }
    }
    w
}

fn build_dwa_from_merged_automaton(
    merged_automaton: &tuple_merger::MergedAutomaton,
    comps: &[DetDFA],
    atom_weights: &Vec<Weight>,
    sigma: &Alphabet,
) -> DWA {
    let mut dwa_states = DWAStates::default();
    for _ in 0..merged_automaton.states.len() {
        dwa_states.add_state();
    }
    let mut dwa = DWA { states: dwa_states, body: DWABody { start_state: merged_automaton.start_state_id } };

    // We need a way to compute successor tuples to determine edge weights.
    // Re-create the component structures for the tuple_merger.
    let merger_components: Vec<tuple_merger::Component> = comps.iter().map(|dfa| {
        let sink = dfa.find_sink_index(sigma);
        let mut sparse_trans = Vec::with_capacity(dfa.n_states);
        for s in 0..dfa.n_states {
            let mut exceptions = BTreeMap::new();
            if let Some(sink_idx) = sink {
                for (sym, &target) in dfa.trans[s].iter().enumerate() {
                    if target != sink_idx {
                        exceptions.insert(sym, target);
                    }
                }
            } else {
                for (sym, &target) in dfa.trans[s].iter().enumerate() {
                    exceptions.insert(sym, target);
                }
            }
            sparse_trans.push(exceptions);
        }
        tuple_merger::Component { transitions: sparse_trans }
    }).collect();

    for state in &merged_automaton.states {
        let sid = state.id;
        let rep = &state.representative_tuple;

        // Final weight
        let mut w_final = Weight::zeros();
        for (i, pos) in rep.iter().enumerate() {
            if let Some(s) = pos {
                if comps[i].finals[*s] {
                    w_final |= &atom_weights[i];
                }
            }
        }
        if !w_final.is_empty() {
            let _ = dwa.set_final_weight(sid, w_final);
        }

        // Default (OTHER)
        let def_to_id = state.transitions[sigma.other_index];
        let def_succ_tuple = tuple_merger::successor_tuple(rep, sigma.other_index, &merger_components);
        let def_w = edge_weight_from_tuple(atom_weights, &def_succ_tuple);
        let _ = dwa.set_default_transition(sid, def_to_id, def_w);

        // Exceptions
        for (li, &lbl) in sigma.labels.iter().enumerate() {
            let to_id = state.transitions[li];
            let succ_tuple = tuple_merger::successor_tuple(rep, li, &merger_components);
            let w = edge_weight_from_tuple(atom_weights, &succ_tuple);
            let _ = dwa.add_transition(sid, lbl, to_id, w);
        }
    }

    dwa
}

/* ------------------------------
   End of merged determinization
   ------------------------------ */
