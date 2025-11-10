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
use crate::precompute4::weighted_automata::format_i16_char;

/// This determinization implements a tuple-based merging construction:
/// 1) Partition weight space into atoms.
/// 2) For each atom, determinize the corresponding unweighted NFA to a DFA.
/// 3) Enumerate the set of reachable product tuples T = Vec<Option<usize>> of component DFA states,
///    where None represents the component's sink state.
/// 4) Compute the coarsest congruence (minimal quotient) of this deterministic Moore machine:
///    two tuples are equivalent iff they have identical final outputs and, for every symbol,
///    identical transition outputs and transitions into equivalent classes. We build this
///    on-the-fly using signature-based union-find with predecessor-triggered refinements.
/// 5) For each quotient state, compute:
///    - final weight = union of atom-weights for indices whose representative component is accepting.
///    - transitions from a representative tuple on the global alphabet (labels + OTHER),
///      attached directly as (to_group_id, weight) pairs.
/// 6) Convert merged product directly to a DWA:
///    - For a transition with destination group, the edge weight is the union of atom-weights for i alive after that symbol.
///    - Final weight as computed in (5).
impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
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
        let merged = minimize_tuples_to_states(
            start_tuple,
            &comp_dfas,
            &sigma,
            &comp_sinks,
            &atoms.atoms,
        );
        crate::debug!(
            4,
            "Merged into {} product states in {:?}",
            merged.states.len(),
            now_merge.elapsed()
        );
        if 5 <= crate::r#macro::get_macro_debug_level() {
            // Helper to format tuples like [0, _, 2] instead of [Some(0), None, Some(2)]
            let format_tuple = |tuple: &ProductDFAStateTuple| -> String {
                let parts: Vec<String> = tuple
                    .iter()
                    .map(|opt| match opt {
                        Some(v) => v.to_string(),
                        None => "_".to_string(),
                    })
                    .collect();
                format!("[{}]", parts.join(", "))
            };

            println!("\n--- MergedProduct ({} states, start={}) ---", merged.states.len(), merged.start);
            for (gid, state) in merged.states.iter().enumerate() {
                println!("  State {}:", gid);
                println!("    - Representative: {}", format_tuple(&state.representative_tuple));
                if let Some(w) = &state.final_weight {
                    println!("    - Final Weight: {}", w);
                }
                println!("    - Transitions:");
                println!("      - Default -> State {} (weight: {})", state.trans_default_gid, state.trans_default_weight);

                for (lbl, (gid_to, w)) in &state.trans_exceptions {
                    let char_repr = super::common::format_i16_char(*lbl);
                    println!("      - {} -> State {} (weight: {})", char_repr, gid_to, w);
                }
                println!("    - Contains {} tuples.", state.all_tuples.len());
            }
        }

        // 6) Convert merged product to a DWA
        let now_convert = Instant::now();
        let dwa = build_dwa_from_merged(&merged, &atoms.atoms, &sigma);
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

/// Lightweight 64-bit fingerprint for Weight (SimpleBitset) by hashing its ranges.
/// We also verify equality with full Weight comparisons when merging, so collisions
/// cannot produce incorrect merges; this is only used as a map key accelerator.
fn weight_fingerprint(w: &Weight) -> u64 {
    // Some constants for simple mixing
    let mut h: u64 = 0x9E3779B97F4A7C15;
    let mut it = w.rsb.ranges();
    while let Some(r) = it.next() {
        let s = *r.start() as u64;
        let e = *r.end() as u64;
        h ^= s.wrapping_mul(0x9E3779B185EBCA87).rotate_left(13);
        h ^= e.wrapping_mul(0xC2B2AE3D27D4EB4F).rotate_left(7);
        h = h.wrapping_mul(0x165667B19E3779F9);
    }
    h
}

/* ------------------------------
   Tuple enumeration and optimal merging (minimal quotient)
   ------------------------------ */

type ProductDFAStateTuple = Vec<Option<usize>>;

/// Given a product tuple and a symbol, compute the successor tuple:
/// - If a component is None (sink), it remains None.
/// - Otherwise follow the transition; if it goes to the component's sink, record None; else Some(next).
fn successor_tuple(
    tuple: &ProductDFAStateTuple,
    sym: usize,
    comps: &[DetDFA],
    comp_sinks: &[Option<usize>],
) -> ProductDFAStateTuple {
    let k = comps.len();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        match tuple[i] {
            Some(s) => {
                let v = comps[i].trans[s][sym];
                if let Some(sink) = comp_sinks[i] {
                    if v == sink {
                        out.push(None);
                    } else {
                        out.push(Some(v));
                    }
                } else {
                    out.push(Some(v));
                }
            }
            None => out.push(None),
        }
    }
    out
}

/// Unify two tuples pointwise:
/// - If both Some(a) and Some(b) with a != b => None (incompatible)
/// - If one is Some and the other None => result Some(value)
/// - If both None => None
fn unify_tuples(a: &ProductDFAStateTuple, b: &ProductDFAStateTuple) -> Option<ProductDFAStateTuple> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = a.clone();
    for i in 0..a.len() {
        match (out[i], b[i]) {
            (Some(x), Some(y)) => {
                if x != y {
                    return None;
                }
            }
            (Some(_), None) => {}
            (None, Some(y)) => {
                out[i] = Some(y);
            }
            (None, None) => {}
        }
    }
    Some(out)
}

#[derive(Clone, Debug)]
struct ProductDFAState {
    representative_tuple: ProductDFAStateTuple,
    all_tuples: BTreeSet<ProductDFAStateTuple>,
    final_weight: Option<Weight>,
    // Labeled transitions: label -> (to_group_id, weight)
    trans_exceptions: BTreeMap<i16, (usize, Weight)>,
    // Default (OTHER) transition: (to_group_id, weight)
    trans_default_gid: usize,
    trans_default_weight: Weight,
}

impl Default for ProductDFAState {
    fn default() -> Self {
        Self {
            representative_tuple: Vec::new(),
            all_tuples: BTreeSet::new(),
            final_weight: None,
            trans_exceptions: BTreeMap::new(),
            trans_default_gid: 0,
            trans_default_weight: Weight::zeros(),
        }
    }
}

#[derive(Clone, Debug)]
struct MergedProduct {
    states: Vec<ProductDFAState>,
    start: usize,
}

/// Simple union-find structure for merging groups by signature.
#[derive(Default)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn make_set(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        self.rank.push(0);
        id
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let r = self.find(self.parent[x]);
            self.parent[x] = r;
        }
        self.parent[x]
    }
    fn union(&mut self, a: usize, b: usize) -> usize {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb { return ra; }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
        ra
    }
}

/// Internal data for groups during construction/minimization.
#[derive(Clone)]
struct GroupData {
    tuple: ProductDFAStateTuple,
    final_weight: Option<Weight>,
    // aligned with sigma.labels order
    ex_dst: Vec<usize>,
    ex_w: Vec<Weight>,
    // default (OTHER)
    def_dst: usize,
    def_w: Weight,
    expanded: bool,
}

impl GroupData {
    fn new(tuple: ProductDFAStateTuple, num_labels: usize) -> Self {
        Self {
            tuple,
            final_weight: None,
            ex_dst: vec![0; num_labels],
            ex_w: vec![Weight::zeros(); num_labels],
            def_dst: 0,
            def_w: Weight::zeros(),
            expanded: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SigKey {
    final_hash: u64,
    def_w_hash: u64,
    def_dst_root: usize,
    // sorted by sigma.labels order
    ex: Vec<(i16, u64, usize)>,
}

/// Check exact (collision-free) signature equality for two groups given sigma and union-find.
fn signatures_equal(
    g1: usize,
    g2: usize,
    groups: &Vec<GroupData>,
    sigma: &Alphabet,
    uf: &mut UnionFind,
) -> bool {
    // final weights equal
    if groups[g1].final_weight != groups[g2].final_weight {
        return false;
    }
    // default weight equal
    if groups[g1].def_w != groups[g2].def_w {
        return false;
    }
    // default dest root equal
    if uf.find(groups[g1].def_dst) != uf.find(groups[g2].def_dst) {
        return false;
    }
    // exceptions equal: same length, same labels, same weights and root destinations
    if groups[g1].ex_w.len() != groups[g2].ex_w.len() {
        return false;
    }
    for (li, &lbl) in sigma.labels.iter().enumerate() {
        if groups[g1].ex_w[li] != groups[g2].ex_w[li] {
            return false;
        }
        if uf.find(groups[g1].ex_dst[li]) != uf.find(groups[g2].ex_dst[li]) {
            return false;
        }
    }
    true
}

fn compute_signature_key(
    gid: usize,
    groups: &Vec<GroupData>,
    sigma: &Alphabet,
    uf: &mut UnionFind,
) -> SigKey {
    let final_hash = match &groups[gid].final_weight {
        Some(w) => weight_fingerprint(w).wrapping_add(1),
        None => 0,
    };
    let def_w_hash = weight_fingerprint(&groups[gid].def_w);
    let def_dst_root = uf.find(groups[gid].def_dst);
    let mut ex = Vec::with_capacity(sigma.labels.len());
    for (li, &lbl) in sigma.labels.iter().enumerate() {
        let w_hash = weight_fingerprint(&groups[gid].ex_w[li]);
        let dst_root = uf.find(groups[gid].ex_dst[li]);
        ex.push((lbl, w_hash, dst_root));
    }
    SigKey { final_hash, def_w_hash, def_dst_root, ex }
}

fn get_or_create_group(
    t: ProductDFAStateTuple,
    tuple_to_gid: &mut HashMap<ProductDFAStateTuple, usize>,
    groups: &mut Vec<GroupData>,
    preds: &mut Vec<BTreeSet<usize>>,
    uf: &mut UnionFind,
    sigma: &Alphabet,
) -> usize {
    if let Some(&gid) = tuple_to_gid.get(&t) {
        return uf.find(gid);
    }
    let gid = uf.make_set();
    tuple_to_gid.insert(t.clone(), gid);
    groups.push(GroupData::new(t, sigma.labels.len()));
    preds.push(BTreeSet::new());
    gid
}

/// Optimal merging: compute minimal quotient of reachable tuples under (F, per-symbol W, successors).
fn minimize_tuples_to_states(
    start_tuple: ProductDFAStateTuple,
    comps: &[DetDFA],
    sigma: &Alphabet,
    comp_sinks: &[Option<usize>],
    atom_weights: &Vec<Weight>,
) -> MergedProduct {
    // Interning of tuples to group IDs
    let mut tuple_to_gid: HashMap<ProductDFAStateTuple, usize> = HashMap::new();
    // Groups under construction
    let mut groups: Vec<GroupData> = Vec::new();
    // Predecessor lists for recheck propagation (by group ID)
    let mut preds: Vec<BTreeSet<usize>> = Vec::new();
    // Union-find for merging equivalent groups
    let mut uf = UnionFind::default();
    // Signature -> canonical group mapping (use hash keys + equality recheck)
    let mut sig_to_gid: HashMap<SigKey, usize> = HashMap::new();

    // Worklists
    let mut expand_q: VecDeque<usize> = VecDeque::new();
    let mut recheck_q: VecDeque<usize> = VecDeque::new();

    // Seed
    let start_gid = get_or_create_group(start_tuple.clone(), &mut tuple_to_gid, &mut groups, &mut preds, &mut uf, sigma);
    expand_q.push_back(start_gid);

    // Expand groups with transitions and base signature data
    while let Some(gid0) = expand_q.pop_front() {
        let gid = uf.find(gid0);
        if groups[gid].expanded {
            continue;
        }

        let tuple = groups[gid].tuple.clone();
        // Compute final weight F(tuple)
        let mut w_final = Weight::zeros();
        for (i, pos) in tuple.iter().enumerate() {
            if let Some(s) = pos {
                if comps[i].finals[*s] {
                    w_final |= &atom_weights[i];
                }
            }
        }
        groups[gid].final_weight = if w_final.is_empty() { None } else { Some(w_final) };

        // Default (OTHER)
        let succ_def = successor_tuple(&tuple, sigma.other_index, comps, comp_sinks);
        let def_w = edge_weight_from_tuple(atom_weights, &succ_def);
        let def_gid = get_or_create_group(succ_def, &mut tuple_to_gid, &mut groups, &mut preds, &mut uf, sigma);
        groups[gid].def_w = def_w;
        groups[gid].def_dst = def_gid;
        preds[def_gid].insert(gid);

        // Exceptions (labels)
        for (li, _lbl) in sigma.labels.iter().enumerate() {
            let succ = successor_tuple(&tuple, li, comps, comp_sinks);
            let w = edge_weight_from_tuple(atom_weights, &succ);
            let to = get_or_create_group(succ, &mut tuple_to_gid, &mut groups, &mut preds, &mut uf, sigma);
            groups[gid].ex_w[li] = w;
            groups[gid].ex_dst[li] = to;
            preds[to].insert(gid);
        }

        groups[gid].expanded = true;

        // Try to merge by signature
        let key = compute_signature_key(gid, &groups, sigma, &mut uf);
        if let Some(&other) = sig_to_gid.get(&key) {
            let other_root = uf.find(other);
            let gid_root = uf.find(gid);
            if gid_root != other_root && signatures_equal(gid_root, other_root, &groups, sigma, &mut uf) {
                let r = uf.union(gid_root, other_root);
                // Merge predecessor sets and schedule rechecks
                let a = if r == gid_root { other_root } else { gid_root };
                // Merge preds[a] into preds[r]
                let mut moved = Vec::new();
                for p in preds[a].iter() {
                    moved.push(*p);
                }
                for p in moved {
                    preds[r].insert(p);
                }
                preds[a].clear();
                for &p in preds[r].iter() {
                    recheck_q.push_back(p);
                }
                // Update the signature map to point to the representative
                sig_to_gid.insert(key, r);
            } else {
                sig_to_gid.insert(key, gid_root);
            }
        } else {
            sig_to_gid.insert(key, uf.find(gid));
        }
    }

    // Propagate merges via predecessor rechecks until stable
    while let Some(g0) = recheck_q.pop_front() {
        let g = uf.find(g0);
        if !groups[g].expanded {
            continue;
        }
        let key = compute_signature_key(g, &groups, sigma, &mut uf);
        if let Some(&cand) = sig_to_gid.get(&key) {
            let cand_root = uf.find(cand);
            let g_root = uf.find(g);
            if g_root != cand_root && signatures_equal(g_root, cand_root, &groups, sigma, &mut uf) {
                let r = uf.union(g_root, cand_root);
                // Merge preds and schedule upstream rechecks
                let a = if r == g_root { cand_root } else { g_root };
                let mut moved = Vec::new();
                for p in preds[a].iter() {
                    moved.push(*p);
                }
                for p in moved {
                    preds[r].insert(p);
                }
                preds[a].clear();
                for &p in preds[r].iter() {
                    recheck_q.push_back(p);
                }
                sig_to_gid.insert(key, r);
            } else {
                sig_to_gid.insert(key, uf.find(g));
            }
        } else {
            // No candidate for this signature yet
            sig_to_gid.insert(key, uf.find(g));
        }
    }

    // Build quotient graph reachable from start root
    let start_root = uf.find(start_gid);
    // Map root-id -> representative group id (lowest id in the class)
    let mut root_repr: HashMap<usize, usize> = HashMap::new();
    for gid in 0..groups.len() {
        let r = uf.find(gid);
        root_repr.entry(r).and_modify(|e| if gid < *e { *e = gid }).or_insert(gid);
    }
    // Enumerate reachable roots via BFS
    let mut root_to_newid: HashMap<usize, usize> = HashMap::new();
    let mut new_states: Vec<ProductDFAState> = Vec::new();
    let mut q: VecDeque<usize> = VecDeque::new();
    root_to_newid.insert(start_root, 0);
    new_states.push(ProductDFAState {
        representative_tuple: groups[root_repr[&start_root]].tuple.clone(),
        all_tuples: BTreeSet::new(), // optional
        final_weight: groups[root_repr[&start_root]].final_weight.clone(),
        trans_exceptions: BTreeMap::new(),
        trans_default_gid: 0, // fill later
        trans_default_weight: Weight::zeros(),
    });
    q.push_back(start_root);

    let pb_merge = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new_spinner();
        p.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [Determinize: {elapsed_precise}] States: {pos}, Worklist: {msg}")
                .unwrap(),
        );
        Some(p)
    } else {
        None
    };

    while let Some(r) = q.pop_front() {
        let my_id = root_to_newid[&r];
        let repr_gid = root_repr[&r];
        let g = &groups[repr_gid];

        // Default
        let def_root = uf.find(g.def_dst);
        let def_newid = *root_to_newid.entry(def_root).or_insert_with(|| {
            let nid = new_states.len();
            new_states.push(ProductDFAState {
                representative_tuple: groups[root_repr[&def_root]].tuple.clone(),
                all_tuples: BTreeSet::new(),
                final_weight: groups[root_repr[&def_root]].final_weight.clone(),
                trans_exceptions: BTreeMap::new(),
                trans_default_gid: 0,
                trans_default_weight: Weight::zeros(),
            });
            q.push_back(def_root);
            if let Some(p) = &pb_merge {
                p.set_position(new_states.len() as u64);
                p.set_message(format!("{}", q.len()));
            }
            nid
        });
        new_states[my_id].trans_default_gid = def_newid;
        new_states[my_id].trans_default_weight = g.def_w.clone();

        // Exceptions
        for (li, &lbl) in sigma.labels.iter().enumerate() {
            let next_root = uf.find(g.ex_dst[li]);
            let next_newid = *root_to_newid.entry(next_root).or_insert_with(|| {
                let nid = new_states.len();
                new_states.push(ProductDFAState {
                    representative_tuple: groups[root_repr[&next_root]].tuple.clone(),
                    all_tuples: BTreeSet::new(),
                    final_weight: groups[root_repr[&next_root]].final_weight.clone(),
                    trans_exceptions: BTreeMap::new(),
                    trans_default_gid: 0,
                    trans_default_weight: Weight::zeros(),
                });
                q.push_back(next_root);
                if let Some(p) = &pb_merge {
                    p.set_position(new_states.len() as u64);
                    p.set_message(format!("{}", q.len()));
                }
                nid
            });
            new_states[my_id].trans_exceptions.insert(lbl, (next_newid, g.ex_w[li].clone()));
        }
    }

    let pb_merge = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new_spinner();
        p.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [Determinize: {elapsed_precise}] States: {pos}, Worklist: {msg}")
                .unwrap(),
        );
        Some(p)
    } else {
        None
    };

    if let Some(p) = pb_merge {
        p.finish_with_message(format!("Merged into {} states", new_states.len()));
    }

    // Start is 0 in new_states by construction
    MergedProduct { states: new_states, start: 0 }
}

/* ------------------------------
   Build DWA from merged product
   ------------------------------ */

fn edge_weight_from_tuple(atom_weights: &Vec<Weight>, tuple: &ProductDFAStateTuple) -> Weight {
    let mut w = Weight::zeros();
    for (i, pos) in tuple.iter().enumerate() {
        if pos.is_some() {
            w |= &atom_weights[i];
        }
    }
    w
}

fn build_dwa_from_merged(
    merged: &MergedProduct,
    atom_weights: &Vec<Weight>,
    sigma: &Alphabet,
) -> DWA {
    // Prepare DWA with the required number of states
    let mut dwa_states = DWAStates::default();
    for _ in 0..merged.states.len() {
        dwa_states.add_state();
    }
    let mut dwa = DWA { states: dwa_states, body: DWABody { start_state: merged.start } };

    // Final weights
    for sid in 0..merged.states.len() {
        if let Some(w) = &merged.states[sid].final_weight {
            let _ = dwa.set_final_weight(sid, w.clone());
        }
    }

    // Transitions
    for sid in 0..merged.states.len() {
        // Default (OTHER): already mapped to group id and weight
        let _ = dwa.set_default_transition(sid, merged.states[sid].trans_default_gid, merged.states[sid].trans_default_weight.clone());

        // Exceptions
        for (lbl, (to_id, w)) in &merged.states[sid].trans_exceptions {
            let _ = dwa.add_transition(sid, *lbl, *to_id, w.clone());
        }
    }

    dwa
}

/* ------------------------------
   End of merged determinization
   ------------------------------ */

