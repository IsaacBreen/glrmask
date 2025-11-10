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
use std::collections::hash_map::Entry;

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
        let merged = merge_tuples_to_states(
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
                let dest_tuple_def = &state.trans_default;
                if let Some(dest_gid) = find_group_for_tuple(&merged.states, dest_tuple_def) {
                    println!("      - Default -> State {} (via {})", dest_gid, format_tuple(dest_tuple_def));
                } else {
                    println!("      - Default -> UNMAPPED (via {})", format_tuple(dest_tuple_def));
                }

                for (lbl, dest_tuple_ex) in &state.trans_exceptions {
                    let char_repr = super::common::format_i16_char(*lbl);
                    if let Some(dest_gid) = find_group_for_tuple(&merged.states, dest_tuple_ex) {
                        println!("      - {} -> State {} (via {})", char_repr, dest_gid, format_tuple(dest_tuple_ex));
                    } else {
                        println!("      - {} -> UNMAPPED (via {})", char_repr, format_tuple(dest_tuple_ex));
                    }
                }
                println!("    - Contains {} tuples.", state.all_tuples.len());
            }
        }

        // 6) Convert merged product to a DWA
        let now_convert = Instant::now();
        // New: minimize merged product (near-optimal merging), then convert to DWA
        let minimized = minimize_merged_product(&merged, &atoms.atoms, &sigma);
        let dwa = build_dwa_from_minimized(&minimized, &sigma, &atoms.atoms);

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

/* ------------------------------
   Tuple enumeration and merging
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

#[derive(Clone, Debug, Default)]
struct ProductDFAState {
    representative_tuple: ProductDFAStateTuple,
    all_tuples: BTreeSet<ProductDFAStateTuple>,
    final_weight: Option<Weight>,
    // Labeled exceptions: label -> destination tuple (unmapped)
    trans_exceptions: BTreeMap<i16, ProductDFAStateTuple>,
    // Default (OTHER) transition destination tuple (unmapped)
    trans_default: ProductDFAStateTuple,
}

#[derive(Clone, Debug)]
struct MergedProduct {
    states: Vec<ProductDFAState>,
    start: usize,
}

/// An index to accelerate finding a mergeable group for a given tuple.
struct GroupIndex {
    /// For each component `i`, maps a state value `v` to a set of group IDs `g`
    /// where the representative tuple has `rep[i] = Some(v)`.
    by_value: Vec<HashMap<usize, BTreeSet<usize>>>,
    /// For each component `i`, a set of group IDs `g` where `rep[i] = None`.
    by_none: Vec<BTreeSet<usize>>,
    num_components: usize,
}

impl GroupIndex {
    fn new(num_components: usize) -> Self {
        Self {
            by_value: vec![HashMap::new(); num_components],
            by_none: vec![BTreeSet::new(); num_components],
            num_components,
        }
    }

    /// Register a new group with its representative tuple in the index.
    fn add_group(&mut self, gid: usize, rep: &ProductDFAStateTuple) {
        for i in 0..self.num_components {
            if let Some(v) = rep[i] {
                self.by_value[i].entry(v).or_default().insert(gid);
            } else {
                self.by_none[i].insert(gid);
            }
        }
    }

    /// Update the index when a group's representative becomes more specific.
    fn update_rep(&mut self, gid: usize, old_rep: &ProductDFAStateTuple, new_rep: &ProductDFAStateTuple) {
        for i in 0..self.num_components {
            if old_rep[i] != new_rep[i] {
                // This can only happen from None to Some(v)
                if let Some(v) = new_rep[i] {
                    self.by_none[i].remove(&gid);
                    self.by_value[i].entry(v).or_default().insert(gid);
                }
            }
        }
    }

    /// Find a group that can be unified with the given tuple.
    /// Uses a heuristic to check the most constrained component first.
    fn find_unifiable_group(&self, tuple: &ProductDFAStateTuple, states: &[ProductDFAState]) -> Option<usize> {
        let mut best_comp_idx = None;
        let mut min_candidates = usize::MAX;

        // Heuristic: find the component with the smallest candidate set to check.
        for (i, v_opt) in tuple.iter().enumerate() {
            if let Some(v) = v_opt {
                let c_val = self.by_value[i].get(v).map_or(0, |s| s.len());
                let c_none = self.by_none[i].len();
                let count = c_val + c_none;
                if count < min_candidates {
                    min_candidates = count;
                    best_comp_idx = Some(i);
                }
            }
        }

        if let Some(i) = best_comp_idx {
            let v = tuple[i].unwrap();
            let candidates_val = self.by_value[i].get(&v);
            let candidates_none = &self.by_none[i];

            // Iterate over candidates and perform the full unification check.
            let iter = candidates_val.into_iter().flatten().chain(candidates_none.iter());
            for &gid in iter {
                if unify_tuples(&states[gid].representative_tuple, tuple).is_some() {
                    return Some(gid);
                }
            }
        } else {
            // The tuple is all None, so it can unify with any group.
            // Return the first group if it exists.
            if !states.is_empty() {
                return Some(0);
            }
        }

        None
    }
}

/// Merge tuples greedily into product states and compute per-state transitions and final weights from representatives.
fn merge_tuples_to_states(
    start_tuple: ProductDFAStateTuple,
    comps: &[DetDFA],
    sigma: &Alphabet,
    comp_sinks: &[Option<usize>],
    atom_weights: &Vec<Weight>,
) -> MergedProduct {
    let mut states: Vec<ProductDFAState> = Vec::new();
    let mut tuple_to_group: HashMap<ProductDFAStateTuple, usize> = HashMap::new();
    let mut worklist = VecDeque::new(); // Worklist of group IDs
    let mut in_worklist = BTreeSet::new();

    let k = comps.len();
    let mut group_index = GroupIndex::new(k);

    // Create starting group for the start_tuple
    {
        let mut start_state = ProductDFAState::default();
        start_state.representative_tuple = start_tuple.clone();
        start_state.all_tuples.insert(start_tuple.clone());
        states.push(start_state); // gid = 0
        tuple_to_group.insert(start_tuple, 0);
        worklist.push_back(0);
        in_worklist.insert(0);
        group_index.add_group(0, &states[0].representative_tuple);
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

    if let Some(p) = &pb_merge {
        p.set_position(states.len() as u64);
        p.set_message(format!("{}", worklist.len()));
    }

    while let Some(gid) = worklist.pop_front() {
        in_worklist.remove(&gid);
        let rep = states[gid].representative_tuple.clone();

        for sym in 0..sigma.size() {
            let succ_tuple = successor_tuple(&rep, sym, comps, comp_sinks);

            if tuple_to_group.contains_key(&succ_tuple) {
                continue;
            }

            // Find a group for succ_tuple or create a new one
            if let Some(existing_gid) = group_index.find_unifiable_group(&succ_tuple, &states) {
                let old_rep = states[existing_gid].representative_tuple.clone();
                let new_rep = unify_tuples(&old_rep, &succ_tuple).unwrap();

                if new_rep != old_rep {
                    states[existing_gid].representative_tuple = new_rep.clone();
                    group_index.update_rep(existing_gid, &old_rep, &new_rep);

                    if in_worklist.insert(existing_gid) {
                        worklist.push_back(existing_gid);
                    }
                }
                tuple_to_group.insert(succ_tuple.clone(), existing_gid);
                states[existing_gid].all_tuples.insert(succ_tuple.clone());
            } else {
                // Create new group
                let new_gid = states.len();
                let mut new_state = ProductDFAState::default();
                new_state.representative_tuple = succ_tuple.clone();
                new_state.all_tuples.insert(succ_tuple.clone());
                states.push(new_state);
                tuple_to_group.insert(succ_tuple.clone(), new_gid);
                worklist.push_back(new_gid);
                in_worklist.insert(new_gid);
                group_index.add_group(new_gid, &states[new_gid].representative_tuple);

                if let Some(p) = &pb_merge {
                    p.set_position(states.len() as u64);
                    p.set_message(format!("{}", worklist.len()));
                }
            }
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
        p.finish_with_message(format!("Merged into {} states", states.len()));
    }

    // Compute final weights and per-state transitions from representatives
    let pb_attach = if PROGRESS_BAR_ENABLED {
        Some(
            ProgressBar::new(states.len() as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Attach weights/transitions)")
                    .unwrap(),
            ),
        )
    } else {
        None
    };

    for gid in 0..states.len() {
        let rep = states[gid].representative_tuple.clone();

        // Final weight: union of atom-weights where representative component is accepting
        let mut w_final = Weight::zeros();
        for (i, pos) in rep.iter().enumerate() {
            if let Some(s) = pos {
                if comps[i].finals[*s] {
                    w_final |= &atom_weights[i];
                }
            }
        }
        states[gid].final_weight = if w_final.is_empty() { None } else { Some(w_final) };

        // Transitions from representative
        // Default (OTHER)
        let def = successor_tuple(&rep, sigma.other_index, comps, comp_sinks);
        states[gid].trans_default = def;

        // Labeled exceptions
        let mut ex: BTreeMap<i16, ProductDFAStateTuple> = BTreeMap::new();
        for (li, &lbl) in sigma.labels.iter().enumerate() {
            let dst = successor_tuple(&rep, li, comps, comp_sinks);
            ex.insert(lbl, dst);
        }
        states[gid].trans_exceptions = ex;

        if let Some(p) = &pb_attach {
            p.inc(1);
        }
    }
    if let Some(p) = pb_attach {
        p.finish_with_message("Weights/transitions attached");
    }

    MergedProduct { states, start: 0 }
}

/* ------------------------------
   Build DWA from merged product
   ------------------------------ */

/// For a given tuple, find the ID of the merged state group it belongs to by checking for unifiability.
fn find_group_for_tuple(
    merged_states: &[ProductDFAState],
    tuple: &ProductDFAStateTuple,
) -> Option<usize> {
    // This is a linear scan. Can be optimized if it's a bottleneck.
    for (gid, state) in merged_states.iter().enumerate() {
        if unify_tuples(&state.representative_tuple, tuple).is_some() {
            return Some(gid);
        }
    }
    None
}

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
        // Default (OTHER)
        let def_t = &merged.states[sid].trans_default;
        let def_w = edge_weight_from_tuple(atom_weights, def_t);
        if let Some(to_id) = find_group_for_tuple(&merged.states, def_t) {
            let _ = dwa.set_default_transition(sid, to_id, def_w);
        } else {
            assert!(def_t.iter().all(|x| x.is_none()), "Default transition tuple must be all None if unmapped. Found unmapped tuple {:?} for state {}", def_t, sid);
        }

        // Exceptions
        for (lbl, dst_t) in &merged.states[sid].trans_exceptions {
            let w = edge_weight_from_tuple(atom_weights, dst_t);
            if let Some(to_id) = find_group_for_tuple(&merged.states, dst_t) {
                let _ = dwa.add_transition(sid, *lbl, to_id, w);
            } else {
                assert!(dst_t.iter().all(|x| x.is_none()), "Exception transition tuple must be all None if unmapped. Found unmapped tuple {:?} for state {} on label {}", dst_t, sid, format_i16_char(*lbl));
            }
        }
    }

    dwa
}

/* ------------------------------
   Post-merge minimization
   ------------------------------ */

/// A compact bit-mask over components (atoms) used to characterize:
/// - which components survive on a symbol (edge mask),
/// - which components accept in a state (final mask).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct Mask {
    words: Vec<u64>, // little chunks, words.len() == ceil(k/64)
    nbits: usize,    // k
}
impl Mask {
    fn zeros(k: usize) -> Self {
        let nwords = (k + 63) / 64;
        Self { words: vec![0u64; nwords], nbits: k }
    }
    fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }
    #[inline]
    fn set_bit(&mut self, i: usize) {
        let w = i >> 6;
        let b = i & 63;
        self.words[w] |= 1u64 << b;
    }
}

fn mask_from_tuple(tuple: &ProductDFAStateTuple, k: usize) -> Mask {
    let mut m = Mask::zeros(k);
    for (i, pos) in tuple.iter().enumerate().take(k) {
        if pos.is_some() {
            m.set_bit(i);
        }
    }
    m
}

fn mask_from_weight(w: &Weight, atoms: &Vec<Weight>) -> Mask {
    let k = atoms.len();
    let mut m = Mask::zeros(k);
    for (i, atom) in atoms.iter().enumerate() {
        if !(&*w & atom).is_empty() {
            m.set_bit(i);
        }
    }
    m
}

fn weight_from_mask(mask: &Mask, atoms: &Vec<Weight>) -> Weight {
    let mut out = Weight::zeros();
    let k = atoms.len();
    for i in 0..k {
        let w = mask.words[i >> 6];
        let bit = (w >> (i & 63)) & 1;
        if bit == 1 {
            out |= &atoms[i];
        }
    }
    out
}

/// A minimized deterministic machine over Sigma' (labels + OTHER), with:
/// - start state index
/// - per-state final mask (weight = union of atoms for set bits),
/// - per-symbol transitions to a single state with an associated edge mask.
#[derive(Clone, Debug)]
struct MinMergedState {
    /// Keep one representative tuple (not used for transitions/weights, only for debugging)
    representative_tuple: ProductDFAStateTuple,
    /// Which atom-components accept in this state (determines final weight)
    final_mask: Mask,
    /// For each symbol in Sigma': (dest_state, mask). If dest is the implicit sink, store (usize::MAX, zeros-mask).
    trans: Vec<(usize, Mask)>,
}

#[derive(Clone, Debug)]
struct MinMergedProduct {
    start: usize,
    states: Vec<MinMergedState>,
}

/// Build a minimized merged product by partition refinement on the greedy MergedProduct.
/// States are equivalent iff:
/// - their final masks are equal, and
/// - for every symbol sym, both the presence mask and the destination block are equal.
fn minimize_merged_product(
    merged: &MergedProduct,
    atom_weights: &Vec<Weight>,
    sigma: &Alphabet,
) -> MinMergedProduct {
    let n = merged.states.len();
    if n == 0 {
        return MinMergedProduct { start: 0, states: vec![] };
    }
    let k = atom_weights.len();
    let a = sigma.size();

    // Precompute, for each state and symbol:
    // - dest tuple
    // - presence mask (determines edge weight)
    // - dest state (in old merged indexing) if any, else None (implicit sink)
    let mut edge_masks: Vec<Vec<Mask>> = vec![vec![Mask::zeros(k); a]; n];
    let mut edge_dest_gid: Vec<Vec<Option<usize>>> = vec![vec![None; a]; n];
    let mut final_masks: Vec<Mask> = vec![Mask::zeros(k); n];

    for s in 0..n {
        // Final mask from state's final_weight
        if let Some(ref fw) = merged.states[s].final_weight {
            final_masks[s] = mask_from_weight(fw, atom_weights);
        } else {
            final_masks[s] = Mask::zeros(k);
        }

        for sym in 0..a {
            let dest_tuple = if sym == sigma.other_index {
                &merged.states[s].trans_default
            } else {
                // Map symbol index to label and fetch exception tuple
                let lbl = sigma.labels[sym];
                merged.states[s].trans_exceptions.get(&lbl).expect("missing exception tuple for label")
            };
            let mask = mask_from_tuple(dest_tuple, k);
            edge_masks[s][sym] = mask;
            let dest_gid = find_group_for_tuple(&merged.states, dest_tuple);
            edge_dest_gid[s][sym] = dest_gid;
        }
    }

    // Partition refinement
    // Use usize::MAX as a "SINK" target id in signatures.
    let sink_id: usize = usize::MAX;

    // Initial partition: by final_mask only
    let mut block_of: Vec<usize> = vec![0; n];
    {
        use std::collections::HashMap;
        let mut key_to_block: HashMap<Mask, usize> = HashMap::new();
        let mut next_block = 0usize;
        for s in 0..n {
            let key = final_masks[s].clone();
            let bid = match key_to_block.entry(key) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => {
                    let id = next_block;
                    next_block += 1;
                    e.insert(id);
                    id
                }
            };
            block_of[s] = bid;
        }
    }

    // Iteratively refine until stable
    loop {
        use std::collections::HashMap;

        #[derive(Clone, Debug, Eq, PartialEq, Hash)]
        struct SigKey {
            final_mask: Mask,
            // For each symbol: (dest_block, mask)
            dest: Vec<(usize, Mask)>,
        }

        let mut new_block_of: Vec<usize> = vec![usize::MAX; n];
        let mut key_to_block: HashMap<SigKey, usize> = HashMap::new();
        let mut next_block = 0usize;

        for s in 0..n {
            let mut dest_vec: Vec<(usize, Mask)> = Vec::with_capacity(a);
            for sym in 0..a {
                let d = edge_dest_gid[s][sym].map(|g| block_of[g]).unwrap_or(sink_id);
                dest_vec.push((d, edge_masks[s][sym].clone()));
            }
            let key = SigKey {
                final_mask: final_masks[s].clone(),
                dest: dest_vec,
            };
            let bid = match key_to_block.entry(key) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => {
                    let id = next_block;
                    next_block += 1;
                    e.insert(id);
                    id
                }
            };
            new_block_of[s] = bid;
        }

        if new_block_of == block_of {
            // Stable
            break;
        }
        block_of = new_block_of;
    }

    // Build minimized machine
    let num_blocks = 1 + block_of.iter().copied().max().unwrap_or(0);
    let mut reps: Vec<usize> = vec![usize::MAX; num_blocks];
    for s in 0..n {
        let b = block_of[s];
        if reps[b] == usize::MAX {
            reps[b] = s;
        }
    }

    // For each block, produce a canonical state (use the first representative).
    let mut out_states: Vec<MinMergedState> = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let s = reps[b];
        // Build per-symbol transitions as (dest block, mask) using the representative only.
        let mut trans: Vec<(usize, Mask)> = Vec::with_capacity(a);
        for sym in 0..a {
            let dest_block = edge_dest_gid[s][sym].map(|g| block_of[g]).unwrap_or(sink_id);
            trans.push((dest_block, edge_masks[s][sym].clone()));
        }
        out_states.push(MinMergedState {
            representative_tuple: merged.states[s].representative_tuple.clone(),
            final_mask: final_masks[s].clone(),
            trans,
        });
    }

    // Start block = block_of(old_start)
    let start_block = block_of[merged.start];
    MinMergedProduct { start: start_block, states: out_states }
}

/// Build a DWA out of the minimized merged product.
/// - For each state, the final weight is the union of atom-weights for set bits in final_mask.
/// - For each symbol:
///     - If sym == OTHER and transition not to sink: set default transition to (dest, weight(mask)).
///     - If sym is a label and transition not to sink: add exception transition with its weight.
fn build_dwa_from_minimized(
    min: &MinMergedProduct,
    sigma: &Alphabet,
    atom_weights: &Vec<Weight>,
) -> DWA {
    let n = min.states.len();
    let mut dwa_states = DWAStates::default();
    for _ in 0..n {
        dwa_states.add_state();
    }
    let mut dwa = DWA { states: dwa_states, body: DWABody { start_state: min.start } };

    // Final weights
    for sid in 0..n {
        let w = weight_from_mask(&min.states[sid].final_mask, atom_weights);
        if !w.is_empty() {
            let _ = dwa.set_final_weight(sid, w);
        }
    }

    // Transitions
    let a = sigma.size();
    for sid in 0..n {
        for sym in 0..a {
            let (dst, mask) = &min.states[sid].trans[sym];
            if *dst == usize::MAX || mask.is_empty() {
                continue; // implicit sink or zero-weight: no transition in DWA
            }
            let w = weight_from_mask(mask, atom_weights);
            if sym == sigma.other_index {
                let _ = dwa.set_default_transition(sid, *dst, w);
            } else {
                let lbl = sigma.labels[sym];
                let _ = dwa.add_transition(sid, lbl, *dst, w);
            }
        }
    }

    dwa
}

/* ------------------------------
   End of merged determinization
   ------------------------------ */
