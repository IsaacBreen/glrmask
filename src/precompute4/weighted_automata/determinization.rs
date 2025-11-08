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

/*
Shared-minimization determinization (NO synchronous product):

High-level outline:
1) Compute weight atoms: a disjoint partition of the weight space (Vec<RangeInclusive<usize>>),
   with a cached Weight for each atom. Let k = number of atoms.

2) Build the alphabet Sigma' = (all explicit labels used anywhere in NWA) ∪ {OTHER}, filtered
   by future-acceptance masks as before. Its size is base_a = labels.len() + 1 (OTHER).

3) For each atom i (0..k-1):
   - Build an NFA restricted to that atom (keep only edges whose weights intersect the atom).
   - Determinize to a complete DFA over Sigma'.
   - Minimize the DFA.
   - Optionally record a sink state index in this DFA (complete self-loop, non-accepting).

4) Combine all k minimized DFAs into a single "combined" complete DFA over an extended alphabet:
   - Extended alphabet size A_ext = base_a + k, where the last k symbols are "ENTRY_i" (one per atom).
   - Add a fresh super_start (id=0) and a global sink (id=1).
   - From super_start, on symbol ENTRY_i, jump to the start state of DFA_i (offset in the combined graph).
   - All base symbols from super_start go to the sink.
   - For every state of DFA_i, base-symbol transitions copy over; ENTRY_* symbols lead to the global sink.
   - All sink transitions loop to itself.

   Keep an "origin" vector mapping each combined-old-state to (component_index, local_state_index)
   for states originating from some DFA_i; super_start and sink have None.

5) Minimize the combined DFA once (Hopcroft), getting a partition mapping old->new states.

6) Compute per-minimized-state weights:
   - For each old combined state that originated from component i and local state s:
       let new = old_to_new[old].
       If s is not the sink of component i, add atom_weight[i] to w_live[new].
       If s is accepting in component i, add atom_weight[i] to w_final[new].

7) Choose a DWA start state: image of (most common) component-start among i=0..k-1.
   - Ideally all component starts map to the same minimized state; if not, choose the most frequent one
     and continue. We log a debug message when starts disagree.

8) Build the final DWA:
   - Number of states = number of states in the minimized combined DFA.
   - For each state s, set its final_weight = w_final[s] (if non-empty).
   - For each state s, set a default transition on OTHER to minimized_dfa.trans[s][other_index],
     with trans weight = w_live[s]. Then, for each explicit label l, let dst = trans[s][l_index];
     if dst != default_dst, add exception transition on label l to dst with the same w_live[s].
   - Set DWA.start_state = chosen minimized start.

This construction shares as much structure as possible across per-atom components, avoids the
explicit synchronous product blow-up, and assigns weights per edge (and finals) by aggregating
over the atoms for which that edge/state is relevant.

Note: This approach relies on the fact that DFA minimization merges only states whose full
transition behavior (on the base alphabet) is identical, so using a single target per label
per state is always consistent.
*/

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now_total = Instant::now();

        // Work on a clone of the NWA.
        let mut nwa = self.clone();
        debug_log(3, || format!("Starting determinization for NWA with {} states", nwa.states.len()));

        // Compute future-acceptance masks (used to filter the alphabet).
        let fut = nwa.compute_future_weights();

        // 1) Weight atoms (disjoint partition)
        let now_atoms = Instant::now();
        let atoms = WeightPartition::from_nwa(&nwa);
        debug_log(4, || {
            format!("Built weight partition with {} atoms in {:?}", atoms.intervals.len(), now_atoms.elapsed())
        });

        // Edge case: no atoms => no weight can be produced => return trivial DWA.
        if atoms.intervals.is_empty() {
            let dwa = DWA::new();
            debug_log(3, || format!("No atoms -> returning trivial DWA in {:?}", now_total.elapsed()));
            return dwa;
        }

        // 2) Base alphabet Sigma' over explicit labels plus OTHER
        let now_sigma = Instant::now();
        let sigma = Alphabet::from_nwa_with_future(&nwa, &fut);
        let base_a = sigma.size();
        let other_index = sigma.other_index;
        debug_log(4, || {
            format!("Built alphabet with {} labels (base size {}) in {:?}", sigma.labels.len(), base_a, now_sigma.elapsed())
        });

        // 3) For each atom, build and minimize DFA_i
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

        // 4) Build combined pre-DFA (with ENTRY symbols)
        let now_comb = Instant::now();
        let combined = CombinedPreDFA::build(&comp_dfas, base_a, other_index);
        debug_log(4, || {
            format!(
                "Combined pre-DFA: states={}, alphabet_size={}, (base={}, entries={}), time={:?}",
                combined.dfa.n_states,
                combined.alphabet_size(),
                base_a,
                combined.num_entries(),
                now_comb.elapsed()
            )
        });

        // 5) Minimize combined DFA once, and capture old->new mapping.
        let now_min = Instant::now();
        let (min_dfa, old_to_new) = combined.dfa.clone().minimized_with_mapping(combined.alphabet_size());
        crate::debug!(4,
            format!(
                "Combined DFA minimized: states {} -> {} in {:?}",
                combined.dfa.n_states,
                min_dfa.n_states,
                now_min.elapsed()
            )
        );

        // 6) Compute per-minimized-state weights (live and final) by aggregating origins.
        let mut w_live = vec![Weight::zeros(); min_dfa.n_states];
        let mut w_final = vec![Weight::zeros(); min_dfa.n_states];

        for old in 0..combined.dfa.n_states {
            let new = old_to_new[old];
            if new == usize::MAX {
                continue;
            }
            if let Some((ci, si)) = combined.origin[old] {
                let atom_w = &atoms.atoms[ci];

                // Live if not sink in component ci
                let is_sink = match comp_sinks[ci] {
                    Some(sk) => si == sk,
                    None => false,
                };
                if !is_sink {
                    w_live[new] |= atom_w;
                }

                // Final if accepting in component ci
                if comp_dfas[ci].finals[si] {
                    w_final[new] |= atom_w;
                }
            }
        }

        // 7) Choose a start state: ideally all per-atom starts map to the same minimized state.
        // Count per minimized state the number of component starts that mapped to it.
        let mut start_counts: HashMap<usize, usize> = HashMap::new();
        for (ci, dfa) in comp_dfas.iter().enumerate() {
            let old_start = combined.offsets[ci] + dfa.start;
            let new = old_to_new[old_start];
            if new != usize::MAX {
                *start_counts.entry(new).or_insert(0) += 1;
            }
        }
        let (chosen_start, total_starts, distinct_start_targets) = {
            let mut best = (0usize, 0usize);
            for (sid, cnt) in &start_counts {
                if *cnt > best.1 {
                    best = (*sid, *cnt);
                }
            }
            (best.0, comp_dfas.len(), start_counts.len())
        };
        if distinct_start_targets > 1 {
            crate::debug!(
                2,
                "Shared-minimization: component start states map to {} distinct minimized states; \
                 choosing the most frequent one (sid={} with {}/{} starts).",
                distinct_start_targets,
                chosen_start,
                start_counts.get(&chosen_start).copied().unwrap_or(0),
                total_starts
            );
        } else {
            crate::debug!(4, "All component starts map to common minimized state {}", chosen_start);
        }

        // 8) Build the final DWA: states = min_dfa.n_states; transitions copy min_dfa (base alphabet);
        //    weights = w_live for edges and w_final for finals.
        let mut dwa_states = DWAStates::default();
        for _ in 0..min_dfa.n_states {
            dwa_states.add_state();
        }
        let mut dwa = DWA {
            states: dwa_states,
            body: DWABody { start_state: chosen_start },
        };

        // Assign final weights
        for sid in 0..min_dfa.n_states {
            if !w_final[sid].is_empty() {
                let _ = dwa.set_final_weight(sid, w_final[sid].clone());
            }
        }

        // Assign transitions & edge-weights using base alphabet only.
        // Default transition = OTHER column; exceptions for labels that differ from default target.
        for sid in 0..min_dfa.n_states {
            let edge_weight = w_live[sid].clone();

            // Default
            let def_dst = min_dfa.trans[sid][other_index];
            let _ = dwa.set_default_transition(sid, def_dst, edge_weight.clone());

            // Exceptions
            for (li, &lbl) in sigma.labels.iter().enumerate() {
                let dst = min_dfa.trans[sid][li];
                if dst != def_dst {
                    let _ = dwa.add_transition(sid, lbl, dst, edge_weight.clone());
                }
            }
        }

        debug_log(3, || {
            format!("NWA::determinize_to_dwa (shared-minimization) total time: {:?}", now_total.elapsed())
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
   Per-atom NFA and DFA (unchanged core)
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
                        // OTHER: take all default transitions.
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
        self.minimize_inner(sigma.size());
    }

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

    /// Minimize and return mapping from old to new indices.
    fn minimized_with_mapping(mut self, alphabet_size: usize) -> (DetDFA, Vec<usize>) {
        // 1) Remove unreachable from start
        let reachable = {
            let mut visited = vec![false; self.n_states];
            let mut q = VecDeque::new();
            visited[self.start] = true;
            q.push_back(self.start);
            while let Some(u) = q.pop_front() {
                for sym in 0..alphabet_size {
                    let v = self.trans[u][sym];
                    if !visited[v] {
                        visited[v] = true;
                        q.push_back(v);
                    }
                }
            }
            visited
        };
        let mut old_to_compact = vec![usize::MAX; self.n_states];
        let mut n_compact = 0usize;
        for s in 0..self.n_states {
            if reachable[s] {
                old_to_compact[s] = n_compact;
                n_compact += 1;
            }
        }
        if n_compact == 0 {
            // Create trivial dead DFA
            let out = DetDFA {
                n_states: 1,
                start: 0,
                finals: vec![false],
                trans: vec![vec![0; alphabet_size]],
            };
            let mapping = vec![0usize; self.n_states];
            return (out, mapping);
        }

        let mut finals_c = vec![false; n_compact];
        let mut trans_c = vec![vec![0usize; alphabet_size]; n_compact];
        for old in 0..self.n_states {
            let nc = old_to_compact[old];
            if nc == usize::MAX {
                continue;
            }
            finals_c[nc] = self.finals[old];
            for sym in 0..alphabet_size {
                trans_c[nc][sym] = old_to_compact[self.trans[old][sym]];
            }
        }
        let start_c = old_to_compact[self.start];

        // 2) Hopcroft on compact DFA
        let (part_id, reps, num_parts) = hopcroft_minimize(&finals_c, &trans_c, alphabet_size);

        // 3) Build quotient DFA
        let mut finals2 = vec![false; num_parts];
        for pid in 0..num_parts {
            finals2[pid] = finals_c[reps[pid]];
        }
        let mut trans2 = vec![vec![0usize; alphabet_size]; num_parts];
        for pid in 0..num_parts {
            let s = reps[pid];
            for sym in 0..alphabet_size {
                let v = trans_c[s][sym];
                trans2[pid][sym] = part_id[v];
            }
        }
        let start2 = part_id[start_c];
        let minimized = DetDFA {
            n_states: num_parts,
            start: start2,
            finals: finals2,
            trans: trans2,
        };

        // 4) Build mapping old->new
        let mut old_to_new = vec![usize::MAX; self.n_states];
        for old in 0..self.n_states {
            let c = old_to_compact[old];
            if c != usize::MAX {
                old_to_new[old] = part_id[c];
            }
        }

        (minimized, old_to_new)
    }

    fn minimize_inner(&mut self, alphabet_size: usize) {
        // Remove states unreachable from start first
        let reachable = {
            let mut visited = vec![false; self.n_states];
            let mut q = VecDeque::new();
            visited[self.start] = true;
            q.push_back(self.start);
            while let Some(u) = q.pop_front() {
                for sym in 0..alphabet_size {
                    let v = self.trans[u][sym];
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
            self.trans = vec![vec![0; alphabet_size]];
            return;
        }

        let mut finals = vec![false; new_states];
        let mut trans = vec![vec![0usize; alphabet_size]; new_states];

        for i in 0..self.n_states {
            if !reachable[i] {
                continue;
            }
            let ni = map[i];
            finals[ni] = self.finals[i];
            for a in 0..alphabet_size {
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
        let a = alphabet_size;

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
}

/// Hopcroft minimization for complete DFAs (returns partition id per state, representatives, and count)
fn hopcroft_minimize(
    finals: &Vec<bool>,
    trans: &Vec<Vec<usize>>,
    alphabet_size: usize,
) -> (Vec<usize>, Vec<usize>, usize) {
    let n = finals.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }
    if n == 1 {
        return (vec![0], vec![0], 1);
    }

    // Initial partition: accepting vs non-accepting
    let mut part_id = vec![0usize; n];
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let (accepting_block, non_accepting_block): (Vec<_>, Vec<_>) = (0..n).partition(|&s| finals[s]);

    if accepting_block.is_empty() || non_accepting_block.is_empty() {
        // All accepting or all non-accepting -> nothing to split
        let mut reps = vec![0usize];
        reps[0] = 0;
        for i in 0..n {
            part_id[i] = 0;
        }
        return (part_id, reps, 1);
    }

    blocks.push(accepting_block);
    blocks.push(non_accepting_block);
    for (pid, block) in blocks.iter().enumerate() {
        for &s in block {
            part_id[s] = pid;
        }
    }

    // Build inverse transitions for each symbol
    let mut inv: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); n]; alphabet_size];
    for s in 0..n {
        for sym in 0..alphabet_size {
            let v = trans[s][sym];
            inv[sym][v].push(s);
        }
    }

    // Worklist
    let mut worklist: BTreeSet<(usize, usize)> = BTreeSet::new();
    let smaller_initial_set = if blocks[0].len() <= blocks[1].len() { 0 } else { 1 };
    for sym in 0..alphabet_size {
        worklist.insert((smaller_initial_set, sym));
    }

    while let Some(&(b, sym)) = worklist.iter().next() {
        worklist.remove(&(b, sym));

        // Compute preimage of block b under symbol sym
        let mut pre: Vec<usize> = Vec::new();
        for &v in &blocks[b] {
            pre.extend_from_slice(&inv[sym][v]);
        }
        if pre.is_empty() {
            continue;
        }
        pre.sort_unstable();
        pre.dedup();

        // For each block, split by intersection with pre
        let mut affected: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
        for &s in &pre {
            let pid = part_id[s];
            affected.entry(pid).or_default().0.push(s);
        }
        for (pid, (ref mut in_pre, ref mut not_in_pre)) in affected.iter_mut() {
            // Fill not_in_pre
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

        // Apply replacements (block splits)
        for (pid, mut in_pre, mut not_in_pre) in to_replace {
            in_pre.sort_unstable();
            not_in_pre.sort_unstable();

            let pid2 = blocks.len();
            blocks.push(not_in_pre);
            blocks[pid] = in_pre;

            // Update part_id map for all states in the newly created blocks
            for &s in &blocks[pid] {
                part_id[s] = pid;
            }
            for &s in &blocks[pid2] {
                part_id[s] = pid2;
            }

            // Update worklist according to Hopcroft's algorithm
            for sym2 in 0..alphabet_size {
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

    let num_parts = blocks.len();
    let mut reps: Vec<usize> = vec![0; num_parts];
    for (pid, block) in blocks.iter().enumerate() {
        reps[pid] = block[0];
    }

    (part_id, reps, num_parts)
}

/* ------------------------------
   Combined pre-DFA builder (ENTRY trick)
   ------------------------------ */

/// Combined pre-DFA built from component DFAs and k ENTRY symbols.
///
/// Alphabet size = base_a + k (k entry symbols).
/// State 0 = super_start; State 1 = global sink.
/// For each component i:
///   - offset[i] is starting index of DFA_i's states inside the combined graph
///   - For state s in DFA_i:
///       - base transitions copied from DFA_i
///       - all ENTRY_* transitions go to sink
/// Super_start:
///   - on base symbols -> sink
///   - on ENTRY_i -> offset[i] + DFA_i.start
///
/// Also stores 'origin' per old-state: Some((comp_i, local_state)) for component states; None for super_start and sink.
struct CombinedPreDFA {
    dfa: DetDFA,
    origin: Vec<Option<(usize, usize)>>, // length = dfa.n_states
    offsets: Vec<usize>,                  // per component i, offset
    base_a: usize,
    other_index: usize,
    k_entries: usize,
    super_start: usize,
    sink: usize,
}
impl CombinedPreDFA {
    fn build(comps: &[DetDFA], base_a: usize, other_index: usize) -> Self {
        let k = comps.len();
        let a_ext = base_a + k;

        // Reserve 0: super_start, 1: sink
        let super_start = 0usize;
        let sink = 1usize;

        let mut offsets: Vec<usize> = Vec::with_capacity(k);
        let mut next = 2usize;
        for dfa in comps {
            offsets.push(next);
            next += dfa.n_states;
        }
        let n_total = next;

        // Initialize
        let mut trans = vec![vec![sink; a_ext]; n_total];
        let mut finals = vec![false; n_total];
        let mut origin = vec![None; n_total];

        // Sink loops
        for sym in 0..a_ext {
            trans[sink][sym] = sink;
        }
        finals[sink] = false;

        // Super_start: base -> sink (already), ENTRY_i -> component i start
        for i in 0..k {
            let entry_col = base_a + i;
            let target = offsets[i] + comps[i].start;
            trans[super_start][entry_col] = target;
        }
        finals[super_start] = false;

        // Components
        for (ci, dfa) in comps.iter().enumerate() {
            let off = offsets[ci];
            for s in 0..dfa.n_states {
                let csid = off + s;
                origin[csid] = Some((ci, s));
                finals[csid] = dfa.finals[s];
                // Base transitions
                for sym in 0..base_a {
                    let v = dfa.trans[s][sym];
                    trans[csid][sym] = off + v;
                }
                // ENTRY transitions -> sink (already initialized)
                for ent in base_a..a_ext {
                    trans[csid][ent] = sink;
                }
            }
        }

        CombinedPreDFA {
            dfa: DetDFA {
                n_states: n_total,
                start: super_start,
                finals,
                trans,
            },
            origin,
            offsets,
            base_a,
            other_index,
            k_entries: k,
            super_start,
            sink,
        }
    }

    #[inline]
    fn alphabet_size(&self) -> usize {
        self.base_a + self.k_entries
    }
    #[inline]
    fn num_entries(&self) -> usize {
        self.k_entries
    }
}

/* ------------------------------
   End of combined-shared determinization
   ------------------------------ */
