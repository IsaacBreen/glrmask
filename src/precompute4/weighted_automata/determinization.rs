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
                println!("    - Contains {} tuples.", state.num_tuples);
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
// ... lines 270-479 ...
}

/* ------------------------------
   Tuple enumeration and merging
   ------------------------------ */

type ProductDFAStateTuple = Vec<Option<usize>>;

/// Given a product tuple and a symbol, compute the successor tuple:
// ... lines 449-479 ...
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
    num_tuples: usize,
    final_weight: Option<Weight>,
    // Labeled exceptions: label -> destination tuple (unmapped)
    trans_exceptions: BTreeMap<i16, ProductDFAStateTuple>,
    // Default (OTHER) transition destination tuple (unmapped)
    trans_default: ProductDFAStateTuple,
    // If merged into another state, holds the target group id
    merged_into: Option<usize>,
}

#[derive(Clone, Debug)]
struct MergedProduct {
    states: Vec<ProductDFAState>,
    start: usize,
}

/// An index to accelerate finding a mergeable group for a given tuple.
struct GroupIndex {
// ... lines 508-526 ...
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

    /// Remove a group completely from the index (using its last known representative).
    fn remove_group(&mut self, gid: usize, rep: &ProductDFAStateTuple) {
        for i in 0..self.num_components {
            match rep[i] {
                Some(v) => {
                    if let Some(set) = self.by_value[i].get_mut(&v) {
                        set.remove(&gid);
                        if set.is_empty() {
                            self.by_value[i].remove(&v);
                        }
                    }
                }
                None => {
                    self.by_none[i].remove(&gid);
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

#[inline]
fn tuple_is_all_none(t: &ProductDFAStateTuple) -> bool {
    t.iter().all(|x| x.is_none())
}

/// Join two compatible tuples (assumes unification is possible).
#[inline]
fn join_tuples(mut a: ProductDFAStateTuple, b: &ProductDFAStateTuple) -> ProductDFAStateTuple {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        match (a[i], b[i]) {
            (Some(x), Some(y)) => {
                debug_assert_eq!(x, y, "join_tuples called on incompatible tuples");
            }
            (None, Some(y)) => {
                a[i] = Some(y);
            }
            _ => {}
        }
    }
    a
}

/// Collect all group IDs whose representatives are unifiable with `tuple`.
fn collect_unifiable_groups(
    tuple: &ProductDFAStateTuple,
    index: &GroupIndex,
    states: &[ProductDFAState],
) -> BTreeSet<usize> {
    // For each coordinate with Some(v), candidates are union(by_value[i][v], by_none[i]).
    // We intersect these per-coordinate candidate sets.
    let mut per_coord: Vec<BTreeSet<usize>> = Vec::new();
    for (i, vopt) in tuple.iter().enumerate() {
        if let Some(v) = vopt {
            let mut set_i: BTreeSet<usize> = BTreeSet::new();
            if let Some(by_v) = index.by_value[i].get(v) {
                set_i.extend(by_v.iter().copied());
            }
            set_i.extend(index.by_none[i].iter().copied());
            per_coord.push(set_i);
        }
    }
    // If tuple has no Some, there are no constraints; return empty to let caller handle sink case.
    if per_coord.is_empty() {
        return BTreeSet::new();
    }
    // Intersect all sets
    let mut it = per_coord.into_iter();
    let mut acc = it.next().unwrap();
    for s in it {
        acc = &acc & &s;
    }
    // Filter out merged ones and recheck unifiability (paranoia)
    acc.into_iter()
        .filter(|&gid| states[gid].merged_into.is_none())
        .filter(|&gid| unify_tuples(&states[gid].representative_tuple, tuple).is_some())
        .collect()
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
    let mut sink_gid: Option<usize> = None;

    // Helper to follow merged_into chain
    let mut get_root = |mut g: usize| -> usize {
        while let Some(to) = states[g].merged_into {
            g = to;
        }
        g
    };

    // Create starting group for the start_tuple
    {
        let mut start_state = ProductDFAState::default();
        start_state.representative_tuple = start_tuple.clone();
        start_state.num_tuples = 1;
        start_state.merged_into = None;
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
        // Skip if this group got merged into another
        if states[gid].merged_into.is_some() {
            continue;
        }
        let rep = states[gid].representative_tuple.clone();

        for sym in 0..sigma.size() {
            let succ_tuple = successor_tuple(&rep, sym, comps, comp_sinks);

            if tuple_to_group.contains_key(&succ_tuple) {
                continue;
            }

            // Special-case sink tuple (all None): keep a single sink group.
            if tuple_is_all_none(&succ_tuple) {
                let sgid = if let Some(id) = sink_gid {
                    id
                } else {
                    let new_gid = states.len();
                    let mut new_state = ProductDFAState::default();
                    new_state.representative_tuple = succ_tuple.clone();
                    new_state.num_tuples = 0; // will increment below
                    new_state.merged_into = None;
                    states.push(new_state);
                    group_index.add_group(new_gid, &states[new_gid].representative_tuple);
                    sink_gid = Some(new_gid);
                    if let Some(p) = &pb_merge {
                        p.set_position(states.len() as u64);
                        p.set_message(format!("{}", worklist.len()));
                    }
                    new_gid
                };
                states[sgid].num_tuples += 1;
                tuple_to_group.insert(succ_tuple, sgid);
                continue;
            }

            // Collect all unifiable groups and compute the join-closure
            let mut candidates = collect_unifiable_groups(&succ_tuple, &group_index, &states);
            let mut joined_rep = succ_tuple.clone();
            for g in candidates.iter().copied() {
                joined_rep = join_tuples(joined_rep, &states[g].representative_tuple);
            }
            // Expand closure: merging may enable additional groups to unify with the joined representative
            loop {
                let extra = collect_unifiable_groups(&joined_rep, &group_index, &states)
                    .into_iter()
                    .filter(|gid2| !candidates.contains(gid2))
                    .collect::<Vec<_>>();
                if extra.is_empty() {
                    break;
                }
                for g in extra.into_iter() {
                    joined_rep = join_tuples(joined_rep, &states[g].representative_tuple);
                    candidates.insert(g);
                }
            }

            if candidates.is_empty() {
                // Create a brand-new group for this successor
                let new_gid = states.len();
                let mut new_state = ProductDFAState::default();
                new_state.representative_tuple = succ_tuple.clone();
                new_state.num_tuples = 1;
                new_state.merged_into = None;
                states.push(new_state);
                tuple_to_group.insert(succ_tuple.clone(), new_gid);
                worklist.push_back(new_gid);
                in_worklist.insert(new_gid);
                group_index.add_group(new_gid, &states[new_gid].representative_tuple);

                if let Some(p) = &pb_merge {
                    p.set_position(states.len() as u64);
                    p.set_message(format!("{}", worklist.len()));
                }
            } else {
                // Merge all candidates into one target, with representative = joined_rep
                let mut cands_vec: Vec<usize> = candidates.into_iter().collect();
                cands_vec.sort_unstable();
                let target = cands_vec[0];
                let old_rep_t = states[target].representative_tuple.clone();
                // Update target's representative in the index
                if joined_rep != old_rep_t {
                    states[target].representative_tuple = joined_rep.clone();
                    group_index.update_rep(target, &old_rep_t, &joined_rep);
                }
                // Merge all other candidates into target
                for &g in &cands_vec[1..] {
                    if states[g].merged_into.is_some() {
                        continue;
                    }
                    let rep_g = states[g].representative_tuple.clone();
                    group_index.remove_group(g, &rep_g);
                    states[g].merged_into = Some(target);
                    states[target].num_tuples += states[g].num_tuples;
                }
                // Count this succ tuple
                states[target].num_tuples += 1;
                tuple_to_group.insert(succ_tuple.clone(), target);
                // Make sure target is scheduled for expansion with its new representative
                if in_worklist.insert(target) {
                    worklist.push_back(target);
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

    // Compress merged states: keep only live groups (those not merged into another)
    let mut old_to_new: Vec<usize> = vec![usize::MAX; states.len()];
    let mut new_states: Vec<ProductDFAState> = Vec::new();
    let mut new_id = 0usize;
    for old_id in 0..states.len() {
        if states[old_id].merged_into.is_none() {
            old_to_new[old_id] = new_id;
            let mut s = ProductDFAState::default();
            s.representative_tuple = states[old_id].representative_tuple.clone();
            s.num_tuples = states[old_id].num_tuples;
            new_states.push(s);
            new_id += 1;
        }
    }
    // The start is group 0 originally; redirect if it was merged
    let mut start_root = 0usize;
    while let Some(to) = states[start_root].merged_into {
        start_root = to;
    }
    let new_start = old_to_new[start_root];

    // Compute final weights and per-state transitions from representatives (on compressed states)
    let pb_attach = if PROGRESS_BAR_ENABLED {
        Some(
            ProgressBar::new(new_states.len() as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Attach weights/transitions)")
                    .unwrap(),
            ),
        )
    } else {
        None
    };

    for gid in 0..new_states.len() {
        let rep = new_states[gid].representative_tuple.clone();

        // Final weight: union of atom-weights where representative component is accepting
        let mut w_final = Weight::zeros();
        for (i, pos) in rep.iter().enumerate() {
            if let Some(s) = pos {
                if comps[i].finals[*s] {
                    w_final |= &atom_weights[i];
                }
            }
        }
        new_states[gid].final_weight = if w_final.is_empty() { None } else { Some(w_final) };

        // Transitions from representative
        // Default (OTHER)
        let def = successor_tuple(&rep, sigma.other_index, comps, comp_sinks);
        new_states[gid].trans_default = def;

        // Labeled exceptions
        let mut ex: BTreeMap<i16, ProductDFAStateTuple> = BTreeMap::new();
        for (li, &lbl) in sigma.labels.iter().enumerate() {
            let dst = successor_tuple(&rep, li, comps, comp_sinks);
            ex.insert(lbl, dst);
        }
        new_states[gid].trans_exceptions = ex;

        if let Some(p) = &pb_attach {
            p.inc(1);
        }
    }
    if let Some(p) = pb_attach {
        p.finish_with_message("Weights/transitions attached");
    }

    MergedProduct { states: new_states, start: new_start }
}

/* ------------------------------
   Build DWA from merged product
   ------------------------------ */

/// For a given tuple, find the ID of the merged state group it belongs to by checking for unifiability.
// ... lines 795-810 ...
