use std::time::Instant;
use crate::r#macro::is_debug_level_enabled;
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::NWA;
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

type DWAStateID = usize;

/// Represents a determinized state, which corresponds to a set of NWA configurations.
/// This is the "powerstate" in subset construction.
#[derive(Clone, Debug, Eq, PartialEq)]
struct DWAStateSignature {
    /// The structural key for merging. A sorted, unique vector of NWA state IDs.
    key: Vec<NWAStateID>,
    /// The full powerstate: a list of (NWA state, accumulated weight) pairs.
    /// This is kept canonical (sorted by ID, merged weights) for stability.
    powerstate: Vec<(NWAStateID, Weight)>,
}

impl DWAStateSignature {
    /// Creates a canonical signature from a raw vector of powerstate entries.
    /// Canonicalization involves sorting, merging weights for identical states,
    /// and removing entries with empty weights.
    fn from_powerstate(mut powerstate: Vec<(NWAStateID, Weight)>) -> Self {
        if powerstate.is_empty() {
            return Self { key: vec![], powerstate: vec![] };
        }

        // Sort by NWA state ID to enable efficient merging.
        powerstate.sort_by_key(|(id, _)| *id);

        let mut merged: Vec<(NWAStateID, Weight)> = Vec::with_capacity(powerstate.len());
        for (id, weight) in powerstate {
            if weight.is_empty() {
                continue;
            }
            if let Some(last) = merged.last_mut() {
                if last.0 == id {
                    last.1 |= &weight; // Merge weights for the same NWA state.
                    continue;
                }
            }
            merged.push((id, weight));
        }

        let key = merged.iter().map(|(id, _)| *id).collect();
        Self { key, powerstate: merged }
    }
}

/// Computes the epsilon-closure from a set of source NWA configurations.
fn eps_closure_multi(
    sources: &[(NWAStateID, Weight)],
    nwa: &NWA,
    fut: &[Weight],
    scratch_w: &mut [Weight],
    q: &mut VecDeque<NWAStateID>,
    touched: &mut Vec<NWAStateID>,
) -> Vec<(NWAStateID, Weight)> {
    let n = nwa.states.len();

    // Initialize scratch space from all source configurations.
    for &(s, ref w) in sources {
        if s >= n { continue; }
        let prop = w & &fut[s]; // Mask with future weights to prune dead paths.
        if prop.is_empty() { continue; }

        let old_w = &scratch_w[s];
        if (old_w.clone() | &prop) != *old_w {
            if old_w.is_empty() {
                touched.push(s);
            }
            scratch_w[s] |= &prop;
        }
    }

    q.extend(touched.iter().copied());

    // Propagate weights through the epsilon-transition graph until a fixpoint is reached.
    while let Some(u) = q.pop_front() {
        let base = scratch_w[u].clone();
        if base.is_empty() { continue; }

        for &(v, ref w_eps) in &nwa.states[u].epsilons {
            if v >= n { continue; }
            let mut prop = &base & w_eps;
            if prop.is_empty() { continue; }
            prop &= &fut[v];
            if prop.is_empty() { continue; }

            let old_w = &scratch_w[v];
            if (old_w.clone() | &prop) != *old_w {
                if old_w.is_empty() {
                    touched.push(v);
                }
                scratch_w[v] |= &prop;
                q.push_back(v);
            }
        }
    }

    // Collect results and reset scratch space for the next call.
    let mut out: Vec<(NWAStateID, Weight)> = Vec::with_capacity(touched.len());
    for &i in touched.iter() {
        out.push((i, scratch_w[i].clone()));
        scratch_w[i] = Weight::zeros();
    }
    touched.clear();
    out
}

/// Helper to find an existing representative or create a new one.
/// This implements the core merging logic.
fn find_or_create(
    sig: DWAStateSignature,
    representatives: &mut Vec<DWAStateSignature>,
    dwa: &mut DWA,
    worklist: &mut VecDeque<DWAStateID>,
) -> Option<DWAStateID> {
    if sig.key.is_empty() {
        return None;
    }

    let mut best_candidate: Option<(DWAStateID, usize)> = None;
    let sig_key_set: HashSet<_> = sig.key.iter().collect();

    for (i, rep_sig) in representatives.iter().enumerate() {
        // A representative is compatible if its key is a superset of the signal's key.
        if sig.key.len() <= rep_sig.key.len() && sig_key_set.is_subset(&rep_sig.key.iter().collect()) {
            let cost = rep_sig.key.len(); // Prefer the tightest-fitting (smallest) superset.
            if best_candidate.map_or(true, |(_, best_cost)| cost < best_cost) {
                best_candidate = Some((i, cost));
            }
        }
    }

    if let Some((id, _)) = best_candidate {
        Some(id)
    } else {
        let new_id = representatives.len();
        representatives.push(sig);
        dwa.add_state();
        worklist.push_back(new_id);
        Some(new_id)
    }
}

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();
        let mut nwa = self.clone();
        nwa.simplify();

        if is_debug_level_enabled(5) {
            eprintln!("NWA after simplify:\n{}", nwa);
        }
        let result = nwa.det_fixpoint_efficient();
        if is_debug_level_enabled(5) {
            eprintln!("NWA::determinize_to_dwa result DWA stats:\n{}", result.stats());
            eprintln!("NWA::determinize_to_dwa took: {:?}", now.elapsed());
        }
        result
    }

    /// Efficient determinization algorithm that avoids state space explosion
    /// by merging structurally compatible states on-the-fly.
    fn det_fixpoint_efficient(&self) -> DWA {
        let fut = self.compute_future_weights();
        let n = self.states.len();
        if n == 0 { return DWA::new(); }

        // --- Scratch space for epsilon-closures ---
        let mut scratch_w: Vec<Weight> = vec![Weight::zeros(); n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        let mut touched: Vec<NWAStateID> = Vec::new();

        // --- State for the fixpoint iteration ---
        let mut representatives: Vec<DWAStateSignature> = Vec::new();
        let mut worklist: VecDeque<DWAStateID> = VecDeque::new();
        let mut dwa = DWA::new(); // The DWA we are building.
        dwa.states.0.clear(); // Start with no states.

        // --- Initialization ---
        let initial_powerstate = eps_closure_multi(
            &[(self.body.start_state, Weight::all())],
            self, &fut, &mut scratch_w, &mut q, &mut touched,
        );
        let initial_sig = DWAStateSignature::from_powerstate(initial_powerstate);

        if initial_sig.key.is_empty() { return DWA::new(); } // Start state is a dead end.

        let start_dwa_id = 0;
        dwa.body.start_state = start_dwa_id;
        representatives.push(initial_sig);
        dwa.add_state();
        worklist.push_back(start_dwa_id);
        
        let mut sink_id: Option<DWAStateID> = None;

        // --- Main Fixpoint Loop ---
        while let Some(source_dwa_id) = worklist.pop_front() {
            let source_sig = representatives[source_dwa_id].clone();

            // Determine the set of symbols that require explicit transitions from this state.
            let mut alphabet_of_interest = BTreeSet::new();
            for (nwa_id, _) in &source_sig.powerstate {
                alphabet_of_interest.extend(self.states[*nwa_id].transitions.keys());
                for default in &self.states[*nwa_id].default {
                    alphabet_of_interest.extend(default.exceptions.iter());
                }
            }

            // Helper to compute the successor powerstate for a given symbol (or None for default).
            let mut calculate_successor = |symbol: Option<i16>| -> DWAStateSignature {
                let mut next_configs = Vec::new();
                for (nwa_id, weight) in &source_sig.powerstate {
                    let nwa_state = &self.states[*nwa_id];
                    let mut transitioned = false;
                    if let Some(sym) = symbol {
                        if let Some(transitions) = nwa_state.transitions.get(&sym) {
                            for (target, trans_weight) in transitions {
                                next_configs.push((*target, weight & trans_weight));
                            }
                            transitioned = true;
                        }
                    }
                    if !transitioned {
                        for default in &nwa_state.default {
                            if symbol.map_or(true, |s| !default.exceptions.contains(&s)) {
                                next_configs.push((default.target, weight & &default.weight));
                            }
                        }
                    }
                }
                let closed = eps_closure_multi(&next_configs, self, &fut, &mut scratch_w, &mut q, &mut touched);
                DWAStateSignature::from_powerstate(closed)
            };

            // --- Compute and Set Transitions for the DWA State ---
            let default_sig = calculate_successor(None);
            let default_target_id =
                find_or_create(default_sig.clone(), &mut representatives, &mut dwa, &mut worklist);
            let default_weight = default_sig.powerstate.iter().fold(Weight::zeros(), |mut acc, (_, w)| { acc |= w; acc });

            if let Some(target_id) = default_target_id {
                if !default_weight.is_empty() {
                    dwa.set_default_transition(source_dwa_id, target_id, default_weight.clone()).unwrap();
                }
            }

            for &symbol in &alphabet_of_interest {
                let ex_sig = calculate_successor(Some(symbol));
                let ex_target_id =
                    find_or_create(ex_sig.clone(), &mut representatives, &mut dwa, &mut worklist);
                let ex_weight = ex_sig.powerstate.iter().fold(Weight::zeros(), |mut acc, (_, w)| { acc |= w; acc });

                if ex_target_id != default_target_id || ex_weight != default_weight {
                    let target_id = ex_target_id.unwrap_or_else(|| {
                        if sink_id.is_none() { sink_id = Some(dwa.add_state()); }
                        sink_id.unwrap()
                    });
                    dwa.add_transition(source_dwa_id, symbol, target_id, ex_weight).unwrap();
                }
            }
        }

        // --- Finalization: Set Final Weights ---
        for (dwa_id, sig) in representatives.iter().enumerate() {
            let final_weight = sig.powerstate.iter().fold(Weight::zeros(), |mut acc, (nwa_id, weight)| {
                if let Some(fw) = &self.states[*nwa_id].final_weight {
                    acc |= &(weight & fw);
                }
                acc
            });
            if !final_weight.is_empty() {
                dwa.set_final_weight(dwa_id, final_weight).unwrap();
            }
        }
        dwa
    }
}